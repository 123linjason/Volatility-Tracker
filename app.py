import requests
import pandas as pd
import numpy as np

def fetch_36_quarters_sec(ticker_symbol):
    """
    Fetches multi-year quarterly financials directly from SEC EDGAR XBRL API.
    """
    headers = {'User-Agent': 'YourName contact@yourdomain.com'}
    
    # 1. Map ticker to SEC CIK number
    tickers_json = requests.get(
        "https://www.sec.gov/files/company_tickers.json", 
        headers=headers
    ).json()
    
    cik = None
    for entry in tickers_json.values():
        if entry['ticker'].upper() == ticker_symbol.upper():
            cik = str(entry['cik_str']).zfill(10)
            break
            
    if not cik:
        return pd.DataFrame()

    # 2. Query SEC Company Facts API
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    res = requests.get(facts_url, headers=headers)
    if res.status_code != 200:
        return pd.DataFrame()
        
    facts = res.json().get('facts', {}).get('us-gaap', {})
    
    # 3. Parse Revenue and Diluted EPS series
    rev_data = facts.get('Revenues', {}).get('units', {}).get('USD', [])
    if not rev_data:
        rev_data = facts.get('RevenueFromContractWithCustomerExcludingAssessedTax', {}).get('units', {}).get('USD', [])
        
    eps_data = facts.get('EarningsPerShareDiluted', {}).get('units', {}).get('USD/shares', [])

    # Filter 10-Q (Quarterly) entries
    q_revs = [item for item in rev_data if item.get('form') == '10-Q' and item.get('fp', '').startswith('Q')]
    q_eps = [item for item in eps_data if item.get('form') == '10-Q' and item.get('fp', '').startswith('Q')]

    df_rev = pd.DataFrame(q_revs)[['end', 'val', 'fp', 'fy']].rename(columns={'val': 'Revenue'})
    df_eps = pd.DataFrame(q_eps)[['end', 'val']].rename(columns={'val': 'EPS'})

    # Merge on period end date
    df_merged = pd.merge(df_rev, df_eps, on='end', how='inner')
    df_merged['Date'] = pd.to_datetime(df_merged['end'])
    
    # Clean duplicates and sort chronologically
    df_merged = df_merged.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # Slice up to the last 36 quarters
    return df_merged.tail(36)
