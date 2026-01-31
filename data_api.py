import pandas as pd
from typing import List, Dict, Any, Union
import requests
import os
from dotenv import load_dotenv

load_dotenv()


def fetch_full_url(api_url: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch data from the income tax API.
    
    Args:
        api_url: The API endpoint URL
        
    Returns:
        Either a list of records or a dict containing records
    """
    api_key = os.getenv('INCOME_TAX_API_KEY')
    if not api_key:
        raise ValueError("INCOME_TAX_API_KEY not found in environment variables")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from API: {e}")


def normalize_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert API records to a normalized DataFrame.
    
    Args:
        records: List of record dictionaries from the API
        
    Returns:
        Normalized pandas DataFrame
    """
    if not records:
        return pd.DataFrame()
    
    # Convert to DataFrame and handle missing fields
    df = pd.DataFrame(records)
    
    # Ensure required columns exist with proper types
    required_columns = [
        'taxpayer_id', 'period_end_date', 'income', 'tax_paid', 
        'deductions', 'salary_reported', 'salary_employer', 
        'filed_date', 'amendment_count', 'is_fraud'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # Convert numeric columns
    numeric_columns = ['income', 'tax_paid', 'deductions', 'salary_reported', 
                      'salary_employer', 'amendment_count', 'is_fraud']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert date columns
    date_columns = ['period_end_date', 'filed_date']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df
