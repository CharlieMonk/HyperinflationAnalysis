"""Data transformation and statistics functions for hyperinflation analysis."""

import pandas as pd


def normalize_series(series, base_value=100):
    """Normalize a series so it starts at base_value (e.g., 100)."""
    first_valid = series.first_valid_index()
    if first_valid is None:
        return series
    return (series / series[first_valid]) * base_value


def monthly_pct_change(series):
    """Calculate monthly percent change for a series."""
    return series.pct_change() * 100


def prepare_country_data(country_info, country_name=None, use_pct_change=True):
    """
    Prepare normalized or percent change data for a country.

    Args:
        country_info: Dict with 'data' and 'config' keys
        country_name: Optional country name to include in result
        use_pct_change: If True, return monthly % change; else return normalized values

    Returns:
        Dict with prepared series for charting
    """
    df = country_info['data'].copy()
    config = country_info['config']
    index_name = config['index_name']
    currency_name = config['currency_name']

    if use_pct_change:
        # Monthly percent change
        result = {
            'country': country_name,
            'config': config,
            'index': df.index[1:],  # Skip first row (NaN after pct_change)
            'index_local': monthly_pct_change(df[index_name]).dropna(),
            'index_gold': monthly_pct_change(df[f'{index_name}_Gold']).dropna(),
            'index_silver': monthly_pct_change(df[f'{index_name}_Silver']).dropna(),
            'index_usd': monthly_pct_change(df[f'{index_name}_USD']).dropna(),
            'index_real': monthly_pct_change(df[f'{index_name}_Real']).dropna() if f'{index_name}_Real' in df.columns else pd.Series(dtype=float),
            'index_ppp': monthly_pct_change(df[f'{index_name}_PPP']).dropna() if f'{index_name}_PPP' in df.columns else pd.Series(dtype=float),
            'currency_gold': monthly_pct_change(df[f'{currency_name}_Gold']).dropna(),
            'currency_silver': monthly_pct_change(df[f'{currency_name}_Silver']).dropna(),
            'currency_usd': monthly_pct_change(df[f'{currency_name}/USD']).dropna(),
            'currency_real': monthly_pct_change(df[f'{currency_name}_Real']).dropna() if f'{currency_name}_Real' in df.columns else pd.Series(dtype=float),
            'currency_ppp': monthly_pct_change(df[f'{currency_name}_PPP']).dropna() if f'{currency_name}_PPP' in df.columns else pd.Series(dtype=float),
        }
    else:
        # Normalized to 100
        result = {
            'country': country_name,
            'config': config,
            'index': df.index,
            'index_local': normalize_series(df[index_name]),
            'index_gold': normalize_series(df[f'{index_name}_Gold']),
            'index_silver': normalize_series(df[f'{index_name}_Silver']),
            'index_usd': normalize_series(df[f'{index_name}_USD']),
            'index_real': normalize_series(df[f'{index_name}_Real']) if f'{index_name}_Real' in df.columns and df[f'{index_name}_Real'].notna().any() else pd.Series(dtype=float, index=df.index),
            'index_ppp': normalize_series(df[f'{index_name}_PPP']) if f'{index_name}_PPP' in df.columns and df[f'{index_name}_PPP'].notna().any() else pd.Series(dtype=float, index=df.index),
            'currency_gold': normalize_series(df[f'{currency_name}_Gold']),
            'currency_silver': normalize_series(df[f'{currency_name}_Silver']),
            'currency_usd': normalize_series(df[f'{currency_name}/USD']),
            'currency_real': normalize_series(df[f'{currency_name}_Real']) if f'{currency_name}_Real' in df.columns and df[f'{currency_name}_Real'].notna().any() else pd.Series(dtype=float, index=df.index),
            'currency_ppp': normalize_series(df[f'{currency_name}_PPP']) if f'{currency_name}_PPP' in df.columns and df[f'{currency_name}_PPP'].notna().any() else pd.Series(dtype=float, index=df.index),
        }

    return result


def prepare_all_country_data(all_data, use_pct_change=True):
    """Prepare data for all countries."""
    prepared = {}
    for country, info in all_data.items():
        prepared[country] = prepare_country_data(info, country, use_pct_change)
    return prepared


def compute_performance_stats(normalized_data):
    """
    Compute performance statistics for a country's data.

    Returns dict with start/end values and percent changes.
    """
    stats = {}
    metrics = [
        ('currency_usd', 'Currency in USD'),
        ('currency_real', 'Currency (CPI-adjusted)'),
        ('currency_ppp', 'Currency (PPP-adjusted)'),
        ('currency_gold', 'Currency in Gold'),
        ('currency_silver', 'Currency in Silver'),
        ('index_usd', 'Index in USD'),
        ('index_real', 'Index (CPI-adjusted)'),
        ('index_ppp', 'Index (PPP-adjusted)'),
        ('index_gold', 'Index in Gold'),
        ('index_silver', 'Index in Silver'),
        ('index_local', 'Index (Nominal)'),
    ]

    for key, label in metrics:
        if key not in normalized_data:
            continue
        series = normalized_data[key]
        if len(series) > 0 and series.notna().any():
            start_val = series.dropna().iloc[0] if series.notna().any() else None
            end_val = series.dropna().iloc[-1] if series.notna().any() else None
            if start_val is not None and end_val is not None and start_val != 0:
                pct_change = ((end_val / start_val) - 1) * 100
                stats[label] = {
                    'start': start_val,
                    'end': end_val,
                    'change_pct': pct_change
                }

    return stats
