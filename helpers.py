"""Helper functions for hyperinflation economic analysis visualization."""

import json
import os
import pickle
import urllib.request

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime


def _load_config_data():
    """Load configuration data from JSON file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config_data.json')
    with open(config_path, 'r') as f:
        return json.load(f)


_CONFIG_DATA = _load_config_data()

# Default date range for data fetching
DEFAULT_START_DATE = datetime(1960, 1, 1)
DEFAULT_CACHE_DIR = "data_cache"


def load_or_download(cache_file, download_func, description, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load from cache if exists, otherwise download and cache."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    else:
        if verbose:
            print(f"  Downloading {description}...")
        data = download_func()
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        return data


def load_gold_silver_data(start_date=None, end_date=None, cache_dir=DEFAULT_CACHE_DIR):
    """Load Gold and Silver data from World Bank (historical) and Yahoo Finance (recent)."""
    start_date = start_date or DEFAULT_START_DATE
    end_date = end_date or datetime.now()

    wb_file = os.path.join(cache_dir, "CMO-Historical-Data-Monthly.xlsx")
    wb_url = "https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx"

    # Check old location and move if needed
    if os.path.exists("CMO-Historical-Data-Monthly.xlsx") and not os.path.exists(wb_file):
        os.makedirs(cache_dir, exist_ok=True)
        os.rename("CMO-Historical-Data-Monthly.xlsx", wb_file)

    try:
        # Load World Bank historical data
        if not os.path.exists(wb_file):
            print("  Downloading World Bank commodity data...")
            os.makedirs(cache_dir, exist_ok=True)
            urllib.request.urlretrieve(wb_url, wb_file)

        wb_data = pd.read_excel(wb_file, sheet_name='Monthly Prices', header=4)
        wb_data = wb_data.iloc[1:]  # Skip the units row
        wb_data['Date'] = pd.to_datetime(wb_data.iloc[:, 0].str.replace('M', '-'), format='%Y-%m') + pd.offsets.MonthEnd(0)
        wb_data = wb_data.set_index('Date')

        wb_gold = pd.to_numeric(wb_data['Gold'], errors='coerce')
        wb_silver = pd.to_numeric(wb_data['Silver'], errors='coerce')
        wb_end_date = wb_gold.dropna().index[-1]

        # Get recent data from Yahoo Finance to fill the gap

        def download_yf_gold():
            data = yf.download("GC=F", start=wb_end_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data

        def download_yf_silver():
            data = yf.download("SI=F", start=wb_end_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            return data

        yf_gold = load_or_download("yf_gold.pkl", download_yf_gold, "Yahoo Finance Gold", cache_dir)
        yf_silver = load_or_download("yf_silver.pkl", download_yf_silver, "Yahoo Finance Silver", cache_dir)

        # Convert Yahoo Finance to monthly
        yf_gold_monthly = yf_gold['Close'].resample('ME').last()
        yf_silver_monthly = yf_silver['Close'].resample('ME').last()

        # Combine: use World Bank for historical, Yahoo Finance for recent
        gold_monthly = wb_gold.copy()
        silver_monthly = wb_silver.copy()

        for date in yf_gold_monthly.index:
            if date > wb_end_date and pd.notna(yf_gold_monthly[date]):
                gold_monthly[date] = yf_gold_monthly[date]

        for date in yf_silver_monthly.index:
            if date > wb_end_date and pd.notna(yf_silver_monthly[date]):
                silver_monthly[date] = yf_silver_monthly[date]

        gold_monthly = gold_monthly.sort_index()
        silver_monthly = silver_monthly.sort_index()

        gold_pct = gold_monthly.pct_change(periods=6) * 100
        silver_pct = silver_monthly.pct_change(periods=6) * 100

        return gold_monthly, gold_pct, silver_monthly, silver_pct

    except Exception as e:
        print(f"Warning: Could not load Gold/Silver data: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)


# =============================================================================
# Hyperinflation Economy Data Functions
# =============================================================================

# Hyperinflation periods with their stock market indices and currency tickers
# FRED series codes: https://fred.stlouisfed.org/categories/15
# Sources: Cato Institute Hyperinflation Table, IMF, World Bank
HYPERINFLATION_ECONOMIES = _CONFIG_DATA['hyperinflation_economies']


def _date_to_cache_str(dt):
    """Convert datetime to string for cache key."""
    if hasattr(dt, 'strftime'):
        return dt.strftime('%Y%m')
    return str(dt)[:7].replace('-', '')


def load_stock_index_data(ticker, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load stock index data from Yahoo Finance."""
    cache_key = ticker.replace('^', '').replace('=', '_')
    date_range = f"{_date_to_cache_str(start_date)}_{_date_to_cache_str(end_date)}"

    def download():
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    data = load_or_download(f"index_{cache_key}_{date_range}.pkl", download, f"{ticker} index", cache_dir, verbose)
    return data


def load_currency_data(ticker, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load currency exchange rate data from Yahoo Finance (USD per local currency unit)."""
    cache_key = ticker.replace('^', '').replace('=', '_')
    date_range = f"{_date_to_cache_str(start_date)}_{_date_to_cache_str(end_date)}"

    def download():
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    data = load_or_download(f"currency_{cache_key}_{date_range}.pkl", download, f"{ticker} exchange rate", cache_dir, verbose)
    return data


def load_currency_from_fred(fred_series, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load currency exchange rate data from FRED."""
    date_range = f"{_date_to_cache_str(start_date)}_{_date_to_cache_str(end_date)}"

    def download():
        try:
            data = web.DataReader(fred_series, 'fred', start_date, end_date)
            return data
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not download {fred_series} from FRED: {e}")
            return pd.DataFrame()

    data = load_or_download(f"fred_{fred_series}_{date_range}.pkl", download, f"{fred_series} from FRED", cache_dir, verbose)
    return data


# World Bank country code mapping for CPI data
_WORLDBANK_COUNTRY_CODES = {
    'Argentina': 'ARG',
    'Turkey': 'TUR',
    'Brazil': 'BRA',
    'Russia': 'RUS',
    'Mexico': 'MEX',
    'Indonesia': 'IDN',
}

# US code for World Bank CPI
_WORLDBANK_US_CODE = 'USA'


def load_worldbank_cpi_data(cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """
    Load World Bank headline CPI monthly data.

    Returns a DataFrame with country codes as columns and dates as index.
    Data source: World Bank Global Inflation Database (hcpi_m sheet)
    """
    wb_file = os.path.join(cache_dir, "worldbank_inflation.xlsx")
    wb_url = "https://thedocs.worldbank.org/en/doc/1ad246272dbbc437c74323719506aa0c-0350012021/original/Inflation-data.xlsx"

    # Download if not exists
    if not os.path.exists(wb_file):
        if verbose:
            print("  Downloading World Bank inflation data...")
        os.makedirs(cache_dir, exist_ok=True)
        urllib.request.urlretrieve(wb_url, wb_file)

    # Load from cache if available
    cache_file = os.path.join(cache_dir, "worldbank_cpi_parsed.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Parse the Excel file
    df = pd.read_excel(wb_file, sheet_name='hcpi_m')

    # Get date columns (integers in YYYYMM format)
    date_cols = [c for c in df.columns if isinstance(c, int)]

    # Convert to proper format
    result = {}
    for _, row in df.iterrows():
        country_code = row['Country Code']
        if pd.isna(country_code):
            continue

        # Extract time series for this country
        values = {}
        for col in date_cols:
            if pd.notna(row[col]):
                # Convert YYYYMM to datetime
                year = col // 100
                month = col % 100
                if 1 <= month <= 12:
                    date = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
                    values[date] = row[col]

        if values:
            result[country_code] = pd.Series(values).sort_index()

    # Create DataFrame
    cpi_df = pd.DataFrame(result)
    cpi_df.index.name = 'Date'

    # Cache the parsed data
    with open(cache_file, 'wb') as f:
        pickle.dump(cpi_df, f)

    return cpi_df


def get_worldbank_cpi_for_country(country, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """
    Get World Bank CPI data for a specific country.

    Returns a DataFrame with CPI values, similar format to FRED data.
    """
    wb_code = _WORLDBANK_COUNTRY_CODES.get(country)
    if not wb_code:
        return pd.DataFrame()

    try:
        cpi_df = load_worldbank_cpi_data(cache_dir, verbose=False)

        if wb_code not in cpi_df.columns:
            return pd.DataFrame()

        series = cpi_df[wb_code]
        series = series[(series.index >= start_date) & (series.index <= end_date)]

        # Return as DataFrame with column name matching the pattern used by FRED
        return pd.DataFrame({f'WB_{wb_code}_CPI': series})
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load World Bank CPI for {country}: {e}")
        return pd.DataFrame()


def get_worldbank_us_cpi(start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Get World Bank US CPI data."""
    try:
        cpi_df = load_worldbank_cpi_data(cache_dir, verbose=False)

        if _WORLDBANK_US_CODE not in cpi_df.columns:
            return pd.DataFrame()

        series = cpi_df[_WORLDBANK_US_CODE]
        series = series[(series.index >= start_date) & (series.index <= end_date)]

        return pd.DataFrame({'WB_USA_CPI': series})
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load World Bank US CPI: {e}")
        return pd.DataFrame()


# DBnomics IMF country code mapping for CPI data
_DBNOMICS_COUNTRY_CODES = {
    'Argentina': 'AR',
    'Turkey': 'TR',
    'Brazil': 'BR',
    'Russia': 'RU',
    'Mexico': 'MX',
    'Indonesia': 'ID',
}

_DBNOMICS_US_CODE = 'US'


def get_dbnomics_cpi_for_country(country, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """
    Get DBnomics IMF CPI data for a specific country.

    Returns a DataFrame with CPI values. DBnomics aggregates IMF data which
    is often more up-to-date than FRED or World Bank.
    """
    db_code = _DBNOMICS_COUNTRY_CODES.get(country)
    if not db_code:
        return pd.DataFrame()

    cache_file = os.path.join(cache_dir, f"dbnomics_cpi_{db_code}.pkl")

    try:
        # Check cache first
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
        else:
            from dbnomics import fetch_series
            if verbose:
                print(f"  Fetching DBnomics CPI for {country}...")
            series_id = f"IMF/CPI/M.{db_code}.PCPI_IX"
            df = fetch_series(series_id)
            # Cache the data
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

        if len(df) == 0:
            return pd.DataFrame()

        # Convert to time series format
        df['period'] = pd.to_datetime(df['period'])
        df = df.set_index('period')
        series = df['value']
        series = series[(series.index >= start_date) & (series.index <= end_date)]

        # Align to month-end
        series.index = series.index + pd.offsets.MonthEnd(0)

        return pd.DataFrame({f'DBN_{db_code}_CPI': series})
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load DBnomics CPI for {country}: {e}")
        return pd.DataFrame()


def get_dbnomics_us_cpi(start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Get DBnomics IMF US CPI data."""
    cache_file = os.path.join(cache_dir, f"dbnomics_cpi_{_DBNOMICS_US_CODE}.pkl")

    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
        else:
            from dbnomics import fetch_series
            if verbose:
                print(f"  Fetching DBnomics US CPI...")
            series_id = f"IMF/CPI/M.{_DBNOMICS_US_CODE}.PCPI_IX"
            df = fetch_series(series_id)
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

        if len(df) == 0:
            return pd.DataFrame()

        df['period'] = pd.to_datetime(df['period'])
        df = df.set_index('period')
        series = df['value']
        series = series[(series.index >= start_date) & (series.index <= end_date)]
        series.index = series.index + pd.offsets.MonthEnd(0)

        return pd.DataFrame({'DBN_US_CPI': series})
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not load DBnomics US CPI: {e}")
        return pd.DataFrame()


def load_hyperinflation_economy_data(country, cache_dir=DEFAULT_CACHE_DIR, verbose=True, min_months=None):
    """
    Load all data for a hyperinflation economy.

    Args:
        country: Country name from HYPERINFLATION_ECONOMIES
        cache_dir: Directory for cached data files
        verbose: Print progress messages
        min_months: Minimum months of data to load. If the country's crisis period
                   is shorter, the period will be extended to reach this minimum.

    Returns a DataFrame with:
    - Stock index price (local currency)
    - Currency value in USD (inverted from USD/local rate)
    - Stock index value in USD
    - Gold price in USD (for computing index/gold and currency/gold)
    - Silver price in USD
    """
    if country not in HYPERINFLATION_ECONOMIES:
        raise ValueError(f"Unknown country: {country}. Available: {list(HYPERINFLATION_ECONOMIES.keys())}")

    config = HYPERINFLATION_ECONOMIES[country]
    start_date = pd.to_datetime(config['period_start'])
    end_date = pd.to_datetime(config['period_end']) if config['period_end'] else datetime.now()

    # Extend period if min_months is specified and period is shorter
    if min_months is not None:
        period_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        if period_months < min_months:
            # Add 1 extra month as buffer to account for data availability gaps
            months_to_add = min_months - period_months + 1
            if config['period_end'] is None:
                # Ongoing crisis: extend start date earlier
                start_date = start_date - pd.DateOffset(months=months_to_add)
            else:
                # Historical crisis: extend end date later
                end_date = end_date + pd.DateOffset(months=months_to_add)

    if verbose:
        print(f"\nLoading {country} data ({config['description']})...")

    # Load stock index
    index_data = load_stock_index_data(
        config['index_ticker'], start_date, end_date, cache_dir, verbose
    )

    # Load currency data - try FRED first (more reliable), then Yahoo Finance
    currency_data = None
    fred_series = config.get('fred_currency')

    if fred_series:
        fred_data = load_currency_from_fred(fred_series, start_date, end_date, cache_dir, verbose)
        if len(fred_data) > 0:
            currency_data = fred_data

    # Fallback to Yahoo Finance if FRED didn't work
    if currency_data is None or len(currency_data) == 0:
        currency_data = load_currency_data(
            config['currency_ticker'], start_date, end_date, cache_dir, verbose
        )

    # Load gold and silver (already have this function)
    gold_monthly, _, silver_monthly, _ = load_gold_silver_data(start_date, end_date, cache_dir)

    # Load CPI data for the country (FRED + World Bank supplement)
    cpi_data = None
    fred_cpi = config.get('fred_cpi')
    if fred_cpi:
        cpi_data = load_currency_from_fred(fred_cpi, start_date, end_date, cache_dir, verbose)

    # Supplement with World Bank CPI data where FRED is missing
    wb_cpi_data = get_worldbank_cpi_for_country(country, start_date, end_date, cache_dir, verbose)

    # Supplement with DBnomics (IMF) CPI data where World Bank is missing
    dbn_cpi_data = get_dbnomics_cpi_for_country(country, start_date, end_date, cache_dir, verbose)

    # Load US CPI for PPP calculations (FRED + World Bank + DBnomics supplement)
    us_cpi_series = _CONFIG_DATA.get('fred_us_cpi', 'CPIAUCSL')
    us_cpi_data = load_currency_from_fred(us_cpi_series, start_date, end_date, cache_dir, verbose)
    wb_us_cpi_data = get_worldbank_us_cpi(start_date, end_date, cache_dir, verbose)
    dbn_us_cpi_data = get_dbnomics_us_cpi(start_date, end_date, cache_dir, verbose)

    # Process data to monthly frequency
    if len(index_data) > 0:
        index_monthly = index_data['Close'].resample('ME').last()
    else:
        index_monthly = pd.Series(dtype=float)

    # Process currency data
    if currency_data is not None and len(currency_data) > 0:
        # Check if it's FRED data (DataFrame with single column) or Yahoo Finance data
        if isinstance(currency_data, pd.DataFrame):
            if fred_series and fred_series in currency_data.columns:
                # FRED data: value is local currency per USD
                currency_monthly = currency_data[fred_series].resample('ME').last()
                usd_per_local = 1 / currency_monthly  # Invert to get USD per local
            elif 'Close' in currency_data.columns:
                # Yahoo Finance: USD/LOCAL gives local per USD
                currency_monthly = currency_data['Close'].resample('ME').last()
                usd_per_local = 1 / currency_monthly
            else:
                # Use first column
                col = currency_data.columns[0]
                currency_monthly = currency_data[col].resample('ME').last()
                usd_per_local = 1 / currency_monthly
        else:
            usd_per_local = pd.Series(dtype=float)
    else:
        usd_per_local = pd.Series(dtype=float)

    # Process CPI data - combine FRED and World Bank
    cpi_monthly = pd.Series(dtype=float)
    if cpi_data is not None and len(cpi_data) > 0 and fred_cpi in cpi_data.columns:
        cpi_monthly = cpi_data[fred_cpi].resample('ME').last()

    # Supplement with World Bank data where FRED is missing
    wb_code = _WORLDBANK_COUNTRY_CODES.get(country)
    if wb_cpi_data is not None and len(wb_cpi_data) > 0:
        wb_col = f'WB_{wb_code}_CPI'
        if wb_col in wb_cpi_data.columns:
            wb_cpi_monthly = wb_cpi_data[wb_col].resample('ME').last()
            # Normalize World Bank data to match FRED data scale at overlap point
            if len(cpi_monthly) > 0 and cpi_monthly.notna().any():
                # Find last valid FRED value and corresponding WB value
                last_fred_idx = cpi_monthly.last_valid_index()
                if last_fred_idx is not None and last_fred_idx in wb_cpi_monthly.index:
                    scale_factor = cpi_monthly[last_fred_idx] / wb_cpi_monthly[last_fred_idx]
                    wb_cpi_monthly = wb_cpi_monthly * scale_factor
                    # Fill in missing FRED values with scaled World Bank values
                    cpi_monthly = cpi_monthly.combine_first(wb_cpi_monthly)
            else:
                # No FRED data, use World Bank directly
                cpi_monthly = wb_cpi_monthly

    # Supplement with DBnomics (IMF) data where World Bank is missing
    dbn_code = _DBNOMICS_COUNTRY_CODES.get(country)
    if dbn_cpi_data is not None and len(dbn_cpi_data) > 0:
        dbn_col = f'DBN_{dbn_code}_CPI'
        if dbn_col in dbn_cpi_data.columns:
            dbn_cpi_monthly = dbn_cpi_data[dbn_col].resample('ME').last()
            if len(cpi_monthly) > 0 and cpi_monthly.notna().any():
                last_idx = cpi_monthly.last_valid_index()
                if last_idx is not None and last_idx in dbn_cpi_monthly.index:
                    scale_factor = cpi_monthly[last_idx] / dbn_cpi_monthly[last_idx]
                    dbn_cpi_monthly = dbn_cpi_monthly * scale_factor
                    cpi_monthly = cpi_monthly.combine_first(dbn_cpi_monthly)
            else:
                cpi_monthly = dbn_cpi_monthly

    # Process US CPI data - combine FRED, World Bank, and DBnomics
    us_cpi_monthly = pd.Series(dtype=float)
    if us_cpi_data is not None and len(us_cpi_data) > 0 and us_cpi_series in us_cpi_data.columns:
        us_cpi_monthly = us_cpi_data[us_cpi_series].resample('ME').last()

    # Supplement US CPI with World Bank data where FRED is missing
    if wb_us_cpi_data is not None and len(wb_us_cpi_data) > 0:
        if 'WB_USA_CPI' in wb_us_cpi_data.columns:
            wb_us_cpi_monthly = wb_us_cpi_data['WB_USA_CPI'].resample('ME').last()
            if len(us_cpi_monthly) > 0 and us_cpi_monthly.notna().any():
                last_fred_idx = us_cpi_monthly.last_valid_index()
                if last_fred_idx is not None and last_fred_idx in wb_us_cpi_monthly.index:
                    scale_factor = us_cpi_monthly[last_fred_idx] / wb_us_cpi_monthly[last_fred_idx]
                    wb_us_cpi_monthly = wb_us_cpi_monthly * scale_factor
                    us_cpi_monthly = us_cpi_monthly.combine_first(wb_us_cpi_monthly)
            else:
                us_cpi_monthly = wb_us_cpi_monthly

    # Supplement US CPI with DBnomics data where World Bank is missing
    if dbn_us_cpi_data is not None and len(dbn_us_cpi_data) > 0:
        if 'DBN_US_CPI' in dbn_us_cpi_data.columns:
            dbn_us_cpi_monthly = dbn_us_cpi_data['DBN_US_CPI'].resample('ME').last()
            if len(us_cpi_monthly) > 0 and us_cpi_monthly.notna().any():
                last_idx = us_cpi_monthly.last_valid_index()
                if last_idx is not None and last_idx in dbn_us_cpi_monthly.index:
                    scale_factor = us_cpi_monthly[last_idx] / dbn_us_cpi_monthly[last_idx]
                    dbn_us_cpi_monthly = dbn_us_cpi_monthly * scale_factor
                    us_cpi_monthly = us_cpi_monthly.combine_first(dbn_us_cpi_monthly)
            else:
                us_cpi_monthly = dbn_us_cpi_monthly

    # Combine into DataFrame
    combined = pd.DataFrame({
        f'{config["index_name"]}': index_monthly,
        f'{config["currency_name"]}/USD': usd_per_local,
        'Gold_USD': gold_monthly,
        'Silver_USD': silver_monthly,
        'CPI': cpi_monthly,
        'US_CPI': us_cpi_monthly,
    })

    # Calculate derived metrics
    # Index in USD
    combined[f'{config["index_name"]}_USD'] = combined[f'{config["index_name"]}'] * combined[f'{config["currency_name"]}/USD']

    # Index in Gold (ounces of gold per index unit)
    combined[f'{config["index_name"]}_Gold'] = combined[f'{config["index_name"]}_USD'] / combined['Gold_USD']

    # Index in Silver
    combined[f'{config["index_name"]}_Silver'] = combined[f'{config["index_name"]}_USD'] / combined['Silver_USD']

    # Currency in Gold (ounces of gold per currency unit)
    combined[f'{config["currency_name"]}_Gold'] = combined[f'{config["currency_name"]}/USD'] / combined['Gold_USD']

    # Currency in Silver
    combined[f'{config["currency_name"]}_Silver'] = combined[f'{config["currency_name"]}/USD'] / combined['Silver_USD']

    # Index in Real terms (CPI-adjusted, local purchasing power)
    # Divide by CPI to get real value - higher CPI means lower real value
    if 'CPI' in combined.columns and combined['CPI'].notna().any():
        first_cpi = combined['CPI'].dropna().iloc[0]
        combined[f'{config["index_name"]}_Real'] = combined[f'{config["index_name"]}'] / (combined['CPI'] / first_cpi)
    else:
        combined[f'{config["index_name"]}_Real'] = pd.Series(dtype=float, index=combined.index)

    # Currency in Real terms (purchasing power relative to start)
    if 'CPI' in combined.columns and combined['CPI'].notna().any():
        first_cpi = combined['CPI'].dropna().iloc[0]
        # Currency purchasing power decreases as CPI increases
        combined[f'{config["currency_name"]}_Real'] = first_cpi / combined['CPI']
    else:
        combined[f'{config["currency_name"]}_Real'] = pd.Series(dtype=float, index=combined.index)

    # PPP-adjusted values (using relative CPI between country and US)
    # PPP adjusts for relative inflation differences
    if 'CPI' in combined.columns and 'US_CPI' in combined.columns:
        if combined['CPI'].notna().any() and combined['US_CPI'].notna().any():
            first_cpi = combined['CPI'].dropna().iloc[0]
            first_us_cpi = combined['US_CPI'].dropna().iloc[0]
            # PPP exchange rate adjustment factor
            ppp_factor = (combined['CPI'] / first_cpi) / (combined['US_CPI'] / first_us_cpi)
            # Index in PPP terms (USD adjusted for relative inflation)
            combined[f'{config["index_name"]}_PPP'] = combined[f'{config["index_name"]}_USD'] / ppp_factor
            # Currency in PPP terms
            combined[f'{config["currency_name"]}_PPP'] = combined[f'{config["currency_name"]}/USD'] / ppp_factor
        else:
            combined[f'{config["index_name"]}_PPP'] = pd.Series(dtype=float, index=combined.index)
            combined[f'{config["currency_name"]}_PPP'] = pd.Series(dtype=float, index=combined.index)
    else:
        combined[f'{config["index_name"]}_PPP'] = pd.Series(dtype=float, index=combined.index)
        combined[f'{config["currency_name"]}_PPP'] = pd.Series(dtype=float, index=combined.index)

    # Drop rows with missing key data
    combined = combined.dropna(subset=[f'{config["index_name"]}', f'{config["currency_name"]}/USD'])

    if verbose:
        print(f"  Loaded {len(combined)} months of data")

    return combined, config


def load_all_hyperinflation_data(cache_dir=DEFAULT_CACHE_DIR, verbose=True, min_months=None):
    """
    Load data for all hyperinflation economies.

    Args:
        cache_dir: Directory for cached data files
        verbose: Print progress messages
        min_months: Minimum months of data to load per country. If a country's
                   crisis period is shorter, it will be extended to this minimum.
    """
    all_data = {}

    for country in HYPERINFLATION_ECONOMIES:
        try:
            data, config = load_hyperinflation_economy_data(country, cache_dir, verbose, min_months)
            all_data[country] = {'data': data, 'config': config}
        except Exception as e:
            print(f"Warning: Could not load data for {country}: {e}")

    return all_data


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


# =============================================================================
# Chart Helper Functions for Hyperinflation Analysis
# =============================================================================

# Chart constants
LINE_WIDTH = 1.5
DEFAULT_NUM_ROWS = 3

# Default color scheme for charts (loaded from config_data.json)
CHART_COLORS = _CONFIG_DATA['chart_colors']


def create_single_country_chart(data, colors=None, use_pct_change=True):
    """
    Create a detailed chart for a single country's hyperinflation data.

    Two subplots:
    - Row 1: Currency metrics (USD, CPI-adjusted, PPP-adjusted, gold, silver)
    - Row 2: Index metrics (USD, CPI-adjusted, PPP-adjusted, gold, silver)

    Args:
        data: Dict with series for the country (from prepare_country_data)
        colors: Optional color scheme dict
        use_pct_change: If True, data is monthly % change; else normalized values

    Returns:
        Plotly figure object
    """
    colors = colors or CHART_COLORS
    country = data['country']
    config = data['config']
    idx = data['index']

    # Determine labels and scales based on data type
    if use_pct_change:
        currency_title = f'{country}: {config["currency_name"]} Monthly % Change'
        index_title = f'{country}: {config["index_name"]} Monthly % Change'
        y_type = 'linear'
        ref_line = 0
        hover_suffix = '%'
    else:
        currency_title = f'{country}: {config["currency_name"]} Value (Normalized to 100)'
        index_title = f'{country}: {config["index_name"]} Value (Normalized to 100)'
        y_type = 'linear'
        ref_line = 100
        hover_suffix = ''

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.5, 0.5],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(currency_title, index_title)
    )

    # Row 1: Currency metrics (dotted lines)
    # USD (visible)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['currency_usd'],
            name=f'{config["currency_name"]} / USD',
            line=dict(color=colors['usd'], width=LINE_WIDTH, dash='dot'),
            hovertemplate=f'{config["currency_name"]} in USD: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=1, col=1
    )
    # CPI-adjusted (visible)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['currency_real'],
            name=f'{config["currency_name"]} (CPI-adj)',
            line=dict(color=colors['cpi'], width=LINE_WIDTH, dash='dot'),
            hovertemplate=f'{config["currency_name"]} CPI-adj: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=1, col=1
    )
    # PPP-adjusted (visible)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['currency_ppp'],
            name=f'{config["currency_name"]} (PPP-adj)',
            line=dict(color=colors['ppp'], width=LINE_WIDTH, dash='dot'),
            hovertemplate=f'{config["currency_name"]} PPP-adj: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=1, col=1
    )
    # Gold (hidden by default)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['currency_gold'],
            name=f'{config["currency_name"]} / Gold',
            line=dict(color=colors['gold'], width=LINE_WIDTH, dash='dot'),
            hovertemplate=f'{config["currency_name"]} in Gold: %{{y:.1f}}{hover_suffix}<extra></extra>',
            visible='legendonly',
        ),
        row=1, col=1
    )
    # Silver (hidden by default)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['currency_silver'],
            name=f'{config["currency_name"]} / Silver',
            line=dict(color=colors['silver'], width=LINE_WIDTH, dash='dot'),
            hovertemplate=f'{config["currency_name"]} in Silver: %{{y:.1f}}{hover_suffix}<extra></extra>',
            visible='legendonly',
        ),
        row=1, col=1
    )

    # Row 2: Index metrics (solid lines)
    # USD (visible)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_usd'],
            name=f'{config["index_name"]} / USD',
            line=dict(color=colors['usd'], width=LINE_WIDTH),
            hovertemplate=f'{config["index_name"]} in USD: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=2, col=1
    )
    # CPI-adjusted (visible)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_real'],
            name=f'{config["index_name"]} (CPI-adj)',
            line=dict(color=colors['cpi'], width=LINE_WIDTH),
            hovertemplate=f'{config["index_name"]} CPI-adj: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=2, col=1
    )
    # PPP-adjusted (visible)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_ppp'],
            name=f'{config["index_name"]} (PPP-adj)',
            line=dict(color=colors['ppp'], width=LINE_WIDTH),
            hovertemplate=f'{config["index_name"]} PPP-adj: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=2, col=1
    )
    # Gold (hidden by default)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_gold'],
            name=f'{config["index_name"]} / Gold',
            line=dict(color=colors['gold'], width=LINE_WIDTH),
            hovertemplate=f'{config["index_name"]} in Gold: %{{y:.1f}}{hover_suffix}<extra></extra>',
            visible='legendonly',
        ),
        row=2, col=1
    )
    # Silver (hidden by default)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_silver'],
            name=f'{config["index_name"]} / Silver',
            line=dict(color=colors['silver'], width=LINE_WIDTH),
            hovertemplate=f'{config["index_name"]} in Silver: %{{y:.1f}}{hover_suffix}<extra></extra>',
            visible='legendonly',
        ),
        row=2, col=1
    )
    # Index Nominal (hidden by default)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_local'],
            name=f'{config["index_name"]} (Nominal)',
            line=dict(color=colors.get(country, '#ffffff'), width=LINE_WIDTH),
            hovertemplate=f'{config["index_name"]} Nominal: %{{y:.1f}}{hover_suffix}<extra></extra>',
            visible='legendonly',
        ),
        row=2, col=1
    )

    # Reference lines
    for row in [1, 2]:
        fig.add_hline(
            y=ref_line, line_dash='dash',
            line_color='rgba(255, 255, 255, 0.3)',
            line_width=1, row=row, col=1
        )

    # Subplot title styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color=colors['text'])

    # Layout
    fig.update_layout(
        height=550,
        hovermode='x unified',
        paper_bgcolor=colors['paper'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'], size=10),
        legend=dict(
            orientation='v',
            yanchor='top', y=1,
            xanchor='left', x=1.02,
            font=dict(size=9, color=colors['text']),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(t=60, l=55, r=180, b=35),
        hoverlabel=dict(
            bgcolor=colors['paper'],
            font_size=11,
            font_color=colors['text']
        ),
    )

    # Y-axes
    fig.update_yaxes(
        title_text='Value',
        title_font=dict(size=10, color=colors['text']),
        gridcolor=colors['grid'],
        type=y_type,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='Value',
        title_font=dict(size=10, color=colors['text']),
        gridcolor=colors['grid'],
        type=y_type,
        row=2, col=1
    )

    # X-axes - show month and year in hover
    fig.update_xaxes(
        tickformat='%b %Y',
        hoverformat='%b %Y',
        gridcolor=colors['grid'],
        row=1, col=1
    )
    fig.update_xaxes(
        tickformat='%b %Y',
        hoverformat='%b %Y',
        gridcolor=colors['grid'],
        title_text='Date',
        title_font=dict(size=10),
        row=2, col=1
    )

    return fig


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


def plot_aggregate_chart(prepared_data, colors=None, use_pct_change=False):
    """
    Create aggregate chart comparing all hyperinflation economies.
    X-axis shows months from crisis start (0 = first month), so all countries overlap.

    Args:
        prepared_data: Dict of country -> prepared data from prepare_all_country_data
        colors: Optional color scheme dict (defaults to CHART_COLORS)
        use_pct_change: If True, show monthly % change; else normalized values

    Returns:
        Plotly figure object
    """
    colors = colors or CHART_COLORS
    num_rows = 5  # USD, CPI-adjusted, PPP-adjusted, Gold, Silver

    if use_pct_change:
        titles = (
            'Monthly % Change in USD Terms',
            'Monthly % Change (CPI-adjusted)',
            'Monthly % Change (PPP-adjusted)',
            'Monthly % Change in Gold Terms',
            'Monthly % Change in Silver Terms',
        )
        y_type = 'linear'
        ref_line = 0
        y_labels = ['% Change', '% Change', '% Change', '% Change', '% Change']
    else:
        titles = (
            'Value in USD (Normalized to 100)',
            'Value CPI-adjusted (Normalized to 100)',
            'Value PPP-adjusted (Normalized to 100)',
            'Value in Gold (Normalized to 100)',
            'Value in Silver (Normalized to 100)',
        )
        y_type = 'linear'
        ref_line = 100
        y_labels = ['Value', 'Value', 'Value', 'Value', 'Value']

    fig = make_subplots(
        rows=num_rows, cols=1,
        row_heights=[0.25, 0.20, 0.20, 0.175, 0.175],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=titles
    )

    max_months = 0
    default_visible_countries = {'Russia', 'Brazil', 'Argentina'}

    for country, data in prepared_data.items():
        config = data['config']
        color = colors.get(country, '#ffffff')
        # Convert to months from start (0, 1, 2, ...)
        months = list(range(len(data['index'])))
        max_months = max(max_months, len(months))

        # Determine visibility based on default countries
        is_default_visible = country in default_visible_countries
        index_visible = True if is_default_visible else 'legendonly'
        currency_visible = True if is_default_visible else 'legendonly'

        # Legend groups for synchronized toggling across subplots
        currency_group = f"{country}_currency"
        index_group = f"{country}_index"

        # Row 1: USD-denominated (visible by default)
        fig.add_trace(
            go.Scatter(
                x=months, y=data['currency_usd'],
                name=f"{country} {config['currency_name']}",
                legendgroup=currency_group,
                line=dict(color=color, width=LINE_WIDTH, dash='dot'),
                hovertemplate=f"{country} {config['currency_name']}/USD: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=currency_visible,
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=months, y=data['index_usd'],
                name=f"{country} {config['index_name']}",
                legendgroup=index_group,
                line=dict(color=color, width=LINE_WIDTH),
                hovertemplate=f"{country} {config['index_name']}/USD: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=index_visible,
            ),
            row=1, col=1
        )

        # Row 2: CPI-adjusted (visible by default)
        fig.add_trace(
            go.Scatter(
                x=months, y=data['currency_real'],
                name=f"{config['currency_name']}/CPI",
                legendgroup=currency_group,
                line=dict(color=color, width=LINE_WIDTH, dash='dot'),
                showlegend=False,
                hovertemplate=f"{country} {config['currency_name']} CPI-adj: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=currency_visible,
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=months, y=data['index_real'],
                name=f"{config['index_name']}/CPI",
                legendgroup=index_group,
                line=dict(color=color, width=LINE_WIDTH),
                showlegend=False,
                hovertemplate=f"{country} {config['index_name']} CPI-adj: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=index_visible,
            ),
            row=2, col=1
        )

        # Row 3: PPP-adjusted (visible by default)
        fig.add_trace(
            go.Scatter(
                x=months, y=data['currency_ppp'],
                name=f"{config['currency_name']}/PPP",
                legendgroup=currency_group,
                line=dict(color=color, width=LINE_WIDTH, dash='dot'),
                showlegend=False,
                hovertemplate=f"{country} {config['currency_name']} PPP-adj: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=currency_visible,
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=months, y=data['index_ppp'],
                name=f"{config['index_name']}/PPP",
                legendgroup=index_group,
                line=dict(color=color, width=LINE_WIDTH),
                showlegend=False,
                hovertemplate=f"{country} {config['index_name']} PPP-adj: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=index_visible,
            ),
            row=3, col=1
        )

        # Row 4: Gold-denominated (visible by default)
        fig.add_trace(
            go.Scatter(
                x=months, y=data['currency_gold'],
                name=f"{config['currency_name']}/Gold",
                legendgroup=currency_group,
                line=dict(color=color, width=LINE_WIDTH, dash='dot'),
                showlegend=False,
                hovertemplate=f"{country} {config['currency_name']}/Gold: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=currency_visible,
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=months, y=data['index_gold'],
                name=f"{config['index_name']}/Gold",
                legendgroup=index_group,
                line=dict(color=color, width=LINE_WIDTH),
                showlegend=False,
                hovertemplate=f"{country} {config['index_name']}/Gold: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=index_visible,
            ),
            row=4, col=1
        )

        # Row 5: Silver-denominated (visible by default)
        fig.add_trace(
            go.Scatter(
                x=months, y=data['currency_silver'],
                name=f"{config['currency_name']}/Silver",
                legendgroup=currency_group,
                line=dict(color=color, width=LINE_WIDTH, dash='dot'),
                showlegend=False,
                hovertemplate=f"{country} {config['currency_name']}/Silver: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=currency_visible,
            ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=months, y=data['index_silver'],
                name=f"{config['index_name']}/Silver",
                legendgroup=index_group,
                line=dict(color=color, width=LINE_WIDTH),
                showlegend=False,
                hovertemplate=f"{country} {config['index_name']}/Silver: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=index_visible,
            ),
            row=5, col=1
        )

    # Reference lines
    for row in range(1, num_rows + 1):
        fig.add_hline(
            y=ref_line, line_dash="dash",
            line_color=colors['zero_line'],
            line_width=1, row=row, col=1
        )

    x_range = [0, max_months] if max_months > 0 else None

    # Subplot title styling
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color=colors['text'])

    # Layout
    fig.update_layout(
        height=950,
        hovermode='x unified',
        paper_bgcolor=colors['paper'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'], size=10),
        title=dict(y=0.99, yanchor='top'),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.01,
            xanchor='center', x=0.5,
            font=dict(size=8, color=colors['text']),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(t=120, l=55, r=55, b=35),
        hoverlabel=dict(bgcolor=colors['paper'], font_size=11, font_color=colors['text']),
        spikedistance=-1,
    )

    # Y-axes
    yaxis_base = dict(gridcolor=colors['grid'], automargin=True, color=colors['text'], type=y_type)
    fig.update_yaxes(title_text=y_labels[0], title_font=dict(size=10, color=colors['usd']), row=1, col=1, **yaxis_base)
    fig.update_yaxes(title_text=y_labels[1], title_font=dict(size=10, color=colors['cpi']), row=2, col=1, **yaxis_base)
    fig.update_yaxes(title_text=y_labels[2], title_font=dict(size=10, color=colors['ppp']), row=3, col=1, **yaxis_base)
    fig.update_yaxes(title_text=y_labels[3], title_font=dict(size=10, color=colors['gold']), row=4, col=1, **yaxis_base)
    fig.update_yaxes(title_text=y_labels[4], title_font=dict(size=10, color=colors['silver']), row=5, col=1, **yaxis_base)

    # X-axes - months from crisis start
    xaxis_base = dict(
        range=x_range,
        nticks=10,
        tickangle=0,
        showticklabels=True,
        automargin=True,
        gridcolor=colors['grid'],
        color=colors['text'],
    )
    for row in range(1, num_rows + 1):
        extra = {'title_text': 'Months from Crisis Start', 'title_font': dict(size=10, color=colors['text'])} if row == num_rows else {}
        fig.update_xaxes(row=row, col=1, **xaxis_base, **extra)

    return fig


def show_country_analysis(data, colors=None, use_pct_change=False):
    """
    Display chart and print performance stats for a single country.

    Args:
        data: Prepared data dict for a single country (from prepare_country_data)
        colors: Optional color scheme dict
        use_pct_change: If True, show monthly % change; else normalized values

    Returns:
        Plotly figure object
    """
    colors = colors or CHART_COLORS

    fig = create_single_country_chart(data, colors, use_pct_change)
    fig.show()

    stats = compute_performance_stats(data)
    print("\nPerformance Summary (Normalized Start=100):")
    for metric, values in stats.items():
        print(f"  {metric:20s}: {values['start']:8.1f} -> {values['end']:8.1f}  ({values['change_pct']:+.1f}%)")


def print_data_summary(all_data):
    """Print a summary of loaded hyperinflation data."""
    print("\n" + "=" * 70)
    print("Data Summary")
    print("=" * 70)
    for country, info in all_data.items():
        df = info['data']
        config = info['config']
        print(f"\n{country} ({config['description']})")
        print(f"  Period: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
        print(f"  Data points: {len(df)}")
