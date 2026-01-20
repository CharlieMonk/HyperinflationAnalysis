"""Data fetching functions for hyperinflation analysis."""

import os
import pickle
import urllib.request
from datetime import datetime

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

from .config import (
    DEFAULT_START_DATE,
    DEFAULT_CACHE_DIR,
    HYPERINFLATION_ECONOMIES,
    _WORLDBANK_COUNTRY_CODES,
    _WORLDBANK_US_CODE,
    _DBNOMICS_COUNTRY_CODES,
    _DBNOMICS_US_CODE,
    get_fred_us_cpi_series,
)
from .cache import load_or_download, _date_to_cache_str


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


def _supplement_cpi_series(primary_series, supplementary_sources):
    """Supplement a CPI series with additional sources where primary is missing.

    Args:
        primary_series: pd.Series with primary CPI data
        supplementary_sources: List of (DataFrame, column_name) tuples to fill gaps

    Returns:
        Combined series with gaps filled from supplementary sources
    """
    result = primary_series.copy() if len(primary_series) > 0 else pd.Series(dtype=float)

    for supp_df, col_name in supplementary_sources:
        if supp_df is None or len(supp_df) == 0:
            continue
        if col_name not in supp_df.columns:
            continue

        supp_monthly = supp_df[col_name].resample('ME').last()

        if len(result) > 0 and result.notna().any():
            # Normalize supplementary data to match primary data at overlap point
            last_idx = result.last_valid_index()
            if last_idx is not None and last_idx in supp_monthly.index:
                scale_factor = result[last_idx] / supp_monthly[last_idx]
                supp_monthly = supp_monthly * scale_factor
                result = result.combine_first(supp_monthly)
        else:
            # No primary data, use supplementary directly
            result = supp_monthly

    return result


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

    # Load CPI data for the country (FRED + World Bank + DBnomics supplement)
    cpi_data = None
    fred_cpi = config.get('fred_cpi')
    if fred_cpi:
        cpi_data = load_currency_from_fred(fred_cpi, start_date, end_date, cache_dir, verbose)

    # Supplement with World Bank CPI data where FRED is missing
    wb_cpi_data = get_worldbank_cpi_for_country(country, start_date, end_date, cache_dir, verbose)

    # Supplement with DBnomics (IMF) CPI data where World Bank is missing
    dbn_cpi_data = get_dbnomics_cpi_for_country(country, start_date, end_date, cache_dir, verbose)

    # Load US CPI for PPP calculations (FRED + World Bank + DBnomics supplement)
    us_cpi_series = get_fred_us_cpi_series()
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

    # Process CPI data using the helper function
    wb_code = _WORLDBANK_COUNTRY_CODES.get(country)
    dbn_code = _DBNOMICS_COUNTRY_CODES.get(country)

    # Start with FRED CPI if available
    if cpi_data is not None and len(cpi_data) > 0 and fred_cpi in cpi_data.columns:
        cpi_monthly = cpi_data[fred_cpi].resample('ME').last()
    else:
        cpi_monthly = pd.Series(dtype=float)

    # Supplement with World Bank and DBnomics
    cpi_monthly = _supplement_cpi_series(
        cpi_monthly,
        [
            (wb_cpi_data, f'WB_{wb_code}_CPI') if wb_code else (None, ''),
            (dbn_cpi_data, f'DBN_{dbn_code}_CPI') if dbn_code else (None, ''),
        ]
    )

    # Process US CPI data using the helper function
    if us_cpi_data is not None and len(us_cpi_data) > 0 and us_cpi_series in us_cpi_data.columns:
        us_cpi_monthly = us_cpi_data[us_cpi_series].resample('ME').last()
    else:
        us_cpi_monthly = pd.Series(dtype=float)

    # Supplement US CPI with World Bank and DBnomics
    us_cpi_monthly = _supplement_cpi_series(
        us_cpi_monthly,
        [
            (wb_us_cpi_data, 'WB_USA_CPI'),
            (dbn_us_cpi_data, 'DBN_US_CPI'),
        ]
    )

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

    # Convert month-end dates to month-start for proper gridline alignment
    # (e.g., 2018-12-31 -> 2018-12-01 so Dec 2018 appears left of Jan 2019 gridline)
    combined.index = combined.index.to_period('M').to_timestamp()

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
