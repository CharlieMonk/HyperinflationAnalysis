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


def load_stock_index_data(ticker, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load stock index data from Yahoo Finance."""
    cache_key = ticker.replace('^', '').replace('=', '_')

    def download():
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    data = load_or_download(f"index_{cache_key}.pkl", download, f"{ticker} index", cache_dir, verbose)
    return data


def load_currency_data(ticker, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load currency exchange rate data from Yahoo Finance (USD per local currency unit)."""
    cache_key = ticker.replace('^', '').replace('=', '_')

    def download():
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data

    data = load_or_download(f"currency_{cache_key}.pkl", download, f"{ticker} exchange rate", cache_dir, verbose)
    return data


def load_currency_from_fred(fred_series, start_date, end_date, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load currency exchange rate data from FRED."""
    def download():
        try:
            data = web.DataReader(fred_series, 'fred', start_date, end_date)
            return data
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not download {fred_series} from FRED: {e}")
            return pd.DataFrame()

    data = load_or_download(f"fred_{fred_series}.pkl", download, f"{fred_series} from FRED", cache_dir, verbose)
    return data


def load_hyperinflation_economy_data(country, cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """
    Load all data for a hyperinflation economy.

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

    # Combine into DataFrame
    combined = pd.DataFrame({
        f'{config["index_name"]}': index_monthly,
        f'{config["currency_name"]}/USD': usd_per_local,
        'Gold_USD': gold_monthly,
        'Silver_USD': silver_monthly,
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

    # Drop rows with missing key data
    combined = combined.dropna(subset=[f'{config["index_name"]}', f'{config["currency_name"]}/USD'])

    if verbose:
        print(f"  Loaded {len(combined)} months of data")

    return combined, config


def load_all_hyperinflation_data(cache_dir=DEFAULT_CACHE_DIR, verbose=True):
    """Load data for all hyperinflation economies."""
    all_data = {}

    for country in HYPERINFLATION_ECONOMIES:
        try:
            data, config = load_hyperinflation_economy_data(country, cache_dir, verbose)
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
            'currency_gold': monthly_pct_change(df[f'{currency_name}_Gold']).dropna(),
            'currency_silver': monthly_pct_change(df[f'{currency_name}_Silver']).dropna(),
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
            'currency_gold': normalize_series(df[f'{currency_name}_Gold']),
            'currency_silver': normalize_series(df[f'{currency_name}_Silver']),
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
    color = colors.get(country, '#ffffff')
    idx = data['index']

    # Determine labels and scales based on data type
    if use_pct_change:
        precious_title = f'{country}: Monthly % Change in Precious Metals'
        nominal_title = f'{country}: Monthly % Change - {config["index_name"]} ({config["currency_name"]})'
        y1_title = 'Monthly % Change'
        y2_title = 'Monthly % Change'
        y_type = 'linear'
        ref_line = 0  # Reference at 0% for percent change
        hover_suffix = '%'
    else:
        precious_title = f'{country}: Value in Precious Metals (Normalized to 100)'
        nominal_title = f'{country}: Nominal {config["index_name"]} ({config["currency_name"]})'
        y1_title = 'Value (Normalized)'
        y2_title = 'Index Value'
        y_type = 'linear'
        ref_line = 100
        hover_suffix = ''

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(precious_title, nominal_title)
    )

    # Row 1: Gold and Silver denominated values
    # Currency in Gold (dotted)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['currency_gold'],
            name=f'{config["currency_name"]} / Gold',
            line=dict(color=colors['gold'], dash='dot', width=LINE_WIDTH),
            hovertemplate=f'{config["currency_name"]} in Gold: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=1, col=1
    )
    # Currency in Silver (dotted)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['currency_silver'],
            name=f'{config["currency_name"]} / Silver',
            line=dict(color=colors['silver'], dash='dot', width=LINE_WIDTH),
            hovertemplate=f'{config["currency_name"]} in Silver: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=1, col=1
    )
    # Index in Gold (solid)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_gold'],
            name=f'{config["index_name"]} / Gold',
            line=dict(color=colors['gold'], dash='solid', width=LINE_WIDTH),
            hovertemplate=f'{config["index_name"]} in Gold: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=1, col=1
    )
    # Index in Silver (solid)
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_silver'],
            name=f'{config["index_name"]} / Silver',
            line=dict(color=colors['silver'], dash='solid', width=LINE_WIDTH),
            hovertemplate=f'{config["index_name"]} in Silver: %{{y:.1f}}{hover_suffix}<extra></extra>',
        ),
        row=1, col=1
    )

    # Row 2: Nominal index in local currency
    fig.add_trace(
        go.Scatter(
            x=idx, y=data['index_local'],
            name=f'{config["index_name"]} (Nominal)',
            line=dict(color=color, width=LINE_WIDTH),
            fill='tozeroy' if not use_pct_change else None,
            fillcolor=f'rgba{tuple(list(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}' if not use_pct_change else None,
            hovertemplate=f'{config["index_name"]}: %{{y:.1f}}{hover_suffix}<extra></extra>',
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
        height=450,
        hovermode='x unified',
        paper_bgcolor=colors['paper'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'], size=10),
        legend=dict(
            orientation='h',
            yanchor='top', y=1.15,
            xanchor='center', x=0.5,
            font=dict(size=9, color=colors['text']),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(t=70, l=55, r=40, b=35),
        hoverlabel=dict(
            bgcolor=colors['paper'],
            font_size=11,
            font_color=colors['text']
        ),
    )

    # Y-axes
    fig.update_yaxes(
        title_text=y1_title,
        title_font=dict(size=10),
        gridcolor=colors['grid'],
        type=y_type,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text=y2_title,
        title_font=dict(size=10),
        gridcolor=colors['grid'],
        type=y_type,
        row=2, col=1
    )

    # X-axes
    fig.update_xaxes(
        tickformat='%Y',
        gridcolor=colors['grid'],
        row=1, col=1
    )
    fig.update_xaxes(
        tickformat='%Y',
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
        ('currency_gold', 'Currency in Gold'),
        ('currency_silver', 'Currency in Silver'),
        ('index_gold', 'Index in Gold'),
        ('index_silver', 'Index in Silver'),
        ('index_local', 'Index (Nominal)'),
    ]

    for key, label in metrics:
        series = normalized_data[key]
        if len(series) > 0:
            start_val = series.iloc[0]
            end_val = series.iloc[-1]
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
    num_rows = DEFAULT_NUM_ROWS

    if use_pct_change:
        titles = (
            'Monthly % Change in Gold Terms',
            'Monthly % Change in Silver Terms',
            'Monthly % Change - Stock Index (Local Currency)'
        )
        y_type = 'linear'
        ref_line = 0
        y_labels = ['% Change', '% Change', '% Change']
    else:
        titles = (
            'Value in Gold (Normalized to 100)',
            'Value in Silver (Normalized to 100)',
            'Stock Index in Local Currency (Normalized to 100)'
        )
        y_type = 'linear'
        ref_line = 100
        y_labels = ['Value', 'Value', 'Index']

    fig = make_subplots(
        rows=num_rows, cols=1,
        row_heights=[0.40, 0.30, 0.30],
        shared_xaxes=True,
        vertical_spacing=0.08,
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

        # Row 1: Gold-denominated
        fig.add_trace(
            go.Scatter(
                x=months, y=data['currency_gold'],
                name=f"{country} {config['currency_name']}",
                legendgroup=currency_group,
                line=dict(color=color, width=LINE_WIDTH, dash='dot'),
                hovertemplate=f"{country} {config['currency_name']}/Gold: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=currency_visible,
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=months, y=data['index_gold'],
                name=f"{country} {config['index_name']}",
                legendgroup=index_group,
                line=dict(color=color, width=LINE_WIDTH),
                hovertemplate=f"{country} {config['index_name']}/Gold: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=index_visible,
            ),
            row=1, col=1
        )

        # Row 2: Silver-denominated
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
            row=2, col=1
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
            row=2, col=1
        )

        # Row 3: Nominal local currency
        fig.add_trace(
            go.Scatter(
                x=months, y=data['index_local'],
                name=f"{config['index_name']} ({config['currency_name']})",
                legendgroup=index_group,
                line=dict(color=color, width=LINE_WIDTH),
                showlegend=False,
                hovertemplate=f"{country} {config['index_name']}: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
                visible=index_visible,
            ),
            row=3, col=1
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
        height=750,
        hovermode='x unified',
        paper_bgcolor=colors['paper'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'], size=10),
        title=dict(y=0.98, yanchor='top'),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='center', x=0.5,
            font=dict(size=8, color=colors['text']),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(t=100, l=55, r=55, b=35),
        hoverlabel=dict(bgcolor=colors['paper'], font_size=11, font_color=colors['text']),
        spikedistance=-1,
    )

    # Y-axes
    yaxis_base = dict(gridcolor=colors['grid'], automargin=True, color=colors['text'], type=y_type)
    fig.update_yaxes(title_text=y_labels[0], title_font=dict(size=10, color=colors['gold']), row=1, col=1, **yaxis_base)
    fig.update_yaxes(title_text=y_labels[1], title_font=dict(size=10, color=colors['silver']), row=2, col=1, **yaxis_base)
    fig.update_yaxes(title_text=y_labels[2], title_font=dict(size=10, color=colors['text']), row=3, col=1, **yaxis_base)

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
