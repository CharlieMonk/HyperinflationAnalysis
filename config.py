"""Configuration loading and constants for hyperinflation analysis."""

import json
import os
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

# Hyperinflation periods with their stock market indices and currency tickers
# FRED series codes: https://fred.stlouisfed.org/categories/15
# Sources: Cato Institute Hyperinflation Table, IMF, World Bank
HYPERINFLATION_ECONOMIES = _CONFIG_DATA['hyperinflation_economies']

# Default color scheme for charts (loaded from config_data.json)
CHART_COLORS = _CONFIG_DATA['chart_colors']

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


def get_fred_us_cpi_series():
    """Get the FRED US CPI series code from config."""
    return _CONFIG_DATA.get('fred_us_cpi', 'CPIAUCSL')
