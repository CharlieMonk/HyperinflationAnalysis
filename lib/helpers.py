"""Helper functions for hyperinflation economic analysis visualization.

This module provides backward compatibility by re-exporting all functions
from the modular structure. For new code, prefer importing directly from
the specific modules: config, cache, loaders, processing, charts.
"""

# Configuration and constants
from .config import (
    DEFAULT_START_DATE,
    DEFAULT_CACHE_DIR,
    HYPERINFLATION_ECONOMIES,
    CHART_COLORS,
)

# Caching utilities
from .cache import load_or_download

# Data loading functions
from .loaders import (
    load_gold_silver_data,
    load_stock_index_data,
    load_currency_data,
    load_currency_from_fred,
    load_worldbank_cpi_data,
    get_worldbank_cpi_for_country,
    get_worldbank_us_cpi,
    get_dbnomics_cpi_for_country,
    get_dbnomics_us_cpi,
    load_hyperinflation_economy_data,
    load_all_hyperinflation_data,
)

# Data processing functions
from .processing import (
    normalize_series,
    monthly_pct_change,
    prepare_country_data,
    prepare_all_country_data,
    compute_performance_stats,
)
