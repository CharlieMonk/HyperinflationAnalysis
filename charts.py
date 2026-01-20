"""Backward compatibility re-exports for charts module."""

from econ_charts import EconChart, enable_unified_spikeline
from hyperinflation_charts import (
    LINE_WIDTH,
    create_single_country_chart,
    plot_aggregate_chart,
    show_country_analysis,
    print_data_summary,
)

# Legacy constant
DEFAULT_NUM_ROWS = 3

__all__ = [
    'EconChart',
    'enable_unified_spikeline',
    'LINE_WIDTH',
    'DEFAULT_NUM_ROWS',
    'create_single_country_chart',
    'plot_aggregate_chart',
    'show_country_analysis',
    'print_data_summary',
]
