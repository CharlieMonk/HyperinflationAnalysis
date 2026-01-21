"""Hyperinflation-specific chart wrappers using EconChart."""

from econ_charts import EconChart
from .config import CHART_COLORS
from .processing import compute_performance_stats


# Chart constants
LINE_WIDTH = 1.5


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
        ref_line = 0
        hover_suffix = '%'
    else:
        currency_title = f'{country}: {config["currency_name"]} Value (Normalized to 100)'
        index_title = f'{country}: {config["index_name"]} Value (Normalized to 100)'
        ref_line = 100
        hover_suffix = ''

    chart = EconChart(
        num_rows=2,
        row_heights=[0.5, 0.5],
        subplot_titles=(currency_title, index_title),
        colors=colors,
        shared_xaxes=True,
        vertical_spacing=0.08,
        height=550,
    )

    # Row 1: Currency metrics (dotted lines)
    # USD (visible)
    chart.add_line(
        row=1, x=idx, y=data['currency_usd'],
        name=f'{config["currency_name"]} / USD',
        color=colors['usd'], width=LINE_WIDTH, dash='dot',
        hover_template=f'{config["currency_name"]} in USD: %{{y:.1f}}{hover_suffix}<extra></extra>',
    )
    # CPI-adjusted (visible)
    chart.add_line(
        row=1, x=idx, y=data['currency_real'],
        name=f'{config["currency_name"]} (CPI-adj)',
        color=colors['cpi'], width=LINE_WIDTH, dash='dot',
        hover_template=f'{config["currency_name"]} CPI-adj: %{{y:.1f}}{hover_suffix}<extra></extra>',
    )
    # PPP-adjusted (visible)
    chart.add_line(
        row=1, x=idx, y=data['currency_ppp'],
        name=f'{config["currency_name"]} (PPP-adj)',
        color=colors['ppp'], width=LINE_WIDTH, dash='dot',
        hover_template=f'{config["currency_name"]} PPP-adj: %{{y:.1f}}{hover_suffix}<extra></extra>',
    )
    # Gold (hidden by default)
    chart.add_line(
        row=1, x=idx, y=data['currency_gold'],
        name=f'{config["currency_name"]} / Gold',
        color=colors['gold'], width=LINE_WIDTH, dash='dot',
        hover_template=f'{config["currency_name"]} in Gold: %{{y:.1f}}{hover_suffix}<extra></extra>',
        visible='legendonly',
    )
    # Silver (hidden by default)
    chart.add_line(
        row=1, x=idx, y=data['currency_silver'],
        name=f'{config["currency_name"]} / Silver',
        color=colors['silver'], width=LINE_WIDTH, dash='dot',
        hover_template=f'{config["currency_name"]} in Silver: %{{y:.1f}}{hover_suffix}<extra></extra>',
        visible='legendonly',
    )

    # Row 2: Index metrics (solid lines)
    # USD (visible)
    chart.add_line(
        row=2, x=idx, y=data['index_usd'],
        name=f'{config["index_name"]} / USD',
        color=colors['usd'], width=LINE_WIDTH,
        hover_template=f'{config["index_name"]} in USD: %{{y:.1f}}{hover_suffix}<extra></extra>',
    )
    # CPI-adjusted (visible)
    chart.add_line(
        row=2, x=idx, y=data['index_real'],
        name=f'{config["index_name"]} (CPI-adj)',
        color=colors['cpi'], width=LINE_WIDTH,
        hover_template=f'{config["index_name"]} CPI-adj: %{{y:.1f}}{hover_suffix}<extra></extra>',
    )
    # PPP-adjusted (visible)
    chart.add_line(
        row=2, x=idx, y=data['index_ppp'],
        name=f'{config["index_name"]} (PPP-adj)',
        color=colors['ppp'], width=LINE_WIDTH,
        hover_template=f'{config["index_name"]} PPP-adj: %{{y:.1f}}{hover_suffix}<extra></extra>',
    )
    # Gold (hidden by default)
    chart.add_line(
        row=2, x=idx, y=data['index_gold'],
        name=f'{config["index_name"]} / Gold',
        color=colors['gold'], width=LINE_WIDTH,
        hover_template=f'{config["index_name"]} in Gold: %{{y:.1f}}{hover_suffix}<extra></extra>',
        visible='legendonly',
    )
    # Silver (hidden by default)
    chart.add_line(
        row=2, x=idx, y=data['index_silver'],
        name=f'{config["index_name"]} / Silver',
        color=colors['silver'], width=LINE_WIDTH,
        hover_template=f'{config["index_name"]} in Silver: %{{y:.1f}}{hover_suffix}<extra></extra>',
        visible='legendonly',
    )
    # Index Nominal (hidden by default)
    chart.add_line(
        row=2, x=idx, y=data['index_local'],
        name=f'{config["index_name"]} (Nominal)',
        color=colors.get(country, '#ffffff'), width=LINE_WIDTH,
        hover_template=f'{config["index_name"]} Nominal: %{{y:.1f}}{hover_suffix}<extra></extra>',
        visible='legendonly',
    )

    # Reference lines
    chart.add_hline(row=1, y=ref_line)
    chart.add_hline(row=2, y=ref_line)

    # Y-axes
    chart.set_yaxis(row=1, title='Value')
    chart.set_yaxis(row=2, title='Value')

    # X-axes - show month and year in hover, with explicit range to avoid padding
    x_range = (idx[0], idx[-1])
    chart.set_xaxis(row=1, tick_format='%b %Y', hover_format='%b %Y', range=x_range)
    chart.set_xaxis(row=2, title='Date', tick_format='%b %Y', hover_format='%b %Y', range=x_range)

    # Layout
    chart.set_legend(orientation='v', position='right')
    chart.set_margins(top=60, left=55, right=180, bottom=35)

    # Enable unified spike line spanning both subplots
    chart.enable_unified_spikeline(spike_color='rgba(255, 255, 255, 0.7)')

    return chart.build()


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
        ref_line = 100
        y_labels = ['Value', 'Value', 'Value', 'Value', 'Value']

    chart = EconChart(
        num_rows=num_rows,
        row_heights=[0.25, 0.20, 0.20, 0.175, 0.175],
        subplot_titles=titles,
        colors=colors,
        shared_xaxes=True,
        vertical_spacing=0.05,
        height=950,
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

        # Row 1: USD-denominated
        chart.add_line(
            row=1, x=months, y=data['currency_usd'],
            name=f"{country} {config['currency_name']}",
            color=color, width=LINE_WIDTH, dash='dot',
            hover_template=f"{country} {config['currency_name']}/USD: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=currency_visible, legendgroup=currency_group,
        )
        chart.add_line(
            row=1, x=months, y=data['index_usd'],
            name=f"{country} {config['index_name']}",
            color=color, width=LINE_WIDTH,
            hover_template=f"{country} {config['index_name']}/USD: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=index_visible, legendgroup=index_group,
        )

        # Row 2: CPI-adjusted
        chart.add_line(
            row=2, x=months, y=data['currency_real'],
            name=f"{config['currency_name']}/CPI",
            color=color, width=LINE_WIDTH, dash='dot',
            hover_template=f"{country} {config['currency_name']} CPI-adj: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=currency_visible, legendgroup=currency_group, showlegend=False,
        )
        chart.add_line(
            row=2, x=months, y=data['index_real'],
            name=f"{config['index_name']}/CPI",
            color=color, width=LINE_WIDTH,
            hover_template=f"{country} {config['index_name']} CPI-adj: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=index_visible, legendgroup=index_group, showlegend=False,
        )

        # Row 3: PPP-adjusted
        chart.add_line(
            row=3, x=months, y=data['currency_ppp'],
            name=f"{config['currency_name']}/PPP",
            color=color, width=LINE_WIDTH, dash='dot',
            hover_template=f"{country} {config['currency_name']} PPP-adj: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=currency_visible, legendgroup=currency_group, showlegend=False,
        )
        chart.add_line(
            row=3, x=months, y=data['index_ppp'],
            name=f"{config['index_name']}/PPP",
            color=color, width=LINE_WIDTH,
            hover_template=f"{country} {config['index_name']} PPP-adj: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=index_visible, legendgroup=index_group, showlegend=False,
        )

        # Row 4: Gold-denominated
        chart.add_line(
            row=4, x=months, y=data['currency_gold'],
            name=f"{config['currency_name']}/Gold",
            color=color, width=LINE_WIDTH, dash='dot',
            hover_template=f"{country} {config['currency_name']}/Gold: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=currency_visible, legendgroup=currency_group, showlegend=False,
        )
        chart.add_line(
            row=4, x=months, y=data['index_gold'],
            name=f"{config['index_name']}/Gold",
            color=color, width=LINE_WIDTH,
            hover_template=f"{country} {config['index_name']}/Gold: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=index_visible, legendgroup=index_group, showlegend=False,
        )

        # Row 5: Silver-denominated
        chart.add_line(
            row=5, x=months, y=data['currency_silver'],
            name=f"{config['currency_name']}/Silver",
            color=color, width=LINE_WIDTH, dash='dot',
            hover_template=f"{country} {config['currency_name']}/Silver: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=currency_visible, legendgroup=currency_group, showlegend=False,
        )
        chart.add_line(
            row=5, x=months, y=data['index_silver'],
            name=f"{config['index_name']}/Silver",
            color=color, width=LINE_WIDTH,
            hover_template=f"{country} {config['index_name']}/Silver: %{{y:.1f}}<br>Month %{{x}}<extra></extra>",
            visible=index_visible, legendgroup=index_group, showlegend=False,
        )

    # Reference lines
    for row in range(1, num_rows + 1):
        chart.add_hline(row=row, y=ref_line, color=colors['zero_line'])

    x_range = [0, max_months] if max_months > 0 else None

    # Y-axes with color-coded titles
    y_title_colors = [colors['usd'], colors['cpi'], colors['ppp'], colors['gold'], colors['silver']]
    for row, (y_label, title_color) in enumerate(zip(y_labels, y_title_colors), start=1):
        chart.set_yaxis(row=row, title=y_label, title_color=title_color)

    # X-axes - months from crisis start
    for row in range(1, num_rows + 1):
        if row == num_rows:
            chart.set_xaxis(row=row, title='Months from Crisis Start', range=x_range)
        else:
            chart.set_xaxis(row=row, range=x_range)

    # Layout
    chart.set_legend(orientation='h', position='top')
    chart.set_margins(top=120, left=55, right=55, bottom=35)

    # Enable unified spike line spanning all subplots
    chart.enable_unified_spikeline(spike_color='rgba(255, 255, 255, 0.7)')

    # Additional layout settings
    chart.fig.update_layout(
        title=dict(y=0.99, yanchor='top'),
        spikedistance=-1,
    )

    return chart.build()


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
