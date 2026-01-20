"""Visualization functions for hyperinflation analysis."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CHART_COLORS
from processing import compute_performance_stats


# Chart constants
LINE_WIDTH = 1.5
DEFAULT_NUM_ROWS = 3


def enable_unified_spikeline(fig, num_rows, x_range=None, spike_color='rgba(255, 255, 255, 0.5)'):
    """
    Enable spike lines that span all subplots in a multi-row figure.

    This is a workaround for Plotly 4.0+ where make_subplots creates separate
    x-axes, preventing spike lines from spanning all subplots.
    See: https://github.com/plotly/plotly.py/issues/1677

    Args:
        fig: Plotly figure with subplots
        num_rows: Number of subplot rows
        x_range: Optional tuple of (min, max) for x-axis range
        spike_color: Color for the spike line
    """
    # Bind all traces to the bottom x-axis
    fig.update_traces(xaxis=f'x{num_rows}')

    # Add invisible traces to upper axes to force tick label rendering
    # (Plotly won't render tick labels for axes with no bound traces)
    if x_range is not None:
        for row in range(1, num_rows):
            fig.add_trace(go.Scatter(
                x=list(x_range),
                y=[0, 0],
                mode='markers',
                marker=dict(opacity=0),
                showlegend=False,
                hoverinfo='skip',
                xaxis=f'x{row}' if row > 1 else 'x',
                yaxis=f'y{row}' if row > 1 else 'y'
            ))

    # Sync upper x-axes to bottom x-axis so they zoom together and align labels
    bottom_xaxis = f'x{num_rows}'
    for row in range(1, num_rows):
        fig.update_xaxes(row=row, col=1, matches=bottom_xaxis)

    # Apply spike settings to all x-axes
    spike_settings = dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor=spike_color,
        spikethickness=1,
        spikedash='dot',
    )
    fig.update_xaxes(**spike_settings)


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

    # Enable unified spike line spanning both subplots
    date_range = (idx.min(), idx.max()) if len(idx) > 0 else None
    enable_unified_spikeline(fig, num_rows=2, x_range=date_range, spike_color='rgba(255, 255, 255, 0.7)')

    return fig


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

    # Enable unified spike line spanning all subplots
    enable_unified_spikeline(fig, num_rows=num_rows, x_range=x_range, spike_color='rgba(255, 255, 255, 0.7)')

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
