"""Reusable economic chart class with dark theme support and multi-subplot capabilities."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Default color scheme for dark theme economic charts
DEFAULT_COLORS = {
    'background': '#1a1a2e',
    'paper': '#16213e',
    'grid': '#2a2a4a',
    'text': '#e8e8e8',
    'spike': 'rgba(255, 255, 255, 0.5)',
    'zero_line': 'rgba(255, 255, 255, 0.3)',
}


def enable_unified_spikeline(fig: go.Figure, num_rows: int, x_range: tuple | None = None,
                             spike_color: str = 'rgba(255, 255, 255, 0.5)') -> None:
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


class EconChart:
    """
    A configurable multi-subplot chart builder with dark theme support.

    Provides a fluent interface for building economic/financial charts with
    consistent styling, unified spike lines, and flexible layout options.

    Example usage:
        chart = EconChart(
            num_rows=3,
            subplot_titles=('Revenue', 'Costs', 'Profit'),
            height=700,
        )
        chart.add_line(row=1, x=dates, y=revenue, name='Revenue', color='#00ff00')
        chart.add_line(row=2, x=dates, y=costs, name='Costs', color='#ff0000')
        chart.add_line(row=3, x=dates, y=profit, name='Profit', color='#0000ff')
        chart.add_hline(row=3, y=0)
        chart.enable_unified_spikeline()
        fig = chart.build()
        fig.show()
    """

    def __init__(
        self,
        num_rows: int,
        row_heights: list[float] | None = None,
        subplot_titles: tuple[str, ...] | None = None,
        colors: dict[str, str] | None = None,
        shared_xaxes: bool = True,
        vertical_spacing: float = 0.05,
        height: int = 600,
    ) -> None:
        """
        Initialize an EconChart.

        Args:
            num_rows: Number of subplot rows
            row_heights: List of relative heights for each row. Auto-equal if None.
            subplot_titles: Tuple of titles for each subplot
            colors: Theme colors dict (defaults to DEFAULT_COLORS)
            shared_xaxes: Whether to share x-axes across subplots
            vertical_spacing: Spacing between subplots (0-1)
            height: Chart height in pixels
        """
        self.num_rows = num_rows
        self.colors = colors or DEFAULT_COLORS.copy()
        self.height = height
        self._spike_enabled = False
        self._spike_color = self.colors.get('spike', 'rgba(255, 255, 255, 0.5)')
        self._x_range: tuple | None = None

        # Auto-calculate equal row heights if not specified
        if row_heights is None:
            row_heights = [1.0 / num_rows] * num_rows

        self.fig = make_subplots(
            rows=num_rows,
            cols=1,
            row_heights=row_heights,
            shared_xaxes=shared_xaxes,
            vertical_spacing=vertical_spacing,
            subplot_titles=subplot_titles,
        )

        # Apply default dark theme layout
        self.fig.update_layout(
            height=height,
            hovermode='x unified',
            paper_bgcolor=self.colors.get('paper', DEFAULT_COLORS['paper']),
            plot_bgcolor=self.colors.get('background', DEFAULT_COLORS['background']),
            font=dict(color=self.colors.get('text', DEFAULT_COLORS['text']), size=10),
            hoverlabel=dict(
                bgcolor=self.colors.get('paper', DEFAULT_COLORS['paper']),
                font_size=11,
                font_color=self.colors.get('text', DEFAULT_COLORS['text']),
            ),
        )

        # Style subplot titles
        if subplot_titles:
            for annotation in self.fig['layout']['annotations']:
                annotation['font'] = dict(
                    size=11,
                    color=self.colors.get('text', DEFAULT_COLORS['text'])
                )

        # Apply default grid color to all axes
        grid_color = self.colors.get('grid', DEFAULT_COLORS['grid'])
        for row in range(1, num_rows + 1):
            self.fig.update_xaxes(gridcolor=grid_color, row=row, col=1)
            self.fig.update_yaxes(gridcolor=grid_color, row=row, col=1)

    def add_line(
        self,
        row: int,
        x: Any,
        y: Any,
        name: str,
        color: str,
        width: float = 1.5,
        dash: str | None = None,
        hover_template: str | None = None,
        visible: bool | str = True,
        legendgroup: str | None = None,
        showlegend: bool = True,
    ) -> EconChart:
        """
        Add a line trace to the chart.

        Args:
            row: Row number (1-indexed)
            x: X-axis data
            y: Y-axis data
            name: Trace name for legend
            color: Line color
            width: Line width
            dash: Line dash style ('solid', 'dot', 'dash', 'longdash', 'dashdot')
            hover_template: Custom hover template
            visible: True, False, or 'legendonly'
            legendgroup: Group name for synchronized legend toggling
            showlegend: Whether to show in legend

        Returns:
            Self for method chaining
        """
        line_dict: dict[str, Any] = {'color': color, 'width': width}
        if dash:
            line_dict['dash'] = dash

        trace_kwargs: dict[str, Any] = {
            'x': x,
            'y': y,
            'name': name,
            'line': line_dict,
            'visible': visible,
            'showlegend': showlegend,
        }
        if hover_template:
            trace_kwargs['hovertemplate'] = hover_template
        if legendgroup:
            trace_kwargs['legendgroup'] = legendgroup

        self.fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=1)

        # Track x range for unified spikeline
        self._update_x_range(x)

        return self

    def add_scatter(
        self,
        row: int,
        x: Any,
        y: Any,
        name: str,
        color: str,
        marker_size: int = 6,
        hover_template: str | None = None,
        visible: bool | str = True,
        legendgroup: str | None = None,
        showlegend: bool = True,
    ) -> EconChart:
        """
        Add a scatter (markers only) trace to the chart.

        Args:
            row: Row number (1-indexed)
            x: X-axis data
            y: Y-axis data
            name: Trace name for legend
            color: Marker color
            marker_size: Marker size
            hover_template: Custom hover template
            visible: True, False, or 'legendonly'
            legendgroup: Group name for synchronized legend toggling
            showlegend: Whether to show in legend

        Returns:
            Self for method chaining
        """
        trace_kwargs: dict[str, Any] = {
            'x': x,
            'y': y,
            'name': name,
            'mode': 'markers',
            'marker': dict(color=color, size=marker_size),
            'visible': visible,
            'showlegend': showlegend,
        }
        if hover_template:
            trace_kwargs['hovertemplate'] = hover_template
        if legendgroup:
            trace_kwargs['legendgroup'] = legendgroup

        self.fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=1)

        # Track x range for unified spikeline
        self._update_x_range(x)

        return self

    def set_yaxis(
        self,
        row: int,
        title: str | None = None,
        title_color: str | None = None,
        scale_type: str = 'linear',
        gridcolor: str | None = None,
    ) -> EconChart:
        """
        Configure y-axis for a specific row.

        Args:
            row: Row number (1-indexed)
            title: Axis title
            title_color: Title color (defaults to text color)
            scale_type: 'linear' or 'log'
            gridcolor: Grid line color

        Returns:
            Self for method chaining
        """
        update_kwargs: dict[str, Any] = {'type': scale_type}

        if title:
            title_font_color = title_color or self.colors.get('text', DEFAULT_COLORS['text'])
            update_kwargs['title_text'] = title
            update_kwargs['title_font'] = dict(size=10, color=title_font_color)

        if gridcolor:
            update_kwargs['gridcolor'] = gridcolor

        self.fig.update_yaxes(row=row, col=1, **update_kwargs)
        return self

    def set_xaxis(
        self,
        row: int,
        title: str | None = None,
        tick_format: str | None = None,
        hover_format: str | None = None,
        range: tuple | None = None,
        gridcolor: str | None = None,
    ) -> EconChart:
        """
        Configure x-axis for a specific row.

        Args:
            row: Row number (1-indexed)
            title: Axis title
            tick_format: Tick label format (e.g., '%b %Y' for dates)
            hover_format: Hover label format
            range: Tuple of (min, max) for axis range
            gridcolor: Grid line color

        Returns:
            Self for method chaining
        """
        update_kwargs: dict[str, Any] = {}

        if title:
            update_kwargs['title_text'] = title
            update_kwargs['title_font'] = dict(size=10, color=self.colors.get('text', DEFAULT_COLORS['text']))

        if tick_format:
            update_kwargs['tickformat'] = tick_format

        if hover_format:
            update_kwargs['hoverformat'] = hover_format

        if range:
            update_kwargs['range'] = range

        if gridcolor:
            update_kwargs['gridcolor'] = gridcolor

        self.fig.update_xaxes(row=row, col=1, **update_kwargs)
        return self

    def add_hline(
        self,
        row: int,
        y: float,
        color: str | None = None,
        dash: str = 'dash',
        width: float = 1,
    ) -> EconChart:
        """
        Add a horizontal reference line to a subplot.

        Args:
            row: Row number (1-indexed)
            y: Y-value for the line
            color: Line color (defaults to zero_line color)
            dash: Line dash style
            width: Line width

        Returns:
            Self for method chaining
        """
        line_color = color or self.colors.get('zero_line', DEFAULT_COLORS['zero_line'])
        self.fig.add_hline(
            y=y,
            line_dash=dash,
            line_color=line_color,
            line_width=width,
            row=row,
            col=1,
        )
        return self

    def enable_unified_spikeline(self, spike_color: str | None = None) -> EconChart:
        """
        Enable spike lines that span all subplots.

        Args:
            spike_color: Color for the spike line

        Returns:
            Self for method chaining
        """
        self._spike_enabled = True
        if spike_color:
            self._spike_color = spike_color
        return self

    def set_legend(
        self,
        orientation: str = 'h',
        position: str = 'top',
    ) -> EconChart:
        """
        Configure legend position and orientation.

        Args:
            orientation: 'h' for horizontal, 'v' for vertical
            position: 'top', 'bottom', or 'right'

        Returns:
            Self for method chaining
        """
        legend_kwargs: dict[str, Any] = {
            'orientation': orientation,
            'font': dict(size=9, color=self.colors.get('text', DEFAULT_COLORS['text'])),
            'bgcolor': 'rgba(0,0,0,0)',
        }

        if position == 'top':
            legend_kwargs.update({
                'yanchor': 'bottom',
                'y': 1.01,
                'xanchor': 'center',
                'x': 0.5,
            })
        elif position == 'bottom':
            legend_kwargs.update({
                'yanchor': 'top',
                'y': -0.1,
                'xanchor': 'center',
                'x': 0.5,
            })
        elif position == 'right':
            legend_kwargs.update({
                'yanchor': 'top',
                'y': 1,
                'xanchor': 'left',
                'x': 1.02,
            })

        self.fig.update_layout(legend=legend_kwargs)
        return self

    def set_margins(
        self,
        top: int = 60,
        left: int = 55,
        right: int = 55,
        bottom: int = 35,
    ) -> EconChart:
        """
        Set chart margins.

        Args:
            top: Top margin in pixels
            left: Left margin in pixels
            right: Right margin in pixels
            bottom: Bottom margin in pixels

        Returns:
            Self for method chaining
        """
        self.fig.update_layout(margin=dict(t=top, l=left, r=right, b=bottom))
        return self

    def set_title(self, text: str, font_size: int = 13) -> EconChart:
        """
        Set chart title.

        Args:
            text: Title text
            font_size: Font size

        Returns:
            Self for method chaining
        """
        self.fig.update_layout(
            title=dict(
                text=text,
                font=dict(size=font_size, color=self.colors.get('text', DEFAULT_COLORS['text'])),
                y=0.99,
                yanchor='top',
            )
        )
        return self

    def build(self) -> go.Figure:
        """
        Finalize and return the Plotly figure.

        Applies unified spike line if enabled.

        Returns:
            Plotly Figure object
        """
        if self._spike_enabled:
            enable_unified_spikeline(
                self.fig,
                self.num_rows,
                x_range=self._x_range,
                spike_color=self._spike_color,
            )
        return self.fig

    def show(self) -> None:
        """Display the chart."""
        self.build().show()

    def to_html(self, path: str, include_plotlyjs: bool | str = True) -> None:
        """
        Export chart to HTML file.

        Args:
            path: Output file path
            include_plotlyjs: Whether to include plotly.js ('cdn', True, False)
        """
        self.build().write_html(path, include_plotlyjs=include_plotlyjs)

    def _update_x_range(self, x: Any) -> None:
        """Update tracked x-axis range for unified spikeline."""
        try:
            if hasattr(x, '__len__') and len(x) > 0:
                x_min = min(x)
                x_max = max(x)
                if self._x_range is None:
                    self._x_range = (x_min, x_max)
                else:
                    self._x_range = (
                        min(self._x_range[0], x_min),
                        max(self._x_range[1], x_max),
                    )
        except (TypeError, ValueError):
            pass  # Skip if x is not comparable
