"""Plotly chart builders shared by the Streamlit pages.

Design rules (applied to every chart):
  - colours follow the meaning, never the rank: income is always blue, spending always orange;
    positive savings blue, negative savings red; magnitude charts use one hue
  - thin marks (2px lines, 8px markers, rounded bar ends), one y-axis, recessive grid
  - labels and values use neutral ink, not the series colour; every chart has hover tooltips
  - colours checked for colour-blind separation and contrast in light and dark mode
Each builder returns None for empty data so the page can show an empty-state message instead.
"""

import pandas as pd
import plotly.graph_objects as go

INCOME_COLOR = "#2a78d6"      # blue
SPENDING_COLOR = "#eb6834"    # orange
POSITIVE_COLOR = "#2a78d6"    # savings above zero
NEGATIVE_COLOR = "#e34948"    # savings below zero
MAGNITUDE_COLOR = "#2a78d6"   # single-hue bars (spending by category)
BEFORE_COLOR = "#a8a6a0"      # neutral grey: the forecast before a what-if change
INK_MUTED = "#898781"         # axis labels and direct labels (readable on light and dark)
GRID_COLOR = "rgba(137, 135, 129, 0.22)"
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID_COLOR, ticks="")
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, ticks="")
    return fig


def _month_axis(monthly: pd.DataFrame) -> pd.Series:
    return monthly["month"].dt.to_timestamp()


def income_vs_spending_chart(monthly: pd.DataFrame) -> go.Figure | None:
    """Two lines over the months: income (blue) and spending (orange)."""
    if monthly.empty:
        return None
    x = _month_axis(monthly)
    fig = go.Figure()
    for name, column, color in (("Income", "income", INCOME_COLOR), ("Spending", "expenses", SPENDING_COLOR)):
        fig.add_trace(go.Scatter(
            x=x, y=monthly[column], name=name, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color, line=dict(width=2, color="rgba(252,252,251,0.9)")),
            hovertemplate=f"{name}: €%{{y:,.2f}}<extra></extra>",
        ))
        # direct label at the last point, in neutral ink
        fig.add_annotation(x=x.iloc[-1], y=monthly[column].iloc[-1], text=name, showarrow=False,
                           xanchor="left", xshift=10, font=dict(color=INK_MUTED, size=12))
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(dtick="M1", tickformat="%b %Y", hoverformat="%B %Y")
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", rangemode="tozero")
    return _style(fig)


def monthly_savings_bar_chart(monthly: pd.DataFrame) -> go.Figure | None:
    """Savings per month; months below zero are red."""
    if monthly.empty:
        return None
    colors = [NEGATIVE_COLOR if v < 0 else POSITIVE_COLOR for v in monthly["savings"]]
    fig = go.Figure(go.Bar(
        x=_month_axis(monthly), y=monthly["savings"],
        marker=dict(color=colors, cornerradius=4, line=dict(width=0)),
        hovertemplate="%{x|%B %Y}<br>Savings: €%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=INK_MUTED)
    fig.update_layout(bargap=0.45, showlegend=False)
    fig.update_xaxes(dtick="M1", tickformat="%b %Y")
    fig.update_yaxes(tickprefix="€", tickformat=",.0f")
    return _style(fig, height=280)


def category_bar_chart(by_category: pd.DataFrame, top_n: int = 8) -> go.Figure | None:
    """Horizontal bars of spending per category, largest on top, in one hue.

    Categories beyond the top_n largest are folded into 'Other' so the chart stays readable.
    """
    if by_category.empty:
        return None
    data = by_category.sort_values("expenses", ascending=False, ignore_index=True)
    if len(data) > top_n:
        rest = data.iloc[top_n - 1:]["expenses"].sum()
        data = pd.concat([data.iloc[:top_n - 1],
                          pd.DataFrame({"category": ["Other"], "expenses": [rest]})], ignore_index=True)
    total = data["expenses"].sum()
    share = data["expenses"] / total
    data = data.iloc[::-1]  # Plotly draws horizontal bars bottom-up
    fig = go.Figure(go.Bar(
        x=data["expenses"], y=data["category"], orientation="h",
        marker=dict(color=MAGNITUDE_COLOR, cornerradius=4, line=dict(width=0)),
        text=[f"€{v:,.0f}" for v in data["expenses"]], textposition="outside",
        textfont=dict(color=INK_MUTED), cliponaxis=False,
        customdata=share.iloc[::-1],
        hovertemplate="%{y}: €%{x:,.2f} (%{customdata:.0%} of spending)<extra></extra>",
    ))
    fig.update_layout(bargap=0.35, showlegend=False)
    fig.update_xaxes(visible=False, range=[0, data["expenses"].max() * 1.18])
    fig.update_yaxes(showgrid=False, ticklabelstandoff=10)
    return _style(fig, height=max(160, 44 * len(data) + 40))


def forecast_fan_chart(monthly: pd.DataFrame, forecast) -> go.Figure | None:
    """Past income and spending as lines, then next month's forecast fanning out to its 10-90 % range.

    monthly: complete past months (aggregate_monthly); forecast: a predictor Forecast.
    """
    if monthly.empty:
        return None
    x = _month_axis(monthly)
    target = forecast.target_month.to_timestamp()
    fig = go.Figure()
    for name, column, series, color, fill in (
            ("Income", "income", "income", INCOME_COLOR, "rgba(42,120,214,0.16)"),
            ("Spending", "expenses", "expenses", SPENDING_COLOR, "rgba(235,104,52,0.16)")):
        f = forecast.get(series)
        last_x, last_y = x.iloc[-1], monthly[column].iloc[-1]
        # the fan: from the last real value out to the range of next month
        fig.add_trace(go.Scatter(
            x=[last_x, target, target, last_x], y=[last_y, f["upper"], f["lower"], last_y],
            fill="toself", fillcolor=fill, line=dict(width=0), mode="lines", hoverinfo="skip",
            showlegend=False, legendgroup=name))
        fig.add_trace(go.Scatter(
            x=x, y=monthly[column], name=name, mode="lines+markers", legendgroup=name,
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color, line=dict(width=2, color="rgba(252,252,251,0.9)")),
            hovertemplate=f"{name}: €%{{y:,.2f}}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[last_x, target], y=[last_y, f["predicted"]], mode="lines+markers", legendgroup=name,
            showlegend=False, line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=[0, 10], color=color, symbol="diamond", line=dict(width=2, color="rgba(252,252,251,0.9)")),
            customdata=[[last_y, last_y], [f["lower"], f["upper"]]],
            hovertemplate=(f"{name} forecast: €%{{y:,.2f}}<br>likely €%{{customdata[0]:,.0f}} – "
                           f"€%{{customdata[1]:,.0f}}<extra></extra>")))
        fig.add_annotation(x=target, y=f["predicted"], text=name, showarrow=False,
                           xanchor="left", xshift=12, font=dict(color=INK_MUTED, size=12))
    fig.add_vline(x=x.iloc[-1], line_width=1, line_dash="dot", line_color=INK_MUTED)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickvals=list(x) + [target], tickformat="%b %Y", hoverformat="%B %Y",
                     range=[x.iloc[0] - pd.Timedelta(days=12), target + pd.Timedelta(days=55)])
    fig.update_yaxes(tickprefix="€", tickformat=",.0f", rangemode="tozero")
    return _style(fig, height=380)


def what_if_chart(before, after) -> go.Figure:
    """Income, spending and end balance: forecast (grey) next to the scenario (blue), one shared axis."""
    labels = ["Income", "Spending", "Balance at month end"]

    def values(f):
        s = f.series.set_index("series")["predicted"]
        return [s["income"], s["expenses"], f.balance["predicted"]]

    fig = go.Figure()
    for name, forecast, color in (("Forecast", before, BEFORE_COLOR), ("Your scenario", after, INCOME_COLOR)):
        v = values(forecast)
        fig.add_trace(go.Bar(
            x=labels, y=v, name=name, marker=dict(color=color, cornerradius=4, line=dict(width=0)),
            text=[f"€{x:,.0f}" for x in v], textposition="outside", textfont=dict(color=INK_MUTED),
            cliponaxis=False, hovertemplate=f"{name} · %{{x}}: €%{{y:,.2f}}<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color=INK_MUTED)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(tickprefix="€", tickformat=",.0f")
    fig.update_xaxes(showgrid=False)
    return _style(fig, height=360)
