"""
Theme, Custom CSS Styles, and Plotly Chart Templates for Portfolio Risk Shiny Dashboard.
"""

from typing import Optional
from shiny import ui
import plotly.graph_objects as go
import plotly.io as pio

# Custom Color Palette
PALETTE = {
    "primary": "#1E3A8A",       # Deep Navy Blue
    "primary_light": "#3B82F6", # Bright Blue
    "secondary": "#065F46",     # Emerald Green
    "secondary_light": "#10B981", # Mint Green
    "accent_orange": "#D97706", # Amber Orange
    "accent_red": "#DC2626",    # Crimson Red
    "accent_purple": "#7C3AED", # Royal Purple
    "accent_cyan": "#0891B2",   # Cyan / Teal
    "bg_card": "#FFFFFF",
    "bg_subtle": "#F8FAFC",
    "border": "#E2E8F0",
    "text_dark": "#0F172A",
    "text_muted": "#64748B",
}

# Distinct Stock Colors for Multi-Asset Overlays
STOCK_COLORS = [
    "#1E3A8A", "#065F46", "#D97706", "#7C3AED", "#DC2626",
    "#0891B2", "#EC4899", "#4F46E5", "#059669", "#EA580C",
    "#8B5CF6", "#0284C7", "#16A34A", "#CA8A04", "#E11D48"
]


def get_custom_css() -> str:
    """Returns high-grade modern CSS for the Shiny application."""
    return """
    /* Global layout & typography */
    body {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #F8FAFC;
        color: #0F172A;
    }

    .container-fluid {
        max-width: 98% !important;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 12px;
        height: 100%;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
    }
    .metric-label {
        font-size: 0.80rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }
    .delta-positive { color: #16A34A; }
    .delta-negative { color: #DC2626; }
    .delta-neutral { color: #64748B; }

    /* Section Headers */
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Tab navigation styling */
    .nav-tabs {
        border-bottom: 2px solid #E2E8F0 !important;
        gap: 6px;
        margin-bottom: 1.2rem;
    }
    .nav-tabs .nav-link {
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569;
        border: none !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 18px;
        transition: all 0.15s ease;
        background-color: transparent;
    }
    .nav-tabs .nav-link:hover {
        color: #1D4ED8;
        background-color: #F1F5F9;
    }
    .nav-tabs .nav-link.active {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        border-bottom: 3px solid #1D4ED8 !important;
    }

    /* Modern Table Styling */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        margin-top: 6px;
        margin-bottom: 12px;
        background: #FFFFFF;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }
    .custom-table th {
        background-color: #F8FAFC;
        color: #475569;
        font-weight: 600;
        text-align: left;
        padding: 10px 14px;
        border-bottom: 1px solid #E2E8F0;
    }
    .custom-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #F1F5F9;
        color: #1E293B;
    }
    .custom-table tbody tr:hover {
        background-color: #F8FAFC;
    }

    /* Info Alert Callouts */
    .info-callout {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #166534;
    }
    .warning-callout {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #991B1B;
    }

    /* Card styling */
    .card {
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
        background-color: #FFFFFF !important;
    }
    .card-header {
        background-color: #F8FAFC !important;
        border-bottom: 1px solid #E2E8F0 !important;
        font-weight: 600 !important;
        color: #0F172A !important;
    }

    /* Form control adjustments */
    .form-control, .form-select {
        border-radius: 6px;
        border: 1px solid #CBD5E1;
        font-size: 0.9rem;
    }
    .form-control:focus, .form-select:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }

    /* =========================================================================
       Busy & Loading Animation Indicators
       ========================================================================= */
    /* Top Gradient Loading Bar when Shiny is busy */
    html.shiny-busy::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981, #3B82F6, #1E3A8A);
        background-size: 200% 100%;
        animation: shiny-loading-bar-anim 1.2s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        z-index: 99999;
    }

    @keyframes shiny-loading-bar-anim {
        0% { background-position: 100% 0; }
        100% { background-position: -100% 0; }
    }

    /* Recalculating output fade & shimmer animation */
    .shiny-bound-output.recalculating,
    .shiny-ipywidget-output.recalculating,
    .shiny-output-error.recalculating {
        opacity: 0.50 !important;
        pointer-events: none;
        transition: opacity 0.2s ease-in-out;
        position: relative;
    }

    /* Levels & Returns Loading Banner */
    .returns-loading-banner {
        display: none;
        align-items: center;
        gap: 10px;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 9px 16px;
        margin-bottom: 12px;
        color: #1E3A8A;
        font-size: 0.88rem;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(30, 58, 138, 0.08);
    }

    html.shiny-busy .returns-loading-banner {
        display: flex;
        animation: bannerFadeIn 0.25s ease-in-out;
    }

    @keyframes bannerFadeIn {
        from { opacity: 0; transform: translateY(-4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Pulse Dot Animation */
    .spinner-pulse {
        display: inline-block;
        width: 0.9rem;
        height: 0.9rem;
        border-radius: 50%;
        background-color: #2563EB;
        animation: spinner-pulse 0.9s ease-in-out infinite alternate;
    }

    @keyframes spinner-pulse {
        0% { transform: scale(0.6); opacity: 0.4; }
        100% { transform: scale(1.15); opacity: 1; }
    }
    """


def custom_css_header():
    """Generates the Shiny UI style tag with application CSS."""
    return ui.tags.style(get_custom_css())


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_type: str = "neutral"  # "positive", "negative", "neutral"
) -> str:
    """Renders HTML for a metric KPI card."""
    delta_html = ""
    if delta:
        delta_class = f"delta-{delta_type}"
        delta_html = f'<div class="metric-delta {delta_class}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def get_plotly_layout_defaults(
    plot_bgcolor: str = "#FFFFFF",
    paper_bgcolor: str = "#FFFFFF",
    grid_color: str = "#F1F5F9"
) -> dict:
    """Returns clean, consistent layout settings for Plotly charts."""
    return dict(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#334155"),
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=45, r=40, t=45, b=40),
        xaxis=dict(
            gridcolor=grid_color,
            linecolor="#CBD5E1",
            zerolinecolor="#CBD5E1",
            showline=True,
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=grid_color,
            linecolor="#CBD5E1",
            zerolinecolor="#CBD5E1",
            showline=True,
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10)
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF"
        )
    )


def get_grey_theme_layout_defaults(
    plot_bgcolor: str = "#EAECEF",
    paper_bgcolor: str = "#F8FAFC",
    grid_color: str = "#FFFFFF"
) -> dict:
    """
    Returns layout settings with a classic financial grey background
    and clean white gridlines (ggplot2/seaborn institutional style).
    """
    return dict(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=11, color="#1E293B"),
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        margin=dict(l=45, r=40, t=45, b=40),
        xaxis=dict(
            gridcolor=grid_color,
            gridwidth=1.2,
            linecolor="#94A3B8",
            zerolinecolor="#CBD5E1",
            showline=True,
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor=grid_color,
            gridwidth=1.2,
            linecolor="#94A3B8",
            zerolinecolor="#CBD5E1",
            showline=True,
            showgrid=True,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#CBD5E1",
            borderwidth=1,
            font=dict(size=10)
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF"
        )
    )

