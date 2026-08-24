"""
Theme, Custom CSS Styles, and Plotly Chart Templates for Portfolio Risk Dashboard.
"""

import streamlit as st
import streamlit.components.v1 as components
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


def ensure_sidebar_collapsed():
    """
    Injects JavaScript to collapse the sidebar by default and clear
    cached expanded state in the user's browser localStorage/sessionStorage.
    """
    components.html(
        """
        <script>
        (function() {
            try {
                // Clear browser-cached expanded state
                window.parent.localStorage.setItem('stSidebarExpanded', 'false');
                window.parent.sessionStorage.setItem('stSidebarExpanded', 'false');

                // If currently open on load, trigger collapse button
                const parentDoc = window.parent.document;
                const sidebar = parentDoc.querySelector('[data-testid="stSidebar"]');
                const collapseBtn = parentDoc.querySelector('[data-testid="stSidebarCollapseButton"] button') ||
                                    parentDoc.querySelector('button[aria-label="Close sidebar"]');

                if (sidebar && sidebar.getAttribute('aria-expanded') === 'true' && collapseBtn) {
                    collapseBtn.click();
                }
            } catch (e) {
                // Ignore any cross-origin sandboxing limits
            }
        })();
        </script>
        """,
        height=0,
        width=0
    )


def inject_custom_css():
    """Injects high-grade modern CSS into the Streamlit app."""
    css = """
    <style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 96%;
    }

    /* Collapsed sidebar toggle button styling */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        top: 0.8rem;
        left: 0.8rem;
        z-index: 999;
    }

    /* Metric Cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-bottom: 12px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.65rem;
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
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Tab navigation styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
        border-bottom: 3px solid #1D4ED8 !important;
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
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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
