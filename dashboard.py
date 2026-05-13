"""
Pakistan Informal Economy Formalization Tracker
================================================
A data science portfolio project measuring how quickly Pakistan's
informal economy is entering formal systems (2015–2025).

Data sources:
  - SBP Quarterly Payment Systems Review
  - FBR Annual Statistics
  - Pakistan Bureau of Statistics — Labour Force Survey
  - World Bank Global Findex

Run: streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Formalization Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "dark":     "#1C1C1C",
    "text":     "#2C2C2C",
    "muted":    "#6B7280",
    "bg":       "#FAF8F5",
    "card":     "#F5F0E8",
    "border":   "#D4C5B0",
    "green":    "#4A7C59",
    "amber":    "#C0843A",
    "rust":     "#8B6355",
    "blue":     "#5B8DB8",
    "sage":     "#97BC62",
    "chart":    ["#2C5F2E", "#8B6355", "#C0843A", "#5B8DB8", "#7C6F9F"],
}

EVENT_COLORS = {"Tax": C["amber"], "Fintech": C["blue"], "SME": C["green"], "Policy": C["rust"]}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500&display=swap');

  #MainMenu, footer, header {{ visibility: hidden; }}

  .stApp {{ background-color: {C['bg']}; }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background-color: #181818;
    border-right: 1px solid #2A2A2A;
  }}
  [data-testid="stSidebar"] * {{
    color: #C8C0B4 !important;
  }}
  [data-testid="stSidebar"] .stRadio label {{
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }}
  [data-testid="stSidebar"] .stRadio [data-checked="true"] label {{
    color: #F5F0E8 !important;
    font-weight: 600;
  }}

  /* ── Typography ── */
  .page-title {{
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: {C['dark']};
    line-height: 1.15;
    margin-bottom: 0.2rem;
  }}
  .page-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: {C['rust']};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.8rem;
  }}
  .section-head {{
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    color: {C['dark']};
    border-bottom: 2px solid {C['border']};
    padding-bottom: 0.4rem;
    margin: 1.8rem 0 1rem 0;
  }}

  /* ── Metric Cards ── */
  .metric-row {{ display: flex; gap: 1rem; margin-bottom: 1.5rem; }}
  .metric-card {{
    flex: 1;
    background: {C['card']};
    border: 1px solid {C['border']};
    border-top: 3px solid {C['green']};
    padding: 1.1rem 1.3rem;
    border-radius: 4px;
  }}
  .metric-card.amber {{ border-top-color: {C['amber']}; }}
  .metric-card.rust  {{ border-top-color: {C['rust']}; }}
  .metric-card.blue  {{ border-top-color: {C['blue']}; }}
  .metric-val {{
    font-family: 'Playfair Display', serif;
    font-size: 1.85rem;
    font-weight: 700;
    color: {C['dark']};
    line-height: 1;
  }}
  .metric-lbl {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.35rem;
  }}
  .metric-delta {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: {C['green']};
    margin-top: 0.25rem;
  }}
  .metric-delta.neg {{ color: {C['amber']}; }}

  /* ── Callout ── */
  .callout {{
    background: #EDE8E0;
    border-left: 4px solid {C['rust']};
    padding: 1rem 1.5rem;
    border-radius: 0 4px 4px 0;
    margin: 1.2rem 0;
  }}
  .callout-title {{
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    color: {C['dark']};
    margin-bottom: 0.3rem;
    font-weight: 700;
  }}
  .callout-body {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    color: {C['text']};
    line-height: 1.6;
  }}

  /* ── Disclaimer ── */
  .disclaimer {{
    background: #FEFCE8;
    border: 1px solid #FDE68A;
    padding: 0.7rem 1.1rem;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #92400E;
    margin-bottom: 1.2rem;
    line-height: 1.6;
  }}

  /* ── Legend pills ── */
  .legend-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem; }}
  .legend-pill {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: {C['muted']};
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .legend-dot {{
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
  }}

  /* ── Table ── */
  .styled-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
  }}
  .styled-table th {{
    background: {C['dark']};
    color: {C['card']};
    padding: 0.6rem 0.8rem;
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.68rem;
  }}
  .styled-table td {{
    padding: 0.55rem 0.8rem;
    border-bottom: 1px solid {C['border']};
    color: {C['text']};
  }}
  .styled-table tr:nth-child(even) td {{ background: {C['card']}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & INDEX CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
INDEX_WEIGHTS = {
    "fbr_filers_m":              0.30,
    "mobile_banking_users_m":     0.25,
    "digital_payment_vol_bpkr":  0.20,
    "sme_credit_pct":            0.15,
    "informal_employment_pct":   0.10,   # inverted
}

@st.cache_data
def load_data():
    df       = pd.read_csv("data/formalization_data.csv")
    events   = pd.read_csv("data/policy_events.csv")
    regional = pd.read_csv("data/regional_data.csv")
    return df, events, regional

@st.cache_data
def build_index(df_raw):
    df = df_raw.copy()
    scaler = MinMaxScaler()

    for col in ["fbr_filers_m", "mobile_banking_users_m", "digital_payment_vol_bpkr", "sme_credit_pct"]:
        df[f"{col}_norm"] = scaler.fit_transform(df[[col]])

    # Lower informal employment = more formal → invert
    df["informal_employment_pct_norm"] = 1 - scaler.fit_transform(df[["informal_employment_pct"]])

    df["formalization_index"] = (
        df["fbr_filers_m_norm"]              * INDEX_WEIGHTS["fbr_filers_m"]             +
        df["mobile_banking_users_m_norm"]     * INDEX_WEIGHTS["mobile_banking_users_m"]    +
        df["digital_payment_vol_bpkr_norm"]  * INDEX_WEIGHTS["digital_payment_vol_bpkr"] +
        df["sme_credit_pct_norm"]            * INDEX_WEIGHTS["sme_credit_pct"]           +
        df["informal_employment_pct_norm"]   * INDEX_WEIGHTS["informal_employment_pct"]
    ) * 100

    # Year-over-year growth columns
    for col in ["fbr_filers_m", "mobile_banking_users_m", "digital_payment_vol_bpkr"]:
        df[f"{col}_yoy"] = df[col].pct_change() * 100

    return df

def cross_correlation(s1, s2, max_lag=4):
    """Pearson cross-correlation at each lag (-max_lag … +max_lag)."""
    lags, corrs = [], []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            r = s1.iloc[:lag].corr(s2.iloc[-lag:])
        elif lag > 0:
            r = s1.iloc[lag:].corr(s2.iloc[:-lag])
        else:
            r = s1.corr(s2)
        lags.append(lag)
        corrs.append(round(r, 3))
    return lags, corrs


# ─────────────────────────────────────────────────────────────────────────────
# HELPER COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(value, label, delta, color_class=""):
    return f"""
    <div class="metric-card {color_class}">
        <div class="metric-val">{value}</div>
        <div class="metric-lbl">{label}</div>
        <div class="metric-delta">{delta}</div>
    </div>"""

def callout(title, body):
    st.markdown(f"""
    <div class="callout">
        <div class="callout-title">💡 {title}</div>
        <div class="callout-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def disclaimer():
    st.markdown("""
    <div class="disclaimer">
        ⚠ &nbsp; <strong>Data note:</strong> This dataset is illustrative, compiled from public SBP quarterly reports,
        FBR annual statistics, and PBS Labour Force Surveys. Figures for 2023–2024 are preliminary estimates.
        Regional breakdowns are approximations. This is a portfolio project — not a policy document.
    </div>""", unsafe_allow_html=True)

def base_layout(fig, height=420, margin=None):
    """Apply the shared plot theme."""
    m = margin or dict(l=50, r=30, t=20, b=50)
    fig.update_layout(
        height=height,
        margin=m,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,240,232,0.35)",
        font=dict(family="'DM Sans', sans-serif", color=C["text"], size=12),
        legend=dict(
            bgcolor="rgba(245,240,232,0.8)",
            bordercolor=C["border"],
            borderwidth=1,
            font=dict(family="'DM Mono', monospace", size=10),
        ),
        xaxis=dict(showgrid=False, linecolor=C["border"], tickfont=dict(size=11)),
        yaxis=dict(gridcolor="#E8DFD0", linecolor=C["border"], tickfont=dict(size=11)),
    )
    return fig

def add_events(fig, events, y_ann=0.98):
    """Overlay policy event vlines + annotations on a figure."""
    for _, ev in events.iterrows():
        color = EVENT_COLORS.get(ev["category"], C["muted"])
        fig.add_vline(x=ev["year"], line_dash="dot", line_color=color, line_width=1.2, opacity=0.7)
        fig.add_annotation(
            x=ev["year"], y=y_ann, yref="paper",
            text=ev["short_label"],
            textangle=-90, showarrow=False,
            font=dict(size=8.5, color=color, family="'DM Mono', monospace"),
            xanchor="left", yanchor="top",
            bgcolor="rgba(250,248,245,0.75)",
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_overview(df, events, df_idx):
    st.markdown('<div class="page-title">Pakistan Informal Economy<br>Formalization Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Measuring the pace of entry into formal economic systems · 2015–2025</div>', unsafe_allow_html=True)
    disclaimer()

    # ── KPI Metrics ──────────────────────────────────────────────────────
    latest = df.iloc[-1]
    earliest = df.iloc[0]

    cards_html = '<div class="metric-row">'
    cards_html += metric_card(
        f"{latest['fbr_filers_m']:.1f}M", "FBR Tax Filers (2024)",
        f"▲ +{latest['fbr_filers_m'] - earliest['fbr_filers_m']:.1f}M since 2015"
    )
    cards_html += metric_card(
        f"{latest['mobile_banking_users_m']:.0f}M", "Mobile Banking Users (2024)",
        f"▲ {latest['mobile_banking_users_m'] / earliest['mobile_banking_users_m']:.0f}× growth since 2015",
        "amber"
    )
    cards_html += metric_card(
        f"₨{latest['digital_payment_vol_bpkr']:,.0f}B", "Digital Payment Volume (2024)",
        f"▲ {latest['digital_payment_vol_bpkr'] / earliest['digital_payment_vol_bpkr']:.0f}× since 2015",
        "rust"
    )
    cards_html += metric_card(
        f"{latest['informal_employment_pct']:.1f}%", "Informal Employment Share (2024)",
        f"▼ {earliest['informal_employment_pct'] - latest['informal_employment_pct']:.1f}pp since 2015",
        "blue"
    )
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Composite Index Chart ─────────────────────────────────────────────
    st.markdown('<div class="section-head">Composite Formalization Index</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_idx["year"], y=df_idx["formalization_index"],
        fill="tozeroy",
        fillcolor="rgba(74,124,89,0.12)",
        line=dict(color=C["green"], width=2.5),
        mode="lines+markers",
        marker=dict(size=8, color=C["green"], line=dict(color="white", width=1.5)),
        name="Formalization Index",
        hovertemplate="<b>%{x}</b><br>Index: <b>%{y:.1f}</b> / 100<extra></extra>",
    ))

    add_events(fig, events, y_ann=0.97)
    base_layout(fig, height=390)
    fig.update_layout(
        yaxis=dict(title="Index Score (0–100)", range=[0, 108], gridcolor="#E8DFD0"),
        xaxis=dict(title="Year", tickmode="linear", dtick=1),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Legend for events ─────────────────────────────────────────────────
    st.markdown("""
    <div class="legend-row">
        <div class="legend-pill"><span class="legend-dot" style="background:#C0843A"></span>Tax Policy</div>
        <div class="legend-pill"><span class="legend-dot" style="background:#5B8DB8"></span>Fintech / Digital</div>
        <div class="legend-pill"><span class="legend-dot" style="background:#4A7C59"></span>SME Programs</div>
        <div class="legend-pill"><span class="legend-dot" style="background:#8B6355"></span>Broader Policy</div>
    </div>""", unsafe_allow_html=True)

    # ── Insight Callouts ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        callout(
            "Fintech is outpacing fiscal inclusion",
            "Mobile banking adoption has grown <strong>20×</strong> since 2015, yet the FBR tax net expanded by only <strong>5.4×</strong> over the same period. "
            "Digital financial access is widening faster than formal taxation is deepening."
        )
    with col2:
        callout(
            "Informal employment barely moved",
            "Despite significant fintech growth and SME programs, the informal employment share dropped only <strong>6.6 percentage points</strong> over nine years — "
            "suggesting formalization remains shallow and does not yet translate into labour market shifts."
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def page_indicators(df, events):
    st.markdown('<div class="page-title">Individual Indicators</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Disaggregated view of each formalization dimension · 2015–2025</div>', unsafe_allow_html=True)
    disclaimer()

    # ── Subplot grid 2×2 ─────────────────────────────────────────────────
    indicators = [
        ("fbr_filers_m",             "FBR Registered Tax Filers",   "Million filers",   C["chart"][0]),
        ("mobile_banking_users_m",    "Mobile Banking Users",          "Million users",    C["chart"][1]),
        ("digital_payment_vol_bpkr", "Digital Payment Volume",       "Billion PKR",      C["chart"][2]),
        ("sme_credit_pct",           "SME Share of Bank Credit",     "% of total credit",C["chart"][3]),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[i[1] for i in indicators],
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    positions = [(1,1),(1,2),(2,1),(2,2)]
    for (col, title, ylabel, color), (r, c) in zip(indicators, positions):
        fig.add_trace(
            go.Scatter(
                x=df["year"], y=df[col],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=7, color=color, line=dict(color="white", width=1.2)),
                name=title,
                showlegend=False,
                hovertemplate=f"<b>%{{x}}</b><br>{title}: <b>%{{y}}</b><extra></extra>",
            ),
            row=r, col=c
        )
        # Event vlines per subplot
        for _, ev in events.iterrows():
            ec = EVENT_COLORS.get(ev["category"], C["muted"])
            fig.add_vline(
                x=ev["year"], line_dash="dot", line_color=ec,
                line_width=1, opacity=0.5, row=r, col=c
            )

    fig.update_annotations(font=dict(family="'Playfair Display', serif", size=13, color=C["dark"]))
    base_layout(fig, height=580)
    fig.update_layout(margin=dict(l=50, r=20, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)

    # ── Informal Employment ───────────────────────────────────────────────
    st.markdown('<div class="section-head">Informal Employment Share — The Stubborn Indicator</div>', unsafe_allow_html=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["year"], y=df["informal_employment_pct"],
        fill="tozeroy",
        fillcolor="rgba(192,132,58,0.12)",
        line=dict(color=C["amber"], width=2.5),
        mode="lines+markers",
        marker=dict(size=8, color=C["amber"], line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Informal employment: <b>%{y:.1f}%</b><extra></extra>",
    ))
    add_events(fig2, events)
    base_layout(fig2, height=320)
    fig2.update_layout(
        yaxis=dict(title="Informal employment (%)", range=[60, 76]),
        xaxis=dict(title="Year", tickmode="linear", dtick=1),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)
    callout(
        "Why this matters most",
        "The informal employment share is the most resistant indicator in this index. "
        "While fintech and tax metrics show clear upward trends, labour market formalization lags significantly — "
        "pointing to a structural ceiling where workers and small enterprises gain digital access "
        "<em>without</em> entering regulated employment frameworks."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page_correlation(df):
    st.markdown('<div class="page-title">Correlation & Lag Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Does fintech adoption lead tax registration? · Cross-correlation analysis</div>', unsafe_allow_html=True)
    disclaimer()

    callout(
        "The research question",
        "If mobile banking adoption is an on-ramp to the formal economy, then growth in mobile banking should "
        "<em>precede</em> growth in tax registrations by 1–2 years. We can test this using cross-correlation at "
        "different time lags — negative lag means mobile banking lead."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    # ── Cross-correlation chart ───────────────────────────────────────────
    with col1:
        st.markdown('<div class="section-head">Cross-Correlation: Mobile Wallets → Tax Filers</div>', unsafe_allow_html=True)
        lags, corrs = cross_correlation(df["fbr_filers_m"], df["mobile_banking_users_m"], max_lag=4)

        bar_colors = []
        for lag, corr in zip(lags, corrs):
            if lag < 0 and corr > 0.8:
                bar_colors.append(C["green"])
            elif lag > 0:
                bar_colors.append(C["muted"])
            else:
                bar_colors.append(C["amber"])

        fig = go.Figure(go.Bar(
            x=lags, y=corrs,
            marker_color=bar_colors,
            hovertemplate="Lag %{x} yr: r = <b>%{y:.3f}</b><extra></extra>",
            width=0.6,
        ))
        fig.add_hline(y=0, line_color=C["border"], line_width=1)
        fig.add_hrect(y0=0.8, y1=1.0, fillcolor="rgba(74,124,89,0.06)", line_width=0)

        base_layout(fig, height=360)
        fig.update_layout(
            xaxis=dict(title="Lag (years) — negative = mobile banking lead", tickmode="linear", dtick=1),
            yaxis=dict(title="Pearson r", range=[-0.2, 1.1]),
            showlegend=False,
        )
        fig.add_annotation(x=-3.7, y=0.92, text="Strong correlation zone (r > 0.8)",
                           font=dict(size=9, color=C["green"]), showarrow=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: mobile banking vs FBR filers ─────────────────────────────
    with col2:
        st.markdown('<div class="section-head">Mobile Wallets vs. Tax Filers (by Year)</div>', unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["mobile_banking_users_m"],
            y=df["fbr_filers_m"],
            mode="markers+text",
            marker=dict(
                size=12,
                color=df["year"],
                colorscale=[[0, C["chart"][4]], [1, C["green"]]],
                showscale=True,
                colorbar=dict(title="Year", tickfont=dict(size=9)),
                line=dict(color="white", width=1),
            ),
            text=df["year"].astype(str),
            textposition="top right",
            textfont=dict(family="'DM Mono', monospace", size=9, color=C["muted"]),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Mobile banking: %{x:.1f}M<br>"
                "Tax filers: %{y:.2f}M<extra></extra>"
            ),
            showlegend=False,
        ))

        # Trendline
        z = np.polyfit(df["mobile_banking_users_m"], df["fbr_filers_m"], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df["mobile_banking_users_m"].min(), df["mobile_banking_users_m"].max(), 100)
        fig2.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode="lines",
            line=dict(color=C["rust"], width=1.5, dash="dash"),
            name="Trend",
            showlegend=False,
        ))

        base_layout(fig2, height=360)
        fig2.update_layout(
            xaxis=dict(title="Mobile wallet users (M)"),
            yaxis=dict(title="FBR tax filers (M)"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── YoY Growth Comparison ──────────────────────────────────────────────
    st.markdown('<div class="section-head">Year-on-Year Growth: Digital Payments vs. Tax Filers</div>', unsafe_allow_html=True)

    df_growth = df.dropna(subset=["digital_payment_vol_bpkr_yoy", "fbr_filers_m_yoy"]).copy()

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=df_growth["year"], y=df_growth["digital_payment_vol_bpkr_yoy"],
        name="Digital Payment Volume",
        marker_color=C["blue"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Digital payments YoY: +%{y:.1f}%<extra></extra>",
        offsetgroup=1,
    ))
    fig3.add_trace(go.Bar(
        x=df_growth["year"], y=df_growth["fbr_filers_m_yoy"],
        name="FBR Tax Filers",
        marker_color=C["green"],
        opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Tax filers YoY: +%{y:.1f}%<extra></extra>",
        offsetgroup=2,
    ))

    base_layout(fig3, height=360)
    fig3.update_layout(
        barmode="group",
        xaxis=dict(title="Year", tickmode="linear", dtick=1),
        yaxis=dict(title="YoY Growth (%)"),
        legend=dict(orientation="h", y=-0.18),
    )
    st.plotly_chart(fig3, use_container_width=True)

    callout(
        "Key finding: The gap is widening",
        "Digital payment volumes grew as fast as <strong>89% YoY</strong> (2021), while tax filer growth peaked at "
        "<strong>35% YoY</strong> (2018). The persistent gap suggests that fintech adoption is accelerating faster than "
        "the fiscal system can absorb new participants — pointing to a need for better linking of digital financial identity to tax registration."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: REGIONAL SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────
def page_regional(regional):
    st.markdown('<div class="page-title">Regional Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Estimated formalization scores by province · approximate figures</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠ Regional data is estimated from SBP Banking Surveillance Reports and FBR provincial breakdowns.
        Province-level data is significantly less granular than national data — treat these as directional estimates.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 0.8], gap="large")

    with col1:
        st.markdown('<div class="section-head">Composite Formalization Score by Province</div>', unsafe_allow_html=True)

        reg = regional.sort_values("formalization_score")
        colors = [C["amber"] if s < 30 else C["rust"] if s < 50 else C["green"] for s in reg["formalization_score"]]

        fig = go.Figure(go.Bar(
            x=reg["formalization_score"],
            y=reg["province"],
            orientation="h",
            marker_color=colors,
            text=[f"{s}" for s in reg["formalization_score"]],
            textposition="outside",
            textfont=dict(family="'DM Mono', monospace", size=11),
            hovertemplate="<b>%{y}</b><br>Score: %{x}/100<extra></extra>",
        ))
        base_layout(fig, height=380)
        fig.update_layout(
            xaxis=dict(title="Formalization Score (0–100)", range=[0, 100]),
            yaxis=dict(title=None),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">Provincial Breakdown</div>', unsafe_allow_html=True)
        rows = ""
        for _, row in regional.sort_values("formalization_score", ascending=False).iterrows():
            rows += f"""
            <tr>
                <td><strong>{row['province']}</strong></td>
                <td>{row['mobile_penetration_pct']}%</td>
                <td>{row['tax_filers_share_pct']}%</td>
                <td>{row['formalization_score']}</td>
            </tr>"""
        st.markdown(f"""
        <table class="styled-table">
            <thead><tr>
                <th>Province</th>
                <th>Mobile Pen.</th>
                <th>Tax Filers</th>
                <th>Score</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        callout(
            "A tale of two Pakistans",
            "Islamabad scores <strong>78/100</strong> while Balochistan scores <strong>15/100</strong>. "
            "Mobile penetration explains most of this gap — provinces with poor connectivity "
            "are effectively locked out of fintech-led formalization entirely."
        )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
def page_methodology():
    st.markdown('<div class="page-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">How the Composite Formalization Index is constructed</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="section-head">Index Construction</div>', unsafe_allow_html=True)
        st.markdown("""
        The Composite Formalization Index aggregates **five indicators** into a single 0–100 score:

        1. **Normalize** each indicator to [0, 1] using Min-Max scaling over the 2015–2025 range.
        2. **Invert** the informal employment share (lower = more formal).
        3. **Weighted sum** of all normalized indicators.
        4. **Scale** to 0–100 for interpretability.

        A score of **0** represents 2015 baseline levels; **100** represents the theoretical maximum observed within the dataset.
        """)

        # Weight visualization
        weights = list(INDEX_WEIGHTS.values())
        labels  = [
            "FBR Tax Filers",
            "Mobile Banking Users",
            "Digital Payment Volume",
            "SME Bank Credit Share",
            "Informal Employment (inv.)",
        ]
        fig = go.Figure(go.Bar(
            x=weights,
            y=labels,
            orientation="h",
            marker_color=C["chart"],
            text=[f"{w*100:.0f}%" for w in weights],
            textposition="outside",
            textfont=dict(family="'DM Mono', monospace", size=11),
        ))
        base_layout(fig, height=280)
        fig.update_layout(
            xaxis=dict(title="Weight", range=[0, 0.4], tickformat=".0%"),
            yaxis=dict(title=None),
            showlegend=False,
            margin=dict(l=180, r=60, t=10, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">Data Sources</div>', unsafe_allow_html=True)
        st.markdown("""
        <table class="styled-table">
            <thead><tr><th>Indicator</th><th>Source</th><th>Frequency</th></tr></thead>
            <tbody>
                <tr><td>FBR Tax Filers</td><td>Federal Board of Revenue (FBR) — Annual Statistics</td><td>Annual</td></tr>
                <tr><td>Mobile Banking Users</td><td>SBP — Quarterly Payment Systems Review</td><td>Quarterly</td></tr>
                <tr><td>Digital Payment Volume</td><td>SBP — Quarterly Payment Systems Review</td><td>Quarterly</td></tr>
                <tr><td>SME Credit Share</td><td>SBP — Banking Surveillance Report</td><td>Bi-annual</td></tr>
                <tr><td>Informal Employment</td><td>PBS — Labour Force Survey</td><td>Annual</td></tr>
                <tr><td>Raast Users</td><td>SBP — Digital Financial Services Reports</td><td>Quarterly</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-head" style="margin-top:1.5rem">Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Index weights** are researcher-assigned; sensitivity analysis (varying weights ±10%) is recommended for robustness.
        - **Labour Force Survey** has 2–3 year publication lags — 2023/2024 employment figures are interpolated.
        - **Regional data** is derived from fragmented provincial reports; treat as directional estimates only.
        - The index measures **access to formal systems**, not full integration — a mobile wallet user is not necessarily a formal economic actor.
        - **Causality** cannot be established from observational cross-correlation alone; confounders (GDP growth, urbanization) are not controlled.
        """)

    st.markdown('<div class="section-head">Suggested Extensions</div>', unsafe_allow_html=True)
    ecol1, ecol2, ecol3 = st.columns(3)
    extensions = [
        ("🧮", "Granger Causality Test", "Test formally whether mobile wallet growth Granger-causes tax registration using time-series econometrics."),
        ("🗺️", "District-Level Mapping", "If PBS releases district-level LFS microdata, plot formalization scores as a choropleth map of Pakistan."),
        ("📰", "Policy Sentiment Analysis", "Scrape Dawn/Geo news archives and correlate policy announcement sentiment with index movement."),
    ]
    for col, (icon, title, desc) in zip([ecol1, ecol2, ecol3], extensions):
        with col:
            st.markdown(f"""
            <div style="background:{C['card']}; border:1px solid {C['border']}; 
                        padding:1rem 1.2rem; border-radius:4px; height:100%;">
                <div style="font-size:1.5rem; margin-bottom:0.5rem;">{icon}</div>
                <div style="font-family:'Playfair Display',serif; font-size:0.95rem; 
                            color:{C['dark']}; margin-bottom:0.4rem;"><strong>{title}</strong></div>
                <div style="font-family:'DM Sans',sans-serif; font-size:0.82rem; 
                            color:{C['muted']}; line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR + MAIN
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding: 1.5rem 0 1rem 0;">
            <div style="font-family:'DM Mono',monospace; font-size:0.62rem;
                        color:#8B6355; letter-spacing:0.14em; text-transform:uppercase;">
                Data Science Project
            </div>
            <div style="font-family:Georgia,serif; font-size:1.1rem;
                        color:#F5F0E8; margin-top:0.4rem; line-height:1.4; font-weight:700;">
                Pakistan Informal Economy<br>Formalization Tracker
            </div>
        </div>
        <hr style="border:none; border-top:1px solid #2A2A2A; margin: 0.5rem 0 1.5rem 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigate",
            ["Overview", "Indicators", "Correlation Analysis", "Regional Snapshot", "Methodology"],
            label_visibility="collapsed",
        )

        st.markdown("""
        <hr style="border:none; border-top:1px solid #2A2A2A; margin:2rem 0 1rem 0;">
        <div style="font-family:'DM Mono',monospace; font-size:0.62rem; color:#444;
                    line-height:2; text-transform:uppercase; letter-spacing:0.04em;">
            Sources<br>
            — SBP Quarterly Reports<br>
            — FBR Annual Statistics<br>
            — PBS Labour Force Survey<br>
            — World Bank Findex<br>
            <br>
            Coverage · 2015–2025<br>
            Built with Streamlit + Plotly
        </div>
        """, unsafe_allow_html=True)

    return page


def main():
    df, events, regional = load_data()
    df_idx = build_index(df)
    page = sidebar()

    if   page == "Overview":            page_overview(df, events, df_idx)
    elif page == "Indicators":          page_indicators(df, events)
    elif page == "Correlation Analysis": page_correlation(df)
    elif page == "Regional Snapshot":   page_regional(regional)
    elif page == "Methodology":         page_methodology()


if __name__ == "__main__":
    main()
