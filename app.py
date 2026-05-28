"""
HW4 — HAR Model Dashboard
=========================
Streamlit dashboard for visualising every output of the HW4 notebook.
One tab per homework question; reads from the parquet bundle produced
by the export cell at the end of the notebook.

Run:
    pip install streamlit pandas pyarrow plotly numpy
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# =====================================================================
#  Page setup & theme
# =====================================================================
st.set_page_config(
    page_title="HAR & Realized Variance",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for editorial / research-paper aesthetic
st.markdown(
    """
    <style>
    /* Import editorial typography */
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Fraunces', serif !important;
        font-weight: 500 !important;
        letter-spacing: -0.02em;
    }

    h1 { font-size: 2.6rem !important; }

    code, .stCode, .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* Pill-style tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid rgba(180, 180, 180, 0.15);
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 14px;
        font-family: 'Fraunces', serif;
        font-size: 0.95rem;
        font-weight: 500;
        background-color: transparent;
        border-radius: 0;
        border-bottom: 2px solid transparent;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #c14a3a;
        color: #c14a3a !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(180, 180, 180, 0.12);
        padding: 14px 18px;
        border-radius: 4px;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.65;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Fraunces', serif !important;
        font-weight: 500;
        font-size: 1.8rem !important;
    }

    /* Decorative header rule */
    .hw-rule {
        height: 1px;
        background: linear-gradient(90deg, #c14a3a 0%, transparent 60%);
        margin: 6px 0 18px 0;
    }
    .hw-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        color: #c14a3a;
        margin-bottom: 4px;
    }
    .small-note {
        font-size: 0.85rem;
        opacity: 0.7;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Plotly aesthetic
PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#c14a3a"
INK = "#e8e6e1"
MUTED = "#8a8680"
SEQ = ["#c14a3a", "#e8e6e1", "#6a9b88", "#d4a045", "#7e7a92", "#9c5d8c"]


def styled_layout(title=None, height=None):
    """Return a Plotly layout dict matching the app aesthetic."""
    layout = dict(
        template=PLOTLY_TEMPLATE,
        font=dict(family="Inter, sans-serif", size=12, color=INK),
        margin=dict(l=40, r=20, t=50 if title else 20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(180,180,180,0.10)", zerolinecolor="rgba(180,180,180,0.15)"),
        yaxis=dict(gridcolor="rgba(180,180,180,0.10)", zerolinecolor="rgba(180,180,180,0.15)"),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(family="Fraunces, serif", size=18, color=INK))
    if height:
        layout["height"] = height
    return layout


# =====================================================================
#  Data loading
# =====================================================================
DATA_DIR = Path(__file__).parent / "data"

@st.cache_data(show_spinner=False)
def load_from_folder(folder: Path) -> dict[str, pd.DataFrame]:
    """Load every parquet file from the data folder into a dict of DataFrames."""
    tables: dict[str, pd.DataFrame] = {}
    for parquet_file in sorted(folder.glob("*.parquet")):
        name = parquet_file.stem
        if name == "manifest":
            continue
        tables[name] = pd.read_parquet(parquet_file)
    return tables


if not DATA_DIR.exists():
    st.error(f"Data folder not found: `{DATA_DIR}`")
    st.stop()

try:
    tables = load_from_folder(DATA_DIR)
except Exception as e:
    st.error(f"Could not read data folder: {e}")
    st.stop()

# Convenience aliases
rv = tables["rv"].copy();           rv["date"] = pd.to_datetime(rv["date"])
ln_rv = tables["ln_rv"].copy();     ln_rv["date"] = pd.to_datetime(ln_rv["date"])
stats_rv = tables["stats_rv"]
stats_ln = tables["stats_ln"]
adf = tables["adf"]
har = tables["har"]
forecasts = tables["forecasts"].copy()
forecasts["date"] = pd.to_datetime(forecasts["date"])
metrics = tables["metrics"]
summary = tables["summary"]
gc_pairs = tables.get("granger_pairs", pd.DataFrame())
gc_matrix = tables.get("granger_matrix", pd.DataFrame())
harx = tables.get("harx", pd.DataFrame())

TICKERS = sorted([c for c in rv.columns if c != "date"])


# =====================================================================
#  Header
# =====================================================================
st.markdown("# Modelling daily realized variance")
st.markdown('<div class="hw-rule"></div>', unsafe_allow_html=True)
st.markdown(
    f"""
    Univariate analysis of daily realized variance for **{len(TICKERS)} constituents of the Dow Jones Index**
    over **{rv['date'].min().date()} - {rv['date'].max().date()}**.
    """
)


# =====================================================================
#  Tabs (one per question)
# =====================================================================
tab_labels = [
    "RV",
    "RV stats",
    "ln RV",
    "ADF",
    "HAR",
    "OOS Forecasts",
    "ML",
    "Summary",
    "Granger",
    "HAR-X",
]
tabs = st.tabs(tab_labels)


# ---------------------------------------------------------------------
# Q1 — RV construction
# ---------------------------------------------------------------------
with tabs[0]:
    st.markdown("### Building daily realized variances")
    st.markdown(
        """
        For each stock we sample close prices every 5 minutes (78 prices per
        trading day), take log returns, and aggregate per day:
        $$RV_t = \\sum_{i=1}^{77} r_{t,i}^2$$
        """
    )

    sel = st.selectbox("Ticker", TICKERS, index=TICKERS.index("AAPL") if "AAPL" in TICKERS else 0, key="q1_tk")

    s = rv[["date", sel]].dropna().rename(columns={sel: "RV"})
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["date"], y=s["RV"], mode="lines",
        line=dict(color=ACCENT, width=0.8), name=sel,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>RV = %{y:.2e}<extra></extra>",
    ))
    fig.update_layout(**styled_layout(f"{sel} | Daily realized variance", height=420))
    st.plotly_chart(fig, use_container_width=True)

    # Quick summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Observations", f"{s['RV'].notna().sum():,}")
    m2.metric("First date", f"{s['date'].min().date()}")
    m3.metric("Last date", f"{s['date'].max().date()}")
    m4.metric("Max RV", f"{s['RV'].max():.2e}")

    with st.expander("Cross-sectional view: all stocks time series"):
        df_long = rv.melt(id_vars="date", var_name="ticker", value_name="RV").dropna()
        fig2 = px.line(df_long, x="date", y="RV", color="ticker",
                       facet_col="ticker", facet_col_wrap=5,
                       color_discrete_sequence=[ACCENT] * 30,
                       height=900)
        fig2.update_yaxes(matches=None, showticklabels=False)
        fig2.update_xaxes(showticklabels=False)
        fig2.update_traces(line=dict(width=0.4))
        fig2.update_layout(showlegend=False, **{k: v for k, v in styled_layout().items() if k not in ("xaxis","yaxis")})
        fig2.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=10, family="Inter")))
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------
# Q2 — Descriptive statistics of RV
# ---------------------------------------------------------------------
with tabs[1]:
    st.markdown("### Descriptive statistics of the realized variances")
    st.markdown("Raw RV is *strongly right-skewed*, with *sharp peaks* and *heavy tails*.")

    st.dataframe(
        stats_rv.set_index(stats_rv.columns[0]).rename_axis(None).style.format({
            "mean": "{:.2e}", "std": "{:.2e}", "min": "{:.2e}", "max": "{:.2e}",
            "skew": "{:.2f}", "kurtosis": "{:.2f}", "n": "{:,.0f}",
        }),
        height=560, use_container_width=True,
    )

    st.markdown("##### Distribution of RV for a single stock")
    sel_q2 = st.selectbox("Ticker", TICKERS, index=TICKERS.index("AAPL") if "AAPL" in TICKERS else 0, key="q2_tk")
    rv_vals = rv[sel_q2].dropna().values
    mu, sigma = rv_vals.mean(), rv_vals.std()
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=rv_vals, nbinsx=80, histnorm="probability density",
        marker_color=ACCENT, opacity=0.75, name="Empirical",
        hovertemplate="RV = %{x:.2e}<br>density = %{y:.2f}<extra></extra>",
    ))
    xs = np.linspace(rv_vals.min(), rv_vals.max(), 300)
    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    fig_dist.add_trace(go.Scatter(
        x=xs, y=gauss, mode="lines",
        line=dict(color=INK, width=2, dash="dash"), name="Gaussian",
    ))
    sk = float(stats_rv.loc[stats_rv[stats_rv.columns[0]] == sel_q2, "skew"].values[0]) if sel_q2 in stats_rv[stats_rv.columns[0]].values else float("nan")
    kt = float(stats_rv.loc[stats_rv[stats_rv.columns[0]] == sel_q2, "kurtosis"].values[0]) if sel_q2 in stats_rv[stats_rv.columns[0]].values else float("nan")
    fig_dist.update_layout(**styled_layout(f"{sel_q2} | RV distribution  (skew {sk:.2f}, excess kurt. {kt:.2f})", height=360))
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("##### Distribution of the moments across the 30 stocks")
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Skewness", "Excess kurtosis"])
    for col, color, c in zip(["skew", "kurtosis"], [ACCENT, "#6a9b88"], [1, 2]):
        fig.add_trace(go.Box(
            y=stats_rv[col], name=col,
            marker_color=color, boxmean=True, boxpoints="all", jitter=0.4,
            hovertemplate="%{y:.2f}<extra></extra>", showlegend=False,
        ), row=1, col=c)
    fig.update_layout(**styled_layout(height=360))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Q3 — ln(RV)
# ---------------------------------------------------------------------
with tabs[2]:
    st.markdown("### Logarithm of the realized variances")
    st.markdown(
        "The log transformation reduces skewness and heavy tails, making the series approximate a Gaussian distribution and better suited for modelling."
    )

    sel = st.selectbox("Ticker", TICKERS, index=TICKERS.index("AAPL") if "AAPL" in TICKERS else 0, key="q3_tk")
    s = ln_rv[["date", sel]].dropna().rename(columns={sel: "lnRV"})

    c1, c2 = st.columns([1.5, 1])

    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["lnRV"], mode="lines",
            line=dict(color=ACCENT, width=0.8),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>lnRV = %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(**styled_layout(f"{sel} | ln(RV)", height=380))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Histogram + Gaussian reference
        x = s["lnRV"].values
        mu, sigma = x.mean(), x.std()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=x, nbinsx=60, marker_color=ACCENT, opacity=0.7,
                                   histnorm="probability density", name="Empirical"))
        xs = np.linspace(x.min(), x.max(), 200)
        gauss = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xs - mu)/sigma)**2)
        fig.add_trace(go.Scatter(x=xs, y=gauss, mode="lines",
                                 line=dict(color=INK, width=2, dash="dash"),
                                 name="Gaussian"))
        fig.update_layout(**styled_layout("Distribution vs. Gaussian", height=380))
        st.plotly_chart(fig, use_container_width=True)

    # Sample ACF computed on-the-fly
    st.markdown("##### Sample autocorrelation function (ACF) of ln(RV)")
    n_lags = st.slider("Number of lags", 20, 100, 60, key="q3_lags")
    acf_vals = [s["lnRV"].autocorr(lag=k) for k in range(1, n_lags + 1)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, n_lags+1)), y=acf_vals,
                         marker_color=ACCENT, marker_line_width=0,
                         hovertemplate="lag %{x}<br>ρ = %{y:.3f}<extra></extra>"))
    # 95% confidence band  ±1.96/√N
    band = 1.96 / np.sqrt(len(s))
    fig.add_hline(y=band, line_dash="dot", line_color=MUTED)
    fig.add_hline(y=-band, line_dash="dot", line_color=MUTED)
    fig.update_layout(**styled_layout(f"{sel} | Sample ACF of ln(RV)", height=300))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full descriptive statistics for ln(RV)"):
        st.dataframe(
            stats_ln.set_index(stats_ln.columns[0]).rename_axis(None).style.format("{:.3f}"),
            height=540, use_container_width=True,
        )


# ---------------------------------------------------------------------
# Q4 — ADF tests
# ---------------------------------------------------------------------
with tabs[3]:
    st.markdown("### Augmented Dickey-Fuller tests")
    st.markdown(
        "We test for a unit root in ln(RV) with intercept and AIC-selected lag length. "
        "The null hypothesis is that the series is non-stationary (has a unit root), so small p-values indicate strong evidence of stationarity and justify time-series modelling."
    )

    n_reject = int(adf["reject_5pct"].sum()) if "reject_5pct" in adf.columns else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Stocks tested", f"{len(adf)}")
    c2.metric("Reject unit root at 5 %", f"{n_reject}/{len(adf)}")
    c3.metric("Median p-value", f"{adf['p_value'].median():.2e}")

    # Bar chart of ADF stats with critical value line
    adf_sorted = adf.sort_values("ADF_stat")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=adf_sorted["ticker"], y=adf_sorted["ADF_stat"],
        marker_color=[ACCENT if r else MUTED for r in adf_sorted["reject_5pct"]],
        hovertemplate="<b>%{x}</b><br>ADF = %{y:.2f}<extra></extra>",
    ))
    crit = adf["crit_5pct"].iloc[0]
    fig.add_hline(y=crit, line_dash="dash", line_color="#d4a045",
                  annotation_text=f"5% critical = {crit:.2f}",
                  annotation_position="top right")
    fig.update_layout(**styled_layout("ADF statistic by stock",
                                      height=420))
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(adf.set_index(adf.columns[0]).rename_axis(None).style.format({
        "ADF_stat": "{:.3f}", "p_value": "{:.2e}", "crit_5pct": "{:.3f}",
        "lag_used": "{:.0f}", "n_obs": "{:,.0f}",
    }), height=400, use_container_width=True)


# ---------------------------------------------------------------------
# Q5 — HAR coefficients
# ---------------------------------------------------------------------
with tabs[4]:
    st.markdown("### HAR model: full-sample estimation")
    st.latex(r"y_t = \beta_0 + \beta_1\,y_{t-1} + \beta_2\,y^{(5)}_{t-1} + \beta_3\,y^{(22)}_{t-1} + \varepsilon_t")
    st.markdown(
        "Daily, weekly and monthly lag aggregates."
    )

    # Cross-sectional summary
    means = har[["beta0","beta1","beta2","beta3","R2","sum_betas"]].mean()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("β₀ (mean)", f"{means['beta0']:.3f}")
    c2.metric("β₁ daily",   f"{means['beta1']:.3f}")
    c3.metric("β₂ weekly",  f"{means['beta2']:.3f}")
    c4.metric("β₃ monthly", f"{means['beta3']:.3f}")
    c5.metric("Σβᵢ",         f"{means['sum_betas']:.3f}")
    c6.metric("Mean R²",     f"{means['R2']:.3f}")

    # Bar chart per beta
    st.markdown("##### β coefficients across stocks")
    melt = har.melt(id_vars="ticker", value_vars=["beta1","beta2","beta3"],
                    var_name="coef", value_name="value")
    melt["coef"] = melt["coef"].map({"beta1":"β₁ daily","beta2":"β₂ weekly","beta3":"β₃ monthly"})
    fig = px.bar(melt, x="ticker", y="value", color="coef", barmode="group",
                 color_discrete_sequence=[ACCENT, "#6a9b88", "#d4a045"])
    fig.update_layout(**styled_layout(height=380))
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Full table
    fmt = {c: "{:.3f}" for c in ["beta0","beta1","beta2","beta3",
                                  "se_b0","se_b1","se_b2","se_b3",
                                  "R2","R2_adj","sum_betas"]}
    fmt["n"] = "{:,.0f}"
    st.dataframe(har.set_index("ticker").rename_axis(None).style.format(fmt), height=440, use_container_width=True)


# ---------------------------------------------------------------------
# Q6-Q8 — Linear OOS forecasts
# ---------------------------------------------------------------------
with tabs[5]:
    st.markdown("### Out-of-sample 1-step forecasts")
    st.markdown(
        "Expanding window over the second half of each sample. "
        "Three benchmarks: **HAR**, **AR(1)**, and **Random Walk**."
    )

    sel = st.selectbox("Ticker", TICKERS, index=TICKERS.index("AAPL") if "AAPL" in TICKERS else 0, key="q6_tk")
    fc = forecasts[forecasts["ticker"] == sel].sort_values("date")

    if fc.empty:
        st.warning("No forecasts for this ticker.")
    else:
        all_models = [c for c in fc.columns if c.startswith("yhat_")]
        labels = {"yhat_har":"HAR","yhat_ar1":"AR(1)","yhat_rw":"Random Walk",
                  "yhat_ridge":"Ridge","yhat_rf":"Random Forest","yhat_gbm":"LightGBM"}
        model_names = [labels[c] for c in all_models if c in labels]
        chosen_name = st.selectbox("Model", model_names, key="q6_model")
        chosen_col = next(c for c in all_models if labels.get(c) == chosen_name)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fc["date"], y=fc["y_true"], mode="lines",
                                 name="Ground Truth", line=dict(color=INK, width=0.8)))
        fig.add_trace(go.Scatter(x=fc["date"], y=fc[chosen_col], mode="lines",
                                 name=chosen_name,
                                 line=dict(color=ACCENT, width=1.2),
                                 opacity=0.85))
        fig.update_layout(**styled_layout(f"{sel} | Actual ln(RV) vs. {chosen_name} predictions", height=440))
        st.plotly_chart(fig, use_container_width=True)

        err = fc["y_true"] - fc[chosen_col]
        mt = pd.DataFrame([{"MFE": err.mean(), "RMSE": np.sqrt((err**2).mean()), "MAE": err.abs().mean()}],
                          index=[chosen_name])
        st.dataframe(mt.style.format("{:.4f}"), use_container_width=True)

    # Cross-sectional RMSE comparison (linear models only)
    st.markdown("##### RMSE comparison across all stocks (benchmark models only)")
    lin_cols = [c for c in metrics.columns if any(k in c for k in ["har","ar1","rw"]) and c.endswith("_RMSE")]
    if lin_cols:
        df = metrics[["ticker"] + lin_cols].copy()
        df.columns = ["ticker"] + [c.replace("yhat_","").replace("_RMSE","").upper() for c in lin_cols]
        df_long = df.melt(id_vars="ticker", var_name="model", value_name="RMSE")
        fig = px.bar(df_long, x="ticker", y="RMSE", color="model", barmode="group",
                     color_discrete_sequence=[ACCENT, "#6a9b88", MUTED])
        fig.update_layout(**styled_layout(height=400))
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Q9 — ML models
# ---------------------------------------------------------------------
with tabs[6]:
    st.markdown("### Machine Learning models results")
    st.markdown(
        "Three ML families:"
    )
    st.markdown(
        "- **Ridge**: same family as HAR (linear w/ L2 regularization and additional features), captures linearity but not local patterns.\n"
        "- **Random Forest**: bagged decision trees, captures local nonlinearity.\n"
        "- **LightGBM**: gradient-boosted trees, captures local nonlinearity and patterns in the errors."
    )

    ml_cols = [c for c in metrics.columns if c.endswith("_RMSE")]
    label_map = {"yhat_har_RMSE":"HAR","yhat_ar1_RMSE":"AR(1)","yhat_rw_RMSE":"RW",
                 "yhat_ridge_RMSE":"Ridge","yhat_rf_RMSE":"RF","yhat_gbm_RMSE":"LGBM"}
    df = metrics[["ticker"] + ml_cols].copy()
    df.columns = ["ticker"] + [label_map.get(c, c) for c in ml_cols]
    df = df.set_index("ticker")

    # RMSE relative to HAR
    if "HAR" in df.columns:
        rel = df.divide(df["HAR"], axis=0)
        rel_sorted = rel.drop(columns="HAR").mean().sort_values()

        st.markdown("##### Average RMSE relative to HAR (< 1 means model beats HAR)")
        fig = go.Figure()
        colors = [ACCENT if v < 1 else MUTED for v in rel_sorted.values]
        fig.add_trace(go.Bar(x=rel_sorted.index, y=rel_sorted.values,
                             marker_color=colors,
                             hovertemplate="%{x}<br>relative RMSE = %{y:.3f}<extra></extra>"))
        fig.add_hline(y=1.0, line_dash="dash", line_color=INK)
        fig.update_layout(**styled_layout(height=330))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap of relative RMSE per stock
    st.markdown("##### Stock-by-stock relative RMSE")
    if "HAR" in df.columns:
        rel = df.divide(df["HAR"], axis=0).drop(columns="HAR")
        fig = px.imshow(rel.T, color_continuous_scale="RdGy",
                        zmin=0.85, zmax=1.15, aspect="auto",
                        labels=dict(x="ticker", y="model", color="RMSE / HAR"))
        fig.update_layout(**styled_layout(height=320))
        st.plotly_chart(fig, use_container_width=True)

    # Win counts
    if "best_model" in summary.columns:
        wins = summary["best_model"].value_counts()
        st.markdown("##### Number of stocks where each model wins")
        fig = px.bar(x=wins.index, y=wins.values,
                     color_discrete_sequence=[ACCENT], labels={"x":"model","y":"# stocks"})
        fig.update_layout(**styled_layout(height=300))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Q10 — Summary
# ---------------------------------------------------------------------
with tabs[7]:
    st.markdown("### Best model per stock")
    st.markdown(
        "RMSEs of every model side-by-side, plus the winning model per stock."
    )

    # Show summary with winners highlighted
    rmse_cols = [c for c in summary.columns if c.upper() == c and c != "BEST_MODEL"]
    rmse_cols = [c for c in summary.columns if c in ["HAR","AR(1)","RW","RIDGE","RF","GBM"]]

    show = summary.set_index(summary.columns[0]).rename_axis(None).copy()
    fmt = {c: "{:.4f}" for c in show.columns if c not in ["best_model"]}

    def highlight_winner(row):
        styles = [""] * len(row)
        if "best_model" in row.index:
            winner = row["best_model"]
            if winner in row.index:
                idx = list(row.index).index(winner)
                styles[idx] = f"background-color: rgba(193,74,58,0.25); font-weight: 600;"
        return styles

    st.dataframe(show.style.format(fmt).apply(highlight_winner, axis=1),
                 height=560, use_container_width=True)

    # Mean RMSE & wins
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Mean RMSE across stocks")
        rmse_only = show[[c for c in rmse_cols if c in show.columns]].mean().sort_values()
        fig = go.Figure(go.Bar(x=rmse_only.index, y=rmse_only.values,
                               marker_color=ACCENT,
                               hovertemplate="%{x}<br>%{y:.4f}<extra></extra>"))
        fig.update_layout(**styled_layout(height=300))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Best-model wins")
        wins = show["best_model"].value_counts()
        fig = go.Figure(go.Bar(x=wins.index, y=wins.values,
                               marker_color="#6a9b88",
                               hovertemplate="%{x}<br>%{y} stocks<extra></extra>"))
        fig.update_layout(**styled_layout(height=300))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Q11 — Granger
# ---------------------------------------------------------------------
with tabs[8]:
    st.markdown("### Granger causality")
    st.markdown(
        "We test whether the lagged ln(RV) of one stock contains information for "
        "predicting another, beyond the target's own past. p-values < 0.05 indicate **Granger causality**."
    )

    if not gc_pairs.empty:
        st.markdown("##### Tested pairs (5 lags, both directions)")
        st.dataframe(
            gc_pairs.set_index(gc_pairs.columns[0]).rename_axis(None).style.format({
                **{c: "{:.4f}" for c in gc_pairs.columns if c.startswith("p_lag") or c == "min_p"},
                "best_lag": "{:.0f}",
            }),
            use_container_width=True,
        )

    if not gc_matrix.empty:
        st.markdown("##### Full pairwise Granger heatmap (lag 1)")
        st.markdown(
            "<span class='small-note'>Cell [row, col] = p-value for "
            "*row Granger-causes col*. Darker = stronger evidence.</span>",
            unsafe_allow_html=True,
        )
        mat = gc_matrix.set_index(gc_matrix.columns[0]).rename_axis(None)
        # Cap & log for visualisation
        mat_clip = mat.clip(lower=1e-50)
        mat_log = -np.log10(mat_clip)
        fig = px.imshow(mat_log, color_continuous_scale="OrRd",
                        labels=dict(x="target", y="cause", color="-log10(p)"),
                        aspect="auto")
        fig.update_layout(**styled_layout(height=620))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Q12 — HAR-X
# ---------------------------------------------------------------------
with tabs[9]:
    st.markdown("### HAR-X: does adding a Granger-causing series help?")
    st.markdown(
        "For each pair identified as Granger-causal in Q11, we augment the HAR model "
        "with the daily, weekly, and monthly lags of the causing stock's ln(RV) and compare OOS RMSE against plain HAR."
    )

    if not harx.empty:
        h = harx.copy()
        # Relative RMSE: <1 = HAR-X wins
        if "rel_RMSE_HARX_vs_HAR" not in h.columns and {"RMSE_HAR_X","RMSE_HAR"} <= set(h.columns):
            h["rel_RMSE_HARX_vs_HAR"] = h["RMSE_HAR_X"] / h["RMSE_HAR"]
        h_sorted = h.sort_values("rel_RMSE_HARX_vs_HAR")

        fig = go.Figure()
        colors = [ACCENT if v < 1 else MUTED for v in h_sorted["rel_RMSE_HARX_vs_HAR"]]
        fig.add_trace(go.Bar(x=h_sorted["direction"], y=h_sorted["rel_RMSE_HARX_vs_HAR"],
                             marker_color=colors,
                             hovertemplate="<b>%{x}</b><br>RMSE(HAR-X)/RMSE(HAR) = %{y:.4f}<extra></extra>"))
        fig.add_hline(y=1.0, line_dash="dash", line_color=INK)
        fig.update_layout(**styled_layout("Relative RMSE: HAR-X vs HAR (< 1 means HAR-X wins)", height=380))
        st.plotly_chart(fig, use_container_width=True)

        fmt = {c: "{:.4f}" for c in h.columns if any(k in c for k in ["RMSE","MFE","rel"])}
        st.dataframe(h.style.format(fmt), use_container_width=True)
    else:
        st.info("No HAR-X results in the bundle.")


# =====================================================================
#  Footer
# =====================================================================
st.markdown("---")

