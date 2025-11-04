import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly
import plotly.graph_objects as go

# -----------------------------
# Config
# -----------------------------

FACTOR_SUFFIXES = ["Peer Rank"]  # we now use *Peer Rank* columns, not *Score*
WEIGHT_COLUMN_GUESS = ["Abs Weight","Absolute Weight","Weight","Core Weight","Active Weight","AbsWeight","Wt","Weight (%)"]

def _locate_weight_column(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in WEIGHT_COLUMN_GUESS:
        key = candidate.lower()
        if key in cols_lower:
            return cols_lower[key]
    if len(df.columns) >= 2:
        colB = df.columns[1]
        if pd.to_numeric(df[colB], errors="coerce").notna().mean() > 0.6:
            return colB
    return st.selectbox("Select weight column", df.columns, index=1 if len(df.columns)>1 else 0)

def _detect_factor_columns(df: pd.DataFrame) -> list:
    factors = []
    for c in df.columns:
        cname = str(c).strip()
        if any(cname.endswith(sfx) for sfx in FACTOR_SUFFIXES):
            factors.append(cname)
    if not factors:
        candidates = [c for c in df.columns if "rank" in str(c).lower()]
        factors = st.multiselect("Select factor rank columns", candidates, default=candidates)
    return factors

def _read_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    # Try header rows 1 then 0 (Excel row 2 then 1) to match "C2:N2" hint
    for header_row in [1,0,2]:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=header_row)
            if df is not None and df.shape[1] > 1:
                return df
        except Exception:
            pass
    return pd.read_excel(xls, sheet_name=sheet_name, header=0)

def load_portfolio_and_benchmark(file: io.BytesIO, alpha_sheet="alpha", core_sheet="core") -> tuple:
    xls = pd.ExcelFile(file)
    sheets_lower = {s.lower(): s for s in xls.sheet_names}
    a_sheet = sheets_lower.get(alpha_sheet.lower())
    c_sheet = sheets_lower.get(core_sheet.lower())
    if a_sheet is None or c_sheet is None:
        st.error(f"Could not find sheets named '{alpha_sheet}' and '{core_sheet}' (case-insensitive).")
        st.stop()

    alpha_raw = _read_sheet(xls, a_sheet)
    core_raw  = _read_sheet(xls, c_sheet)

    alpha = alpha_raw.copy()
    core  = core_raw.copy()

    alpha.rename(columns={alpha.columns[0]: "Ticker"}, inplace=True)
    core.rename(columns={core.columns[0]: "Ticker"}, inplace=True)

    a_wcol = _locate_weight_column(alpha)
    c_wcol = _locate_weight_column(core)

    a_factors = _detect_factor_columns(alpha)
    c_factors = _detect_factor_columns(core)
    factor_cols = [f for f in a_factors if f in c_factors] or list(dict.fromkeys(a_factors + c_factors))

    def _prep(df, wcol):
        out = df.copy()
        out[wcol] = pd.to_numeric(out[wcol], errors="coerce")
        out = out[(out["Ticker"].astype(str).str.strip()!="") & out[wcol].notna()]
        if out[wcol].sum() > 2.0:
            out[wcol] = out[wcol] / 100.0
        return out

    alpha = _prep(alpha, a_wcol)
    core  = _prep(core, c_wcol)

    if alpha[a_wcol].sum() != 0:
        alpha[a_wcol] = alpha[a_wcol] / alpha[a_wcol].sum()
    if core[c_wcol].sum() != 0:
        core[c_wcol] = core[c_wcol] / core[c_wcol].sum()

    return alpha, a_wcol, core, c_wcol, factor_cols

def _rank_to_value(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    val = s / 100.0
    return val.clip(0,1)

def compute_weighted_exposures(df: pd.DataFrame, wcol: str, factor_cols: list) -> pd.Series:
    exp = {}
    for f in factor_cols:
        vals = _rank_to_value(df[f])
        exp[f] = (df[wcol] * vals).sum(skipna=True)
    return pd.Series(exp)

def compute_quintile_labels(universe: pd.DataFrame, factor: str) -> pd.DataFrame:
    u = universe.copy()
    vals = _rank_to_value(u[factor])  # higher = better
    try:
        labels = [f"Q{i}" for i in range(1,6)]
        qb = pd.qcut(vals, q=5, labels=labels, duplicates="drop")
    except Exception:
        quantiles = vals.rank(pct=True)
        qb = pd.cut(quantiles, bins=[0,0.2,0.4,0.6,0.8,1.0], labels=[f"Q{i}" for i in range(1,6)], include_lowest=True)
    u["Quintile"] = qb.astype(str)
    return u[["Ticker","Quintile"]]

def compute_quintile_weights(universe: pd.DataFrame, sheet_df: pd.DataFrame, wcol: str, factor: str) -> pd.Series:
    lab = compute_quintile_labels(universe, factor)
    df = sheet_df.merge(lab, on="Ticker", how="left")
    out = df.groupby("Quintile")[wcol].sum()
    return out.reindex([f"Q{i}" for i in range(1,6)], fill_value=0.0)

def make_exposure_bar(port_exp: pd.Series, bench_exp: pd.Series) -> plotly.graph_objects.Figure:
    factors = list(port_exp.index)
    fig = go.Figure()
    fig.add_bar(name="Portfolio", x=factors, y=port_exp.values, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    fig.add_bar(name="Benchmark", x=factors, y=bench_exp.values, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    active = port_exp - bench_exp
    fig.add_scatter(name="Active", x=factors, y=active.values, mode="lines+markers", yaxis="y2", hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    fig.update_layout(
        barmode="group",
        title="Weighted Factor Exposures (Peer Rank → value 0–1)",
        yaxis_title="Exposure",
        yaxis2=dict(title="Active", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def make_quintile_chart(alpha_q: pd.Series, core_q: pd.Series, title: str) -> plotly.graph_objects.Figure:
    qs = list(alpha_q.index)
    fig = go.Figure()
    fig.add_bar(name="Portfolio", x=qs, y=alpha_q.values, hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
    fig.add_bar(name="Benchmark", x=qs, y=core_q.values, hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
    active = alpha_q - core_q
    fig.add_scatter(name="Active", x=qs, y=active.values, mode="lines+markers", yaxis="y2", hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
    fig.update_layout(
        barmode="group",
        title=title,
        yaxis_title="Weight share",
        yaxis_tickformat=".0%",
        yaxis2=dict(title="Active", tickformat=".0%", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="ASR - Factor Exposure Dashboard", layout="wide")

with st.sidebar:
    logo_path = Path("ASR_Nederland_logo.svg.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.caption("Place 'ASR_Nederland_logo.svg.png' next to this script to show the logo.")

    st.markdown("### Data")
    uploaded = st.file_uploader("Excel file", type=["xlsx","xlsm","xls"], key="exposures")
    alpha_name = st.text_input("Portfolio sheet", value="alpha")
    core_name  = st.text_input("Benchmark sheet", value="core")

cols = st.columns([1,5])
with cols[0]:
    if Path("ASR_Nederland_logo.svg.png").exists():
        st.image("ASR_Nederland_logo.svg.png", use_container_width=True)
with cols[1]:
    st.title("ASR Factor Exposure Dashboard")

if not uploaded:
    st.info("Upload your Excel workbook to begin.")
    st.stop()

alpha, a_wcol, core, c_wcol, factor_cols = load_portfolio_and_benchmark(uploaded, alpha_sheet=alpha_name, core_sheet=core_name)
if not factor_cols:
    st.error("No factor score columns found (looking for '*Peer Rank').")
    st.stop()

alpha_exp  = compute_weighted_exposures(alpha, a_wcol, factor_cols)
core_exp   = compute_weighted_exposures(core,  c_wcol, factor_cols)
active_exp = alpha_exp - core_exp

st.subheader("Weighted Exposures")
summary_df = pd.DataFrame({"Portfolio": alpha_exp, "Benchmark": core_exp, "Active": active_exp}).sort_index()
st.dataframe(summary_df.style.format("{:.3f}"))
st.plotly_chart(make_exposure_bar(alpha_exp, core_exp), use_container_width=True)

universe = pd.concat([alpha[["Ticker"]+factor_cols], core[["Ticker"]+factor_cols]], axis=0, ignore_index=True).drop_duplicates("Ticker")
weights = pd.merge(universe[["Ticker"]], alpha[["Ticker", a_wcol]].rename(columns={a_wcol:"W_port"}), on="Ticker", how="left")
weights = pd.merge(weights, core[["Ticker", c_wcol]].rename(columns={c_wcol:"W_bench"}), on="Ticker", how="left").fillna(0.0)

st.subheader("Drilldowns")
tab_factor, tab_quint = st.tabs(["Factor contributors", "Quintile constituents"])

with tab_factor:
    f = st.selectbox("Factor", factor_cols, key="factor_contrib")
    scores = universe[["Ticker", f]].copy()
    scores["Value"] = _rank_to_value(scores[f])
    dfc = weights.merge(scores[["Ticker","Value"]], on="Ticker", how="left")
    dfc["Port_Contrib"]  = dfc["W_port"]  * dfc["Value"]
    dfc["Bench_Contrib"] = dfc["W_bench"] * dfc["Value"]
    dfc["Active_Contrib"] = dfc["Port_Contrib"] - dfc["Bench_Contrib"]
    dfc = dfc.sort_values("Active_Contrib", ascending=False)

    k = st.slider("Show top N positive/negative", min_value=5, max_value=50, value=20, step=5)
    top_pos = dfc.head(k)
    top_neg = dfc.tail(k).sort_values("Active_Contrib")

    st.markdown("**Top positive active contributors**")
    st.dataframe(top_pos[["Ticker","W_port","W_bench","Port_Contrib","Bench_Contrib","Active_Contrib"]]
                 .style.format({"W_port":"{:.2%}","W_bench":"{:.2%}","Port_Contrib":"{:.3f}","Bench_Contrib":"{:.3f}","Active_Contrib":"{:.3f}"}))

    st.markdown("**Top negative active contributors**")
    st.dataframe(top_neg[["Ticker","W_port","W_bench","Port_Contrib","Bench_Contrib","Active_Contrib"]]
                 .style.format({"W_port":"{:.2%}","W_bench":"{:.2%}","Port_Contrib":"{:.3f}","Bench_Contrib":"{:.3f}","Active_Contrib":"{:.3f}"}))

with tab_quint:
    f_q = st.selectbox("Factor", factor_cols, key="factor_quint")
    labels = compute_quintile_labels(universe, f_q)
    alpha_qdf = alpha.merge(labels, on="Ticker", how="left")
    core_qdf  = core.merge(labels,  on="Ticker", how="left")

    q_choice = st.selectbox("Quintile", [f"Q{i}" for i in range(1,6)], index=4)
    def _prep(df, wcol):
        out = df[df["Quintile"]==q_choice].copy()
        out["Value"] = _rank_to_value(out[f_q])
        out["Contribution"] = out[wcol] * out["Value"]
        return out[["Ticker", wcol, "Value", "Contribution"]].rename(columns={wcol:"Weight"}).sort_values("Contribution", ascending=False)

    p_tbl = _prep(alpha_qdf, a_wcol)
    b_tbl = _prep(core_qdf,  c_wcol)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Portfolio — {q_choice} of {f_q}**")
        st.dataframe(p_tbl.style.format({"Weight":"{:.2%}","Value":"{:.3f}","Contribution":"{:.3f}"}))
    with c2:
        st.markdown(f"**Benchmark — {q_choice} of {f_q}**")
        st.dataframe(b_tbl.style.format({"Weight":"{:.2%}","Value":"{:.3f}","Contribution":"{:.3f}"}))

    p_sum = p_tbl["Weight"].sum()
    b_sum = b_tbl["Weight"].sum()
    st.markdown(f"**Weight share in {q_choice} — Portfolio:** {p_sum:.2%}  |  **Benchmark:** {b_sum:.2%}  |  **Active:** {(p_sum-b_sum):.2%}")

st.subheader("Quintile Weight Distribution")
factor_choice = st.selectbox("Choose a factor for chart", factor_cols, index=0, key="factor_chart")
alpha_q = compute_quintile_weights(universe, alpha[["Ticker", a_wcol, factor_choice]].rename(columns={a_wcol:"Weight"}), "Weight", factor_choice)
core_q  = compute_quintile_weights(universe, core[["Ticker",  c_wcol, factor_choice]].rename(columns={c_wcol:"Weight"}), "Weight", factor_choice)

if alpha_q.sum() > 0:
    alpha_q = alpha_q / alpha_q.sum()
if core_q.sum() > 0:
    core_q = core_q / core_q.sum()
st.plotly_chart(make_quintile_chart(alpha_q, core_q, f"Quintile Weights by {factor_choice}"), use_container_width=True)

st.subheader("Download Results")
out_xlsx = io.BytesIO()
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    summary_df.to_excel(writer, sheet_name="Weighted_Exposures")
    for f in factor_cols:
        scores = universe[["Ticker", f]].copy()
        scores["Value"] = _rank_to_value(scores[f])
        dfc = weights.merge(scores[["Ticker","Value"]], on="Ticker", how="left")
        dfc["Port_Contrib"]  = dfc["W_port"]  * dfc["Value"]
        dfc["Bench_Contrib"] = dfc["W_bench"] * dfc["Value"]
        dfc["Active_Contrib"] = dfc["Port_Contrib"] - dfc["Bench_Contrib"]
        dfc.to_excel(writer, sheet_name=f"Contrib_{f[:22]}")
    for f in factor_cols:
        lab = compute_quintile_labels(universe, f)
        alpha.merge(lab, on="Ticker", how="left").to_excel(writer, sheet_name=f"A_{f[:23]}_q", index=False)
        core.merge(lab, on="Ticker", how="left").to_excel(writer, sheet_name=f"C_{f[:23]}_q", index=False)

st.download_button("Download exposures_results.xlsx", data=out_xlsx.getvalue(),
                   file_name="exposures_results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Using *Peer Rank* (lower is better). We invert to a 0–1 value so higher means better. Exposures are weight-averaged values; active = portfolio minus benchmark.")
