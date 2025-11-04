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
    s.fillna(50, inplace=True)  # median rank if missing
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
    vals = _rank_to_value(u[factor])  # higher = better (keeping your current convention)
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
        title="Weighted Factor Exposures (Peer Rank scaled 0–1)",
        yaxis_title="Exposure",
        yaxis2=dict(title="Active", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def make_active_only_bar(port_exp: pd.Series, bench_exp: pd.Series, title: str):
    active = (port_exp - bench_exp)
    factors = list(active.index)
    fig = go.Figure()
    fig.add_bar(name="Active", x=factors, y=active.values, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    ymax = float(np.nanmax(np.abs(active.values))) if len(active) else 0.0
    pad  = 0.1 * ymax if ymax > 0 else 0.1
    fig.update_layout(
        title=title, yaxis_title="Active exposure", yaxis=dict(range=[-ymax - pad, ymax + pad]),
        showlegend=False
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

# -----------------------------
# Overview + Drilldowns (unchanged)
# -----------------------------

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
tab_factor, tab_quint, tab_opt = st.tabs(["Factor contributors", "Quintile constituents", "Optimizer"])

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

# -----------------------------
# Optimizer tab (runs only on button)
# -----------------------------
with tab_opt:
    st.markdown("### Portfolio Optimizer")
    try:
        from scipy.optimize import minimize
        SCIPY_OK = True
    except Exception:
        SCIPY_OK = False
        st.error("SciPy is required for the optimizer: `pip install scipy`")

    def _find_gics_cols(df: pd.DataFrame):
        sect_col, ind_col = None, None
        for c in df.columns:
            cl = str(c).lower()
            if sect_col is None and "gics sector" in cl:
                sect_col = c
            if ind_col is None and "gics industry" in cl:
                ind_col = c
        return sect_col, ind_col
    
    def group_exposure(df: pd.DataFrame, wcol: str, gcol: str) -> pd.Series:
        """Weight share by sector/industry; assumes weights sum to ~1."""
        s = df[[gcol, wcol]].dropna().groupby(gcol, dropna=True)[wcol].sum()
        # keep as plain Series, sorted by index by default
        return s
    
    def make_group_bar(port: pd.Series, bench: pd.Series, title: str):
        # align categories
        cats = sorted(set(port.index) | set(bench.index))
        p = port.reindex(cats, fill_value=0.0)
        b = bench.reindex(cats, fill_value=0.0)
        active = p - b

        fig = go.Figure()
        fig.add_bar(name="Portfolio", x=cats, y=p.values, hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
        fig.add_bar(name="Benchmark", x=cats, y=b.values, hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
        fig.add_scatter(name="Active", x=cats, y=active.values, mode="lines+markers",
                        yaxis="y2", hovertemplate="%{x}<br>%{y:.2%}<extra></extra>")
        fig.update_layout(
            barmode="group",
            title=title,
            yaxis_title="Weight share",
            yaxis_tickformat=".0%",
            yaxis2=dict(title="Active", tickformat=".0%", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(tickangle=-30)
        )
        return fig

    if SCIPY_OK:
        # Align tickers/weights to benchmark
        bench = core[["Ticker", c_wcol]].rename(columns={c_wcol:"w_b"}).copy()
        port  = alpha[["Ticker", a_wcol]].rename(columns={a_wcol:"w"}).copy()
        dfw = pd.merge(port, bench, on="Ticker", how="left").fillna({"w_b":0.0})
        if dfw["w"].sum() > 0: dfw["w"] = dfw["w"] / dfw["w"].sum()
        if bench["w_b"].sum() > 0:
            bench["w_b"] = bench["w_b"] / bench["w_b"].sum()
            dfw = dfw.drop(columns=["w_b"]).merge(bench, on="Ticker", how="left").fillna({"w_b":0.0})

        tickers = dfw["Ticker"].tolist()
        N = len(tickers)

        # Convictions editor
        if "conv_table" not in st.session_state or set(st.session_state["conv_table"]["Ticker"]) != set(tickers):
            st.session_state["conv_table"] = pd.DataFrame({"Ticker": tickers, "Conviction": 100.0})
        st.markdown("**Conviction per name (0–100)**")
        st.session_state["conv_table"] = st.data_editor(
            st.session_state["conv_table"], num_rows="fixed", hide_index=True, use_container_width=True
        )

        # Factor weights editors
        if "favor_table" not in st.session_state or set(st.session_state["favor_table"]["Factor"]) != set(factor_cols):
            st.session_state["favor_table"] = pd.DataFrame({"Factor": factor_cols, "Beta": 0.0})
        if "neutral_table" not in st.session_state or set(st.session_state["neutral_table"]["Factor"]) != set(factor_cols):
            st.session_state["neutral_table"] = pd.DataFrame({"Factor": factor_cols, "Lambda": 0.0})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Maximize factors (set Beta > 0)**")
            st.session_state["favor_table"] = st.data_editor(
                st.session_state["favor_table"], num_rows="fixed", hide_index=True, use_container_width=True
            )
        with c2:
            st.markdown("**Neutralize factors (set Lambda > 0)**")
            st.session_state["neutral_table"] = st.data_editor(
                st.session_state["neutral_table"], num_rows="fixed", hide_index=True, use_container_width=True
            )

        # Bounds
        st.markdown("**Per-name bounds**")
        c3, c4, c5 = st.columns(3)
        with c3:
            max_abs_w = st.number_input("Max absolute weight", 0.0, 1.0, 0.10, 0.01, format="%.2f")
        with c4:
            max_active_w = st.number_input("Max |active| per name", 0.0, 1.0, 0.05, 0.01, format="%.2f")
        with c5:
            long_only = st.checkbox("Long-only", True)
        hard_lb = st.checkbox("Hard lower bound: w ≥ w_bench", True)

        # Regularization
        lam_eq = 100

        # Group bands (sector + industry independent)
        sect_col, ind_col = _find_gics_cols(alpha)
        st.markdown("**Bands vs benchmark**")
        g1, g2 = st.columns(2)
        with g1:
            use_sector = st.checkbox("Apply GICS Sector bands", value=False, disabled=(sect_col is None))
            band_sector = st.slider("Sector band (±%)", 0.0, 50.0, 10.0, 1.0, disabled=not use_sector) / 100.0
            if sect_col is None:
                st.caption("_GICS Sector column not found in alpha sheet._")
        with g2:
            use_industry = st.checkbox("Apply GICS Industry bands", value=False, disabled=(ind_col is None))
            band_ind = st.slider("Industry band (±%)", 0.0, 50.0, 10.0, 1.0, disabled=not use_industry) / 100.0
            if ind_col is None:
                st.caption("_GICS Industry column not found in alpha sheet._")

        run_opt = st.button("Run optimizer")

        if run_opt:
            w_b = dfw["w_b"].to_numpy()
            w0  = dfw["w"].to_numpy()

            conv_map = st.session_state["conv_table"].set_index("Ticker")["Conviction"].clip(0,100) / 100.0
            s = pd.Series(tickers).map(conv_map).fillna(1.0).to_numpy()

            # Factor values aligned
            def xvec(colname):
                return _rank_to_value(alpha.set_index("Ticker")[colname].reindex(tickers)).to_numpy()

            favor_dict   = {row["Factor"]: float(row["Beta"])   for _, row in st.session_state["favor_table"].iterrows()   if float(row["Beta"])   > 0}
            neutral_dict = {row["Factor"]: float(row["Lambda"]) for _, row in st.session_state["neutral_table"].iterrows() if float(row["Lambda"]) > 0}

            def objective(w):
                a = w - w_b
                val = float(s @ a)

                # Factor tilts / neutralization
                for f, beta in favor_dict.items():
                    val += beta * float(xvec(f) @ a)
                for f, lam in neutral_dict.items():
                    xa = float(xvec(f) @ a)
                    val -= lam * (xa * xa)

                # Equal-active penalty (concentration control)
                if lam_eq > 0:
                    soft_abs = np.sqrt(a*a + 1e-9)          # smooth |a|
                    dev = soft_abs - soft_abs.mean()
                    pen_eq = float(np.mean(dev*dev))        # variance of |a|
                    val -= lam_eq * pen_eq

                return -val  # minimize

            cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]

            # detect GICS columns on alpha and core separately (names can differ)
            a_sect, a_ind = _find_gics_cols(alpha)
            c_sect, c_ind = _find_gics_cols(core)

            # Normalize full benchmark once
            core_full = core.copy()
            if core_full[c_wcol].sum() != 0:
                core_full[c_wcol] = core_full[c_wcol] / core_full[c_wcol].sum()

            # Helper: full-benchmark group share dict
            def bench_group_share(group_col: str) -> dict:
                if not group_col:
                    return {}
                g = (core_full[[group_col, c_wcol]]
                    .dropna(subset=[group_col])
                    .groupby(group_col, dropna=True)[c_wcol].sum())
                return g.to_dict()
            
            # Get benchmark group shares (FULL core)
            bench_sector_share   = bench_group_share(c_sect) if use_sector and a_sect and c_sect else {}
            bench_industry_share = bench_group_share(c_ind)  if use_industry and a_ind and c_ind else {}

            # --- Sector ACTIVE bands ---
            if use_sector and a_sect and c_sect:
                # Alpha group labels aligned to decision vector
                gser = alpha.set_index("Ticker")[a_sect].reindex(tickers).fillna("NA")
                for g in sorted(gser.unique()):
                    m = (gser == g).to_numpy().astype(float)   # indicator over decision vars
                    wbg = float(bench_sector_share.get(g, 0.0))  # FULL-benchmark share for this sector
                    lo = wbg - band_sector           # percentage points active band
                    up = wbg + band_sector
                    # Inequalities: lo ≤ m@w ≤ up
                    cons.append({"type":"ineq", "fun": lambda w, m=m, lo=lo: (m @ w) - lo})
                    cons.append({"type":"ineq", "fun": lambda w, m=m, up=up: up - (m @ w)})

            # --- Industry ACTIVE bands ---
            if use_industry and a_ind and c_ind:
                gser = alpha.set_index("Ticker")[a_ind].reindex(tickers).fillna("NA")
                for g in sorted(gser.unique()):
                    m = (gser == g).to_numpy().astype(float)
                    wbg = float(bench_industry_share.get(g, 0.0))
                    lo = wbg - band_ind
                    up = wbg + band_ind
                    cons.append({"type":"ineq", "fun": lambda w, m=m, lo=lo: (m @ w) - lo})
                    cons.append({"type":"ineq", "fun": lambda w, m=m, up=up: up - (m @ w)})

            # Bounds + active caps
            bounds = []
            for i in range(N):
                lb = 0.0 if long_only else -max_abs_w
                if hard_lb:
                    lb = max(lb, float(w_b[i]))
                ub = max_abs_w
                bounds.append((lb, ub))
                cons.append({"type":"ineq", "fun": lambda w, i=i:  max_active_w -  (w[i] - w_b[i])})
                cons.append({"type":"ineq", "fun": lambda w, i=i:  max_active_w +  (w[i] - w_b[i])})

            res = minimize(
                objective,
                x0=np.clip(w0, 0, max_abs_w),
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter":1000, "ftol":1e-9},
            )
            if not res.success:
                st.error(f"Optimizer status: {res.message}")

            w_opt = res.x
            a_opt = w_opt - w_b

            out = pd.DataFrame({
                "Ticker": tickers,
                "w_bench": w_b,
                "w_cur": w0,
                "w_opt": w_opt,
                "active_opt": a_opt
            })
            if sect_col is not None:
                out[sect_col] = alpha.set_index("Ticker")[sect_col].reindex(tickers).values
            if ind_col is not None:
                out[ind_col]  = alpha.set_index("Ticker")[ind_col].reindex(tickers).values

            st.subheader("Optimized weights")
            st.dataframe(out.style.format({"w_bench":"{:.2%}","w_cur":"{:.2%}","w_opt":"{:.2%}","active_opt":"{:.2%}"}), use_container_width=True)

            # exposures pre/post
            def wexp(w):
                tmp = alpha.set_index("Ticker").reindex(tickers).copy()
                tmp["w"] = w
                return compute_weighted_exposures(tmp.reset_index(), "w", factor_cols)
            st.subheader("Exposures: optimized vs benchmark")
            st.plotly_chart(make_active_only_bar(wexp(w_opt), compute_weighted_exposures(core,  c_wcol, factor_cols), title='Optimized vs Benchmark Exposures'), use_container_width=True)
            st.subheader("Exposures: initial vs benchmark")
            st.plotly_chart(make_active_only_bar(compute_weighted_exposures(alpha, a_wcol, factor_cols), compute_weighted_exposures(core,  c_wcol, factor_cols)), use_container_width=True, key='second')

            st.subheader("GICS exposures vs benchmark")

            # detect GICS columns on alpha and core separately (names can differ)
            a_sect, a_ind = _find_gics_cols(alpha)
            c_sect, c_ind = _find_gics_cols(core)

            # normalize just to be safe
            alpha_norm = alpha.copy()
            core_norm  = core.copy()
            if alpha_norm[a_wcol].sum() != 0:
                alpha_norm[a_wcol] = alpha_norm[a_wcol] / alpha_norm[a_wcol].sum()
            if core_norm[c_wcol].sum() != 0:
                core_norm[c_wcol]  = core_norm[c_wcol]  / core_norm[c_wcol].sum()

            # --- GICS exposures vs benchmark (using OPTIMIZED weights) ---
            st.subheader("GICS exposures vs benchmark (optimized)")

            # detect GICS columns on alpha and core separately (names can differ)
            a_sect, a_ind = _find_gics_cols(alpha)
            c_sect, c_ind = _find_gics_cols(core)

            # Build a dataframe of OPTIMIZED weights on the alpha universe
            opt_df = alpha.set_index("Ticker").reindex(tickers).reset_index()
            opt_df["w_opt"] = w_opt
            # normalize just in case of tiny drift
            tot = float(opt_df["w_opt"].sum())
            if tot > 0:
                opt_df["w_opt"] = opt_df["w_opt"] / tot

            # FULL benchmark normalized (actual benchmark, not restricted to alpha)
            core_norm = core.copy()
            if core_norm[c_wcol].sum() != 0:
                core_norm[c_wcol] = core_norm[c_wcol] / core_norm[c_wcol].sum()

            # Sector chart (optimized vs full benchmark)
            if a_sect and c_sect:
                p_sec = group_exposure(opt_df, "w_opt", a_sect)           # optimized portfolio sectors
                b_sec = group_exposure(core_norm, c_wcol, c_sect)         # full benchmark sectors
                st.plotly_chart(
                    make_group_bar(p_sec, b_sec, "GICS Sector exposure vs benchmark (optimized)"),
                    use_container_width=True
                )
            else:
                st.info("GICS Sector column not found in both alpha and core sheets — skipping sector chart.")

            # Industry chart (optimized vs full benchmark)
            if a_ind and c_ind:
                p_ind = group_exposure(opt_df, "w_opt", a_ind)
                b_ind = group_exposure(core_norm, c_wcol, c_ind)
                st.plotly_chart(
                    make_group_bar(p_ind, b_ind, "GICS Industry exposure vs benchmark (optimized)"),
                    use_container_width=True
                )
        else:
            st.info("Set convictions, factor weights, bands and bounds, then click **Run optimizer**.")

