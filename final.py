import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Config (aligned with testing.py)
# -----------------------------
FACTOR_SUFFIXES = ["Peer Rank"]  # use *Peer Rank* columns
WEIGHT_COLUMN_GUESS = ["Abs Weight","Absolute Weight","Weight","Core Weight","Active Weight","AbsWeight","Wt","Weight (%)"]

# -----------------------------
# Helpers
# -----------------------------
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
    return st.selectbox("Kies kolom met Weight", df.columns, index=1 if len(df.columns)>1 else 0)

def _detect_factor_columns(df: pd.DataFrame) -> list:
    factors = []
    for c in df.columns:
        cname = str(c).strip()
        if any(cname.endswith(sfx) for sfx in FACTOR_SUFFIXES):
            factors.append(cname)
    if not factors:
        candidates = [c for c in df.columns if "rank" in str(c).lower()]
        factors = st.multiselect("Kies Factor-rank kolommen", candidates, default=candidates)
    return factors

def _read_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
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
        st.error(f"Kon geen tabbladen vinden met namen '{alpha_sheet}' en '{core_sheet}' (niet-hoofdlettergevoelig).")
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
        tick = out["Ticker"].astype(str)
        mask_equity = tick.str.contains("Equity", case=False, na=False)
        mask_client = ~tick.str.contains("client", case=False, na=False)
        out = out[mask_equity & mask_client & out[wcol].notna()]
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

# ---------- IMPORTANT: match testing.py ----------
def _rank_to_value(series: pd.Series) -> pd.Series:
    """
    Converteer rank (1..100, lager = beter) naar 0..1 waarde,
    robuust voor '#N/A Field Not Applicable' e.d.
    Ontbrekend -> 50.
    """
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    s = s.fillna(50.0)
    val = s / 100.0
    return val.clip(0, 1)
# -------------------------------------------------

def compute_weighted_exposures(df: pd.DataFrame, wcol: str, factor_cols: list) -> pd.Series:
    exp = {}
    for f in factor_cols:
        vals = _rank_to_value(df[f])
        exp[f] = (df[wcol] * vals).sum(skipna=True)
    return pd.Series(exp)

def compute_quintile_labels(universe: pd.DataFrame, factor: str) -> pd.DataFrame:
    u = universe.copy()
    vals = _rank_to_value(u[factor])
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

# -----------------------------
# Charts
# -----------------------------
def make_exposure_bar(port_exp: pd.Series, bench_exp: pd.Series) -> go.Figure:
    factors = list(port_exp.index)
    fig = go.Figure()
    fig.add_bar(name="Portfolio", x=factors, y=port_exp.values, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    fig.add_bar(name="Benchmark", x=factors, y=bench_exp.values, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    active = port_exp - bench_exp
    fig.add_scatter(name="Active", x=factors, y=active.values, mode="lines+markers", yaxis="y2", hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    fig.update_layout(
        barmode="group",
        title="Weighted Factor Exposures",
        yaxis_title="Exposure",
        yaxis2=dict(title="Active", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def make_quintile_chart(alpha_q: pd.Series, core_q: pd.Series, title: str) -> go.Figure:
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

def make_active_compare_bar_from_actives(ai: pd.Series, ao: pd.Series, title: str) -> go.Figure:
    """Groepsbalken wanneer actives al berekend zijn."""
    factors = list(ai.index)
    ao = ao.reindex(factors)
    fig = go.Figure()
    fig.add_bar(name="Active (Initial)",   x=factors, y=ai.values, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    fig.add_bar(name="Active (Optimized)", x=factors, y=ao.values, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>")
    ymax = float(np.nanmax(np.abs(np.concatenate([ai.values, ao.values])))) if len(factors) else 0.0
    pad  = 0.1 * ymax if ymax > 0 else 0.1
    fig.update_layout(
        barmode="group",
        title=title,
        yaxis_title="Active exposure",
        yaxis=dict(range=[-ymax - pad, ymax + pad]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="ASR - Factor Exposure Dashboard", layout="wide")

with st.sidebar:
    logo_path = Path("ASR_Nederland_logo.svg.png")
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.caption("Plaats 'ASR_Nederland_logo.svg.png' naast dit script om het logo te tonen.")

    st.markdown("### Data")
    uploaded = st.file_uploader("Excel-bestand", type=["xlsx","xlsm","xls"], key="exposures")
    alpha_name = st.text_input("Portfolio-tabblad", value="alpha")
    core_name  = st.text_input("Benchmark-tabblad", value="core")
    st.caption("We gebruiken kolommen met *Peer Rank* (lager is beter), omgezet naar 0–1 waarden (ontbrekend = 50).")
    

cols = st.columns([1,5])
with cols[1]:
    st.title("ASR Factor Exposure Dashboard")

if not uploaded:
    st.info("Upload je Excel-bestand om te beginnen.")
    st.stop()

alpha, a_wcol, core, c_wcol, factor_cols = load_portfolio_and_benchmark(uploaded, alpha_sheet=alpha_name, core_sheet=core_name)
if not factor_cols:
    st.error("Geen kolommen met *Peer Rank* gevonden.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Overzicht", "Detailanalyse", "Optimizer"])

# -----------------------------
# Overzicht
# -----------------------------
with tab1:
    alpha_exp  = compute_weighted_exposures(alpha, a_wcol, factor_cols)
    core_exp   = compute_weighted_exposures(core,  c_wcol, factor_cols)
    active_exp = alpha_exp - core_exp

    st.subheader("Gewogen Factor Exposures")
    summary_df = pd.DataFrame({"Portfolio": alpha_exp, "Benchmark": core_exp, "Active": active_exp}).sort_index()
    st.dataframe(summary_df.style.format("{:.3f}"))
    st.plotly_chart(
        make_exposure_bar(alpha_exp, core_exp),
        use_container_width=True,
        key="plt_overview_exposures"
    )

    st.subheader("Quintile gewichtsverdeling")
    universe = pd.concat([alpha[["Ticker"]+factor_cols], core[["Ticker"]+factor_cols]], axis=0, ignore_index=True).drop_duplicates("Ticker")
    factor_choice = st.selectbox("Kies een Factor voor de grafiek", factor_cols, index=0, key="factor_chart")
    alpha_q = compute_quintile_weights(universe, alpha[["Ticker", a_wcol, factor_choice]].rename(columns={a_wcol:"Weight"}), "Weight", factor_choice)
    core_q  = compute_quintile_weights(universe, core[["Ticker",  c_wcol, factor_choice]].rename(columns={c_wcol:"Weight"}), "Weight", factor_choice)
    if alpha_q.sum() > 0: alpha_q = alpha_q / alpha_q.sum()
    if core_q.sum()  > 0: core_q  = core_q  / core_q.sum()
    st.plotly_chart(
        make_quintile_chart(alpha_q, core_q, f"Quintile Weights by {factor_choice}"),
        use_container_width=True,
        key=f"plt_overview_quintile_{factor_choice}"
    )

# -----------------------------
# Detailanalyse
# -----------------------------
with tab2:
    universe = pd.concat([alpha[["Ticker"]+factor_cols], core[["Ticker"]+factor_cols]], axis=0, ignore_index=True).drop_duplicates("Ticker")
    weights = pd.merge(universe[["Ticker"]], alpha[["Ticker", a_wcol]].rename(columns={a_wcol:"W_port"}), on="Ticker", how="left")
    weights = pd.merge(weights, core[["Ticker", c_wcol]].rename(columns={c_wcol:"W_bench"}), on="Ticker", how="left").fillna(0.0)

    sub1, sub2 = st.tabs(["Factor-bijdragen", "Quintile-constituenten"])

    with sub1:
        f = st.selectbox("Factor", factor_cols, key="factor_contrib")
        scores = universe[["Ticker", f]].copy()
        scores["Value"] = _rank_to_value(scores[f])
        dfc = weights.merge(scores[["Ticker","Value"]], on="Ticker", how="left")
        dfc["Port_Contrib"]  = dfc["W_port"]  * dfc["Value"]
        dfc["Bench_Contrib"] = dfc["W_bench"] * dfc["Value"]
        dfc["Active_Contrib"] = dfc["Port_Contrib"] - dfc["Bench_Contrib"]
        dfc = dfc.sort_values("Active_Contrib", ascending=False)

        k = st.slider("Toon top N positief/negatief", min_value=5, max_value=50, value=20, step=5)
        top_pos = dfc.head(k)
        top_neg = dfc.tail(k).sort_values("Active_Contrib")

        st.markdown("**Top positieve Active-bijdragen**")
        st.dataframe(top_pos[["Ticker","W_port","W_bench","Port_Contrib","Bench_Contrib","Active_Contrib"]]
                    .style.format({"W_port":"{:.2%}","W_bench":"{:.2%}","Port_Contrib":"{:.3f}","Bench_Contrib":"{:.3f}","Active_Contrib":"{:.3f}"}),
                    use_container_width=True)

        st.markdown("**Top negatieve Active-bijdragen**")
        st.dataframe(top_neg[["Ticker","W_port","W_bench","Port_Contrib","Bench_Contrib","Active_Contrib"]]
                    .style.format({"W_port":"{:.2%}","W_bench":"{:.2%}","Port_Contrib":"{:.3f}","Bench_Contrib":"{:.3f}","Active_Contrib":"{:.3f}"}),
                    use_container_width=True)

    with sub2:
        st.markdown("### Quintile gewichtsverdeling")
        f_q_chart = st.selectbox("Factor (grafiek)", factor_cols, key="factor_quint_chart")
        alpha_qc = compute_quintile_weights(universe, alpha[["Ticker", a_wcol, f_q_chart]].rename(columns={a_wcol:"Weight"}), "Weight", f_q_chart)
        core_qc  = compute_quintile_weights(universe, core[["Ticker",  c_wcol, f_q_chart]].rename(columns={c_wcol:"Weight"}), "Weight", f_q_chart)
        if alpha_qc.sum() > 0: alpha_qc = alpha_qc / alpha_qc.sum()
        if core_qc.sum()  > 0: core_qc  = core_qc  / core_qc.sum()
        st.plotly_chart(
            make_quintile_chart(alpha_qc, core_qc, f"Quintile Weights by {f_q_chart}"),
            use_container_width=True,
            key=f"plt_drill_quintile_{f_q_chart}"
        )

        f_q = st.selectbox("Factor (constituenten)", factor_cols, key="factor_quint")
        labels = compute_quintile_labels(universe, f_q)
        alpha_qdf = alpha.merge(labels, on="Ticker", how="left")
        core_qdf  = core.merge(labels,  on="Ticker", how="left")

        q_choice = st.selectbox("Quintile", [f"Q{i}" for i in range(1,6)], index=4)
        def _prep(df, wcol):
            out = df[df["Quintile"]==q_choice].copy()
            out["Value"] = _rank_to_value(out[f_q])
            out["Contribution"] = out[wcol] * out["Value"]
            return out[["Ticker", wcol, "Value", "Contribution"]].rename(columns={wcol:"Weight"}).sort_values("Contribution", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Portfolio — {q_choice} van {f_q}**")
            st.dataframe(_prep(alpha_qdf, a_wcol).style.format({"Weight":"{:.2%}","Value":"{:.3f}","Contribution":"{:.3f}"}), use_container_width=True)
        with c2:
            st.markdown(f"**Benchmark — {q_choice} van {f_q}**")
            st.dataframe(_prep(core_qdf,  c_wcol).style.format({"Weight":"{:.2%}","Value":"{:.3f}","Contribution":"{:.3f}"}), use_container_width=True)

# -----------------------------
# Optimizer
# -----------------------------
with tab3:
    st.markdown("### Portfolio Optimizer")
    try:
        from scipy.optimize import minimize
        SCIPY_OK = True
    except Exception:
        SCIPY_OK = False
        st.error("SciPy is vereist voor de Optimizer: `pip install scipy`")

    def _find_gics_cols(df: pd.DataFrame):
        sect_col, ind_col = None, None
        for c in df.columns:
            cl = str(c).lower()
            if sect_col is None and "gics sector" in cl:
                sect_col = c
            if ind_col is None and "gics industry" in cl:
                ind_col = c
        return sect_col, ind_col
    
    def _find_fx_col(df: pd.DataFrame):
        fx_col = None
        for c in df.columns:
            cl = str(c).lower()
            if fx_col is None and ("currency" in cl or "crncy" in cl):
                fx_col = c
        return fx_col

    def group_exposure(df: pd.DataFrame, wcol: str, gcol: str) -> pd.Series:
        s = df[[gcol, wcol]].dropna().groupby(gcol, dropna=True)[wcol].sum()
        return s

    def make_group_bar(port: pd.Series, bench: pd.Series, title: str):
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
        # Universa uitlijnen
        bench = core[["Ticker", c_wcol]].rename(columns={c_wcol:"w_b"}).copy()
        port  = alpha[["Ticker", a_wcol]].rename(columns={a_wcol:"w"}).copy()
        dfw = pd.merge(port, bench, on="Ticker", how="left").fillna({"w_b":0.0})
        if dfw["w"].sum() > 0: dfw["w"] = dfw["w"] / dfw["w"].sum()
        if bench["w_b"].sum() > 0:
            bench["w_b"] = bench["w_b"] / bench["w_b"].sum()
            dfw = dfw.drop(columns=["w_b"]).merge(bench, on="Ticker", how="left").fillna({"w_b":0.0})

        tickers = dfw["Ticker"].tolist()

        frozen = st.multiselect("Bevries weight voor deze tickers (keep = initial portfolio weight)",
                        tickers, default=[])

        N = len(tickers)

        # Eenvoudig bewerkbaar Conviction-overzicht
        if "conv_table" not in st.session_state or set(st.session_state["conv_table"]["Ticker"]) != set(tickers):
            st.session_state["conv_table"] = pd.DataFrame({"Ticker": tickers, "Conviction": 100.0})
        st.markdown("**Conviction per naam (0–100)**")
        st.session_state["conv_table"] = st.data_editor(
            st.session_state["conv_table"], num_rows="fixed", hide_index=True, use_container_width=True
        )

        # Favor-tabel (Betas intypen) — geen neutralization
        if "favor_table" not in st.session_state or set(st.session_state["favor_table"]["Factor"]) != set(factor_cols):
            st.session_state["favor_table"] = pd.DataFrame({"Factor": factor_cols, "Beta": 0.0})
        st.markdown("**Maximaliseer Factors (stel Beta > 0 in)**")
        st.session_state["favor_table"] = st.data_editor(
            st.session_state["favor_table"], num_rows="fixed", hide_index=True, use_container_width=True
        )

        # Grenzen
        st.markdown("**Grenzen per naam**")
        c3, c4, c5 = st.columns(3)
        with c3:
            max_abs_w = st.number_input("Max absolute Weight", 0.0, 1.0, 0.10, 0.01, format="%.2f")
        with c4:
            max_active_w = st.number_input("Max |Active| per naam", 0.0, 1.0, 0.05, 0.01, format="%.2f")
        with c5:
            long_only = st.checkbox("Long-only", True)
        hard_lb = st.checkbox("Harde ondergrens: w ≥ w_bench", True)


        # Regularization (hardcoded)
        lam_eq = 100

        # Bands t.o.v. Benchmark
        sect_col, ind_col = _find_gics_cols(alpha)
        fx_col = _find_fx_col(alpha)
        st.markdown("**Bands vs Benchmark**")
        g1, g2, g3 = st.columns(3)
        with g1:
            use_sector = st.checkbox("Pas GICS Sector bands toe", value=False, disabled=(sect_col is None))
            band_sector = st.slider("Sector band (±%)", 0.0, 50.0, 10.0, 1.0, disabled=not use_sector) / 100.0
            if sect_col is None:
                st.caption("_GICS Sector-kolom niet gevonden in het Portfolio-tabblad._")
        with g2:
            use_industry = st.checkbox("Pas GICS Industry bands toe", value=False, disabled=(ind_col is None))
            band_ind = st.slider("Industry band (±%)", 0.0, 50.0, 10.0, 1.0, disabled=not use_industry) / 100.0
            if ind_col is None:
                st.caption("_GICS Industry-kolom niet gevonden in het Portfolio-tabblad._")
        with g3:
            use_fx = st.checkbox("FX currency bands", value=False, disabled=(fx_col is None))
            band_fx = st.slider("FX band (±%)", 0.0, 50.0, 10.0, 1.0, disabled=not use_fx) / 100.0
            if fx_col is None:
                st.caption("_Currency column not found in Portfolio tab._")


        run_opt = st.button("Run Optimizer", key="btn_run_opt")

        if run_opt:
            w_b = dfw["w_b"].to_numpy()
            w0  = dfw["w"].to_numpy()

            conv_map = st.session_state["conv_table"].set_index("Ticker")["Conviction"].clip(0,100) / 100.0
            s = pd.Series(tickers).map(conv_map).fillna(1.0).to_numpy()

            # Factor-waarden + Betas
            def xvec(colname):
                return _rank_to_value(alpha.set_index("Ticker")[colname].reindex(tickers)).to_numpy()
            favor_dict = {row["Factor"]: float(row["Beta"]) for _, row in st.session_state["favor_table"].iterrows() if float(row["Beta"]) > 0}

            def objective(w):
                a = w - w_b
                val = float(s @ a)
                for f, beta in favor_dict.items():
                    val += beta * float(xvec(f) @ a)
                # Equal-active penalty (concentratiecontrole)
                if lam_eq > 0:
                    soft_abs = np.sqrt(a*a + 1e-9)
                    dev = soft_abs - soft_abs.mean()
                    pen_eq = float(np.mean(dev*dev))
                    val -= lam_eq * pen_eq
                return -val  # minimaliseren

            cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]

            # Benchmark-groepdelen (voor bands)
            def bench_group_share(core_df, group_col: str) -> dict:
                if not group_col:
                    return {}
                g = (core_df[[group_col, c_wcol]]
                     .dropna(subset=[group_col])
                     .groupby(group_col, dropna=True)[c_wcol].sum())
                return g.to_dict()

            a_sect, a_ind = _find_gics_cols(alpha)
            c_sect, c_ind = _find_gics_cols(core)

            core_full = core.copy()
            if core_full[c_wcol].sum() != 0:
                core_full[c_wcol] = core_full[c_wcol] / core_full[c_wcol].sum()

            bench_sector_share   = bench_group_share(core_full, c_sect) if c_sect else {}
            bench_industry_share = bench_group_share(core_full, c_ind)  if c_ind  else {}
            c_fx = _find_fx_col(core)
            bench_fx_share       = bench_group_share(core_full, c_fx)   if c_fx   else {}


            # Sector-bands
            if use_sector and a_sect and c_sect:
                gser = alpha.set_index("Ticker")[a_sect].reindex(tickers).fillna("NA")
                for g in sorted(gser.unique()):
                    m = (gser == g).to_numpy().astype(float)
                    wbg = float(bench_sector_share.get(g, 0.0))
                    lo = wbg - band_sector
                    up = wbg + band_sector
                    cons.append({"type":"ineq", "fun": lambda w, m=m, lo=lo: (m @ w) - lo})
                    cons.append({"type":"ineq", "fun": lambda w, m=m, up=up: up - (m @ w)})

            # Industry-bands
            if use_industry and a_ind and c_ind:
                gser = alpha.set_index("Ticker")[a_ind].reindex(tickers).fillna("NA")
                for g in sorted(gser.unique()):
                    m = (gser == g).to_numpy().astype(float)
                    wbg = float(bench_industry_share.get(g, 0.0))
                    lo = wbg - band_ind
                    up = wbg + band_ind
                    cons.append({"type":"ineq", "fun": lambda w, m=m, lo=lo: (m @ w) - lo})
                    cons.append({"type":"ineq", "fun": lambda w, m=m, up=up: up - (m @ w)})

            # FX currency bands
            if use_fx and fx_col and c_fx:
                fx_ser = alpha.set_index("Ticker")[fx_col].reindex(tickers).fillna("NA")
                for cur in sorted(fx_ser.unique()):
                    m = (fx_ser == cur).to_numpy().astype(float)
                    wbg = float(bench_fx_share.get(cur, 0.0))
                    lo = wbg - band_fx
                    up = wbg + band_fx
                    cons.append({"type":"ineq", "fun": lambda w, m=m, lo=lo: (m @ w) - lo})
                    cons.append({"type":"ineq", "fun": lambda w, m=m, up=up:  up - (m @ w)})


            if hard_lb:
                bad = [t for t, w_i, wb_i in zip(tickers, w0, w_b) if (t in frozen and w_i < wb_i - 1e-12)]
                if bad:
                    st.warning(
                        "Harde ondergrends (w ≥ w_bench) conflict met frozen underweights voor: "
                        + ", ".join(bad)
                        + ". Ignoring hard_lb voor frozen names."
                    )


            bounds = []
            for i in range(N):
                if tickers[i] in frozen:
                    # frozen: allow negative active; no hard_lb, no per-name active caps
                    bounds.append((float(w0[i]), float(w0[i])))
                    continue

                lb = 0.0 if long_only else -max_abs_w
                if hard_lb:
                    lb = max(lb, float(w_b[i]))  # only for non-frozen names
                ub = max_abs_w
                bounds.append((lb, ub))

                # per-name active band only for non-frozen:
                cons.append({"type":"ineq", "fun": lambda w, i=i:  max_active_w -  (w[i] - w_b[i])})
                cons.append({"type":"ineq", "fun": lambda w, i=i:  max_active_w +  (w[i] - w_b[i])})


            # Build arrays of lb/ub from bounds
            lb_arr = np.array([b[0] for b in bounds], dtype=float)
            ub_arr = np.array([b[1] for b in bounds], dtype=float)

            # Start from w0, clipped to bounds
            x0 = np.clip(w0, lb_arr, ub_arr)

            # Ensure sum-to-1 while respecting bounds (adjust only non-frozen vars)
            free = ~(np.isclose(lb_arr, ub_arr))   # True if not frozen
            resid = 1.0 - x0.sum()

            if abs(resid) > 1e-12 and free.any():
                if resid > 0:
                    # add weight to free vars up to their remaining headroom
                    room_up = (ub_arr - x0) * free
                    total_room = room_up.sum()
                    if total_room < resid - 1e-12:
                        st.error("Infeasible: not enough upper-bound room to reach sum=1 with current freezes/bands.")
                    else:
                        x0 += np.where(free, resid * (room_up / total_room), 0.0)
                else:
                    # remove weight from free vars down to their lower bounds
                    room_down = (x0 - lb_arr) * free
                    total_room = room_down.sum()
                    if total_room < (-resid) - 1e-12:
                        st.error("Infeasible: not enough lower-bound room to achieve sum=1 with current freezes/bands.")
                    else:
                        x0 -= np.where(free, (-resid) * (room_down / total_room), 0.0)

                    # After you have tickers, w0, w_b, and 'frozen'

            res = minimize(
                objective,
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"maxiter":1000, "ftol":1e-9},
            )
            if not res.success:
                st.error(f"Optimizer-status: {res.message}")

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

            st.subheader("Optimized Weights")
            st.dataframe(out.style.format({"w_bench":"{:.2%}","w_cur":"{:.2%}","w_opt":"{:.2%}","active_opt":"{:.2%}"}), use_container_width=True)

            # ---------- Active-grafiek ----------
            exp_init = compute_weighted_exposures(alpha, a_wcol, factor_cols)
            exp_bench_full = compute_weighted_exposures(core, c_wcol, factor_cols)

            # opt_df inclusief GICS-kolommen (voor latere grafieken)
            opt_df = alpha.set_index("Ticker").reindex(tickers).reset_index()[["Ticker"] + factor_cols].copy()

            # Add GICS and FX columns if present
            if sect_col and sect_col in alpha.columns:
                opt_df[sect_col] = alpha.set_index("Ticker")[sect_col].reindex(tickers).values
            if ind_col and ind_col in alpha.columns:
                opt_df[ind_col] = alpha.set_index("Ticker")[ind_col].reindex(tickers).values
            fx_col = _find_fx_col(alpha)
            if fx_col and fx_col in alpha.columns:
                opt_df[fx_col] = alpha.set_index("Ticker")[fx_col].reindex(tickers).values

            # Add optimized weights
            opt_df["w_opt"] = w_opt
            tot = float(opt_df["w_opt"].sum())
            if tot > 0:
                opt_df["w_opt"] = opt_df["w_opt"] / tot

            exp_opt = compute_weighted_exposures(opt_df.rename(columns={"w_opt":"w"}), "w", factor_cols)

            ai_raw = exp_init - exp_bench_full
            ao_raw = exp_opt  - exp_bench_full

            # Opties voor visualisatie
            st.markdown("**Opties voor visualisatie van Active exposure**")
            use_z = st.checkbox("Standaardiseer Active via Factor-dispersie (z-units)", value=True,
                                help="Deel Active door de cross-sectionele std.dev. over het gecombineerde Portfolio+Benchmark-universum.")

            ai, ao = ai_raw.copy(), ao_raw.copy()
            if use_z:
                all_vals = pd.concat([alpha[["Ticker"]+factor_cols], core[["Ticker"]+factor_cols]], ignore_index=True).drop_duplicates("Ticker")
                sds = {}
                for f in factor_cols:
                    v = _rank_to_value(all_vals[f]).astype(float)
                    sd = float(v.std(ddof=0))
                    if not np.isfinite(sd) or sd < 1e-9:
                        sd = 1.0
                    sds[f] = sd
                sds = pd.Series(sds).reindex(ai.index).fillna(1.0)
                ai = ai / sds
                ao = ao / sds

            st.subheader("Active exposures — Initieel vs Optimized (zelfde grafiek)")
            st.plotly_chart(
                make_active_compare_bar_from_actives(ai, ao, "Active (Initial) vs Active (Optimized)"),
                use_container_width=True,
                key="plt_optimizer_active_compare_fixed"
            )
            # ---------- Einde Active-grafiek ----------

            # ---------- GICS exposures ----------
            st.subheader("GICS exposures vs Benchmark (Optimized)")
            def _find_gics_cols_in(df_):
                sect_col_, ind_col_ = None, None
                for c in df_.columns:
                    cl = str(c).lower()
                    if sect_col_ is None and "gics sector" in cl:
                        sect_col_ = c
                    if ind_col_ is None and "gics industry" in cl:
                        ind_col_ = c
                return sect_col_, ind_col_


            a_sect2, a_ind2 = _find_gics_cols_in(opt_df)
            c_sect, c_ind   = _find_gics_cols(core)

            core_norm = core.copy()
            if core_norm[c_wcol].sum() != 0:
                core_norm[c_wcol] = core_norm[c_wcol] / core_norm[c_wcol].sum()

            if a_sect2 and c_sect and a_sect2 in opt_df.columns and c_sect in core_norm.columns:
                p_sec = group_exposure(opt_df, "w_opt", a_sect2)
                b_sec = group_exposure(core_norm, c_wcol, c_sect)
                st.plotly_chart(
                    make_group_bar(p_sec, b_sec, "GICS Sector exposure vs Benchmark (Optimized)"),
                    use_container_width=True,
                    key="plt_sector_opt"
                )
            else:
                st.info("GICS Sector-kolommen ontbreken in Portfolio/opt of Benchmark — sectorgrafiek overgeslagen.")

            if a_ind2 and c_ind and a_ind2 in opt_df.columns and c_ind in core_norm.columns:
                p_ind = group_exposure(opt_df, "w_opt", a_ind2)
                b_ind = group_exposure(core_norm, c_wcol, c_ind)
                st.plotly_chart(
                    make_group_bar(p_ind, b_ind, "GICS Industry exposure vs Benchmark (Optimized)"),
                    use_container_width=True,
                    key="plt_ind_opt"
                )

            # FX exposures vs Benchmark (Optimized)
            fx_col_opt = _find_fx_col(opt_df)
            c_fx = _find_fx_col(core_norm)
            if fx_col_opt and c_fx and fx_col_opt in opt_df.columns and c_fx in core_norm.columns:
                p_fx = group_exposure(opt_df, "w_opt", fx_col_opt)
                b_fx = group_exposure(core_norm, c_wcol, c_fx)
                st.plotly_chart(
                    make_group_bar(p_fx, b_fx, "FX currency exposure vs Benchmark (Optimized)"),
                    use_container_width=True,
                    key="plt_fx_opt"
                )


            # ---------- Suggesties ----------
            st.subheader("Suggesties uit de Benchmark die de doel-Exposure helpen")
            if len(favor_dict) == 0:
                st.caption("Geen favor Betas > 0 ingesteld — voeg hierboven Betas toe voor suggesties.")
            else:
                w_cur_map = dict(zip(tickers, w0))
                core_slim = core[["Ticker", c_wcol] + factor_cols].copy()
                core_slim["w_cur"] = core_slim["Ticker"].map(w_cur_map).fillna(0.0)
                core_slim["w_bench"] = core_slim[c_wcol]

                lift = np.zeros(len(core_slim))
                for f, beta in favor_dict.items():
                    lift += beta * _rank_to_value(core_slim[f]).to_numpy()
                core_slim["factor_lift"] = lift
                core_slim["room_to_add"] = (core_slim["w_bench"] - core_slim["w_cur"]).clip(lower=0.0)
                core_slim["priority"] = core_slim["factor_lift"] * core_slim["room_to_add"]

                sugg = (core_slim[core_slim["room_to_add"] > 0]
                        .sort_values("priority", ascending=False)
                        .head(20)[["Ticker","w_cur","w_bench","room_to_add","factor_lift","priority"]])

                st.dataframe(
                    sugg.style.format({"w_cur":"{:.2%}","w_bench":"{:.2%}","room_to_add":"{:.2%}","factor_lift":"{:.3f}","priority":"{:.3f}"}),
                    use_container_width=True
                )
                st.caption("Eenvoudige heuristiek: prioriteer namen met hoge factor lift (op basis van Betas) waar je onderwogen bent t.o.v. de Benchmark. Negeert bands en naamcaps.")
        else:
            st.info("Stel Convictions, Betas, bands en grenzen in en klik daarna op **Run Optimizer**.")

# -----------------------------
# Download
# -----------------------------
st.subheader("Resultaten downloaden")
out_xlsx = io.BytesIO()
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
    aexp = compute_weighted_exposures(alpha, a_wcol, factor_cols)
    cexp = compute_weighted_exposures(core, c_wcol, factor_cols)
    pd.DataFrame({"Portfolio":aexp, "Benchmark":cexp, "Active":aexp-cexp}).to_excel(writer, sheet_name="Weighted_Exposures")
    universe = pd.concat([alpha[["Ticker"]+factor_cols], core[["Ticker"]+factor_cols]], axis=0, ignore_index=True).drop_duplicates("Ticker")
    weights = pd.merge(universe[["Ticker"]], alpha[["Ticker", a_wcol]].rename(columns={a_wcol:"W_port"}), on="Ticker", how="left")
    weights = pd.merge(weights, core[["Ticker", c_wcol]].rename(columns={c_wcol:"W_bench"}), on="Ticker", how="left").fillna(0.0)
    for f in factor_cols:
        scores = universe[["Ticker", f]].copy()
        scores["Value"] = _rank_to_value(scores[f])
        dfc = weights.merge(scores[["Ticker","Value"]], on="Ticker", how="left")
        dfc["Port_Contrib"]  = dfc["W_port"]  * dfc["Value"]
        dfc["Bench_Contrib"] = dfc["W_bench"] * dfc["Value"]
        dfc["Active_Contrib"] = dfc["Port_Contrib"] - dfc["Bench_Contrib"]
        dfc.to_excel(writer, sheet_name=f"Contrib_{f[:22]}")

st.download_button("Download exposures_results.xlsx", data=out_xlsx.getvalue(),
                   file_name="exposures_results.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
