from __future__ import annotations

import io
import textwrap
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    breaks_cusumolsresid,
    het_arch,
    het_breuschpagan,
    het_white,
    linear_rainbow,
    linear_reset,
)
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

st.set_page_config(page_title="Econometrics Studio — Diagnostics", page_icon="🧪", layout="wide")

TESTS = [
    "Durbin-Watson", "Jarque-Bera", "D'Agostino-Pearson", "Breusch-Pagan",
    "White", "Breusch-Godfrey", "Ljung-Box", "ARCH LM", "Ramsey RESET",
    "Rainbow linearity", "CUSUM stability", "Condition number",
    "Variance inflation factors", "Influence observations",
]


def init_state() -> None:
    defaults = {
        "df": None, "source_filename": None, "history": [], "code_blocks": [],
        "results": {}, "settings_log": [], "slot_outputs": {}, "dark_mode": False,
        "diagnostic_page_state": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme() -> None:
    if not st.session_state.dark_mode:
        plt.rcdefaults()
        return
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#111827",
        "savefig.facecolor": "#0e1117", "text.color": "#f3f4f6",
        "axes.labelcolor": "#f3f4f6", "axes.edgecolor": "#6b7280",
        "xtick.color": "#d1d5db", "ytick.color": "#d1d5db",
    })
    st.markdown("""
    <style>
    :root{color-scheme:dark;--bg:#0e1117;--panel:#161b22;--surface:#111827;
    --surface2:#1f2937;--border:#374151;--text:#f3f4f6;--muted:#cbd5e1}
    html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
    [data-testid="stMainBlockContainer"],[data-testid="stHeader"]{background:var(--bg)!important;color:var(--text)!important}
    [data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:var(--panel)!important}
    .stApp,.stApp p,.stApp label,.stApp span,.stApp li,.stApp h1,.stApp h2,.stApp h3,
    [data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"],[data-testid="stSidebar"] *{color:var(--text)!important}
    [data-testid="stCaptionContainer"],small{color:var(--muted)!important}
    div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,input,textarea{
    background:var(--surface2)!important;color:var(--text)!important;border-color:#4b5563!important}
    div[data-baseweb="popover"],div[data-baseweb="menu"],ul[role="listbox"],[role="option"]{
    background:var(--surface2)!important;color:var(--text)!important}
    .stButton>button{border-color:#4b5563!important;color:var(--text)!important}
    .stButton>button[kind="primary"]{background:#2563eb!important}
    .eco-wrap{width:100%;overflow:auto;max-height:520px;border:1px solid var(--border);
    border-radius:.6rem;background:var(--surface);margin:.25rem 0 1rem}
    table.eco{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;
    color:var(--text)!important;background:var(--surface)!important;font-size:.88rem}
    table.eco thead th{position:sticky;top:0;background:#1f2937!important;color:#f8fafc!important}
    table.eco tbody th{position:sticky;left:0;background:#172033!important;color:#e2e8f0!important}
    table.eco th,table.eco td{padding:.62rem .72rem;border-right:1px solid #263244;
    border-bottom:1px solid #263244;text-align:right;white-space:nowrap}
    </style>""", unsafe_allow_html=True)


def show_frame(value: Any) -> None:
    if not st.session_state.dark_mode:
        st.dataframe(value, use_container_width=True)
        return
    frame = value.to_frame() if isinstance(value, pd.Series) else pd.DataFrame(value).copy()
    frame = frame.map(lambda x: "" if pd.isna(x) else (f"{float(x):,.6f}" if isinstance(x, (float, np.floating)) else str(x)))
    st.markdown(f'<div class="eco-wrap">{frame.to_html(classes="eco", border=0, escape=True)}</div>', unsafe_allow_html=True)


def clean_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def unique_name() -> str:
    base = f"Diagnostics_{datetime.now().strftime('%H%M%S')}"
    name, index = base, 2
    while name in st.session_state.results:
        name, index = f"{base}_{index}", index + 1
    return name


def latest_name() -> str | None:
    stack = st.session_state.slot_outputs.get("diagnostics", [])
    if isinstance(stack, str):
        stack = [stack]
    return stack[-1] if stack else None


def clear_latest() -> str | None:
    name = latest_name()
    if not name:
        return None
    st.session_state.code_blocks = [b for b in st.session_state.code_blocks if b.get("name") != name]
    st.session_state.results = {k: v for k, v in st.session_state.results.items() if k != name and not k.startswith(name + "_")}
    st.session_state.settings_log = [x for x in st.session_state.settings_log if x.get("name") != name]
    st.session_state.history = [x for x in st.session_state.history if x.get("action") != f"Completed {name}"]
    stack = st.session_state.slot_outputs.get("diagnostics", [])
    if isinstance(stack, str):
        stack = [stack]
    remaining = [x for x in stack if x != name]
    if remaining:
        st.session_state.slot_outputs["diagnostics"] = remaining
    else:
        st.session_state.slot_outputs.pop("diagnostics", None)
    if st.session_state.diagnostic_page_state and st.session_state.diagnostic_page_state.get("name") == name:
        st.session_state.diagnostic_page_state = None
    return name


def coefficient_table(results: Any) -> pd.DataFrame:
    conf = results.conf_int()
    conf.columns = ["CI Lower", "CI Upper"]
    return pd.DataFrame({
        "Coefficient": results.params, "Std. Error": results.bse,
        "Statistic": results.tvalues, "P-value": results.pvalues,
        "CI Lower": conf["CI Lower"], "CI Upper": conf["CI Upper"],
    })


def run_suite(results: Any, selected: list[str], lag: int, alpha: float,
              index: pd.Index, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = set(selected)
    resid = np.asarray(results.resid, dtype=float)
    exog = np.asarray(results.model.exog, dtype=float)
    rows: list[dict[str, Any]] = []

    def num(value: Any) -> float:
        try:
            value = float(np.asarray(value).squeeze())
            return value if np.isfinite(value) else np.nan
        except Exception:
            return np.nan

    def add(test: str, stat: Any, pvalue: Any, null: str, note: str = "") -> None:
        pvalue = num(pvalue)
        decision = "Review statistic" if np.isnan(pvalue) else ("Do not reject H0" if pvalue >= alpha else "Reject H0")
        rows.append({"Test": test, "Statistic": num(stat), "P-value": pvalue,
                     "Null hypothesis": null, f"Decision at {alpha:.0%}": decision, "Notes": note})

    def attempt(test: str, function: Any) -> None:
        try:
            function()
        except Exception as exc:
            rows.append({"Test": test, "Statistic": np.nan, "P-value": np.nan,
                         "Null hypothesis": "", f"Decision at {alpha:.0%}": "Unavailable",
                         "Notes": str(exc)})

    if "Durbin-Watson" in selected:
        add("Durbin-Watson", durbin_watson(resid), np.nan, "Descriptive statistic", "Near 2 suggests little first-order autocorrelation.")
    if "Jarque-Bera" in selected:
        attempt("Jarque-Bera", lambda: (lambda r: add("Jarque-Bera", r[0], r[1], "Residuals are normally distributed"))(jarque_bera(resid)))
    if "D'Agostino-Pearson" in selected:
        attempt("D'Agostino-Pearson", lambda: (lambda r: add("D'Agostino-Pearson", r[0], r[1], "Residuals are normally distributed"))(scipy_stats.normaltest(resid)))
    if "Breusch-Pagan" in selected:
        def bp() -> None:
            lm, lp, fv, fp = het_breuschpagan(resid, exog)
            add("Breusch-Pagan LM", lm, lp, "Residuals are homoskedastic")
            add("Breusch-Pagan F", fv, fp, "Residuals are homoskedastic")
        attempt("Breusch-Pagan", bp)
    if "White" in selected:
        def white() -> None:
            lm, lp, fv, fp = het_white(resid, exog)
            add("White LM", lm, lp, "Residuals are homoskedastic")
            add("White F", fv, fp, "Residuals are homoskedastic")
        attempt("White", white)
    if "Breusch-Godfrey" in selected:
        def bg() -> None:
            lm, lp, fv, fp = acorr_breusch_godfrey(results, nlags=lag)
            add("Breusch-Godfrey LM", lm, lp, "No residual serial correlation")
            add("Breusch-Godfrey F", fv, fp, "No residual serial correlation")
        attempt("Breusch-Godfrey", bg)
    if "Ljung-Box" in selected:
        attempt("Ljung-Box", lambda: (lambda r: add(f"Ljung-Box Q({lag})", r["lb_stat"], r["lb_pvalue"], "No residual autocorrelation"))(acorr_ljungbox(resid, lags=[lag], return_df=True).iloc[-1]))
    if "ARCH LM" in selected:
        def arch() -> None:
            lm, lp, fv, fp = het_arch(resid, nlags=lag)
            add("ARCH LM", lm, lp, "No ARCH effects")
            add("ARCH F", fv, fp, "No ARCH effects")
        attempt("ARCH LM", arch)
    if "Ramsey RESET" in selected:
        attempt("Ramsey RESET", lambda: (lambda r: add("Ramsey RESET", r.fvalue, r.pvalue, "Correct functional form"))(linear_reset(results, power=2, use_f=True)))
    if "Rainbow linearity" in selected:
        attempt("Rainbow linearity", lambda: (lambda r: add("Rainbow linearity", r[0], r[1], "The relationship is linear"))(linear_rainbow(results)))
    if "CUSUM stability" in selected:
        attempt("CUSUM stability", lambda: (lambda r: add("CUSUM stability", r[0], r[1], "Model parameters are stable"))(breaks_cusumolsresid(resid, ddof=max(1, int(results.df_model) + 1))))
    if "Condition number" in selected:
        add("Condition number", results.condition_number, np.nan, "Descriptive multicollinearity/scaling indicator")

    vif_rows = []
    if "Variance inflation factors" in selected:
        for i, name in enumerate(results.model.exog_names):
            if str(name).lower() in {"const", "intercept"}:
                continue
            try:
                value = float(variance_inflation_factor(exog, i))
            except Exception:
                value = np.nan
            vif_rows.append({"Variable": name, "VIF": value,
                             "Tolerance": 1 / value if np.isfinite(value) and value else np.nan,
                             "Flag": "Review" if np.isfinite(value) and value >= 10 else ""})

    influence = pd.DataFrame()
    if "Influence observations" in selected:
        try:
            frame = OLSInfluence(results).summary_frame()
            influence = pd.DataFrame(index=index)
            influence["Fitted"] = np.asarray(results.fittedvalues)
            influence["Residual"] = resid
            for source, target in [("student_resid", "Studentized residual"),
                                   ("hat_diag", "Leverage"),
                                   ("cooks_d", "Cook's distance"),
                                   ("dffits", "DFFITS")]:
                if source in frame:
                    influence[target] = frame[source].to_numpy()
            cook_cutoff, leverage_cutoff = 4 / max(1, len(frame)), 2 * exog.shape[1] / max(1, len(frame))
            influence["Flag"] = influence.apply(lambda row: "; ".join(
                (["Studentized residual"] if abs(row.get("Studentized residual", 0)) > 2 else []) +
                (["High leverage"] if row.get("Leverage", 0) > leverage_cutoff else []) +
                (["Cook's distance"] if row.get("Cook's distance", 0) > cook_cutoff else [])), axis=1)
            if "Cook's distance" in influence:
                influence = influence.sort_values("Cook's distance", ascending=False)
            influence = influence.head(top_n)
        except Exception as exc:
            influence = pd.DataFrame({"Status": [f"Influence diagnostics unavailable: {exc}"]})
    return pd.DataFrame(rows), pd.DataFrame(vif_rows), influence


def export_code(estimator: str, y: str, x: list[str], constant: bool,
                weight: str | None, tests: list[str], lag: int, alpha: float) -> str:
    columns = [y] + x + ([weight] if weight else [])
    model_line = "model = sm.OLS(y_data, x_data)"
    if estimator == "WLS":
        model_line = f"model = sm.WLS(y_data, x_data, weights=sample[{weight!r}])"
    elif estimator == "GLS":
        model_line = "model = sm.GLS(y_data, x_data)"
    constant_line = "x_data = sm.add_constant(x_data, has_constant='add')" if constant else ""
    return f'''# Diagnostic model and selected tests
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox, breaks_cusumolsresid, het_arch, het_breuschpagan, het_white, linear_rainbow, linear_reset
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
sample = data[{columns!r}].apply(pd.to_numeric, errors="coerce").dropna()
y_data = sample[{y!r}]
x_data = sample[{x!r}]
{constant_line}
{model_line}
results = model.fit()
selected_tests = {tests!r}
lag = {lag}
alpha = {alpha!r}
resid = np.asarray(results.resid, dtype=float)
exog = np.asarray(results.model.exog, dtype=float)
print(results.summary())
print("Durbin-Watson:", durbin_watson(resid))
print("Jarque-Bera:", jarque_bera(resid))
print("Breusch-Pagan:", het_breuschpagan(resid, exog))
print("White:", het_white(resid, exog))
print("Breusch-Godfrey:", acorr_breusch_godfrey(results, nlags=lag))
print("Ljung-Box:", acorr_ljungbox(resid, lags=[lag], return_df=True))
print("ARCH LM:", het_arch(resid, nlags=lag))
print("Ramsey RESET:", linear_reset(results, power=2, use_f=True))
print("Rainbow:", linear_rainbow(results))
print("CUSUM:", breaks_cusumolsresid(resid, ddof=int(results.df_model)+1))
'''


init_state()
apply_theme()
with st.sidebar:
    st.header("Diagnostics")
    st.toggle("Dark mode", key="dark_mode")
    if st.session_state.df is not None:
        st.success(f"{len(st.session_state.df):,} rows × {len(st.session_state.df.columns):,} columns")
        st.caption(st.session_state.source_filename or "Current dataset")

st.title("🧪 Diagnostic Tests")
st.caption("Residual normality, serial correlation, heteroskedasticity, functional form, stability, multicollinearity and influence diagnostics.")

if st.session_state.df is None:
    st.info("No dataset is loaded. Upload one here or use the main Data page.")
    uploaded = st.file_uploader("Upload CSV or Excel data", type=["csv", "xlsx", "xls"])
    if uploaded:
        raw, ext = uploaded.getvalue(), uploaded.name.rsplit(".", 1)[-1].lower()
        loaded = pd.read_csv(io.BytesIO(raw)) if ext == "csv" else pd.read_excel(io.BytesIO(raw))
        loaded.columns = [str(c).strip() for c in loaded.columns]
        st.session_state.df, st.session_state.source_filename = loaded, uploaded.name
        st.rerun()
    st.stop()

df = st.session_state.df
numeric = list(df.select_dtypes(include=[np.number]).columns)
if len(numeric) < 2:
    st.error("At least two numeric variables are required.")
    st.stop()

estimator = st.selectbox("Diagnostic model", ["OLS", "WLS", "GLS"])
y = st.selectbox("Dependent variable", numeric)
x = st.multiselect("Explanatory variables", [c for c in numeric if c != y])
constant = st.checkbox("Include intercept", value=True)
weight = None
if estimator == "WLS":
    options = [c for c in numeric if c not in [y] + x]
    weight = st.selectbox("Positive weight variable", options) if options else None
    if not options:
        st.warning("A separate numeric variable is required for WLS weights.")

c1, c2, c3 = st.columns(3)
lag = int(c1.number_input("Serial-correlation/ARCH lag", 1, 50, 1))
alpha = float(c2.selectbox("Significance level", [0.01, 0.05, 0.10], index=1, format_func=lambda v: f"{v:.0%}"))
top_n = int(c3.number_input("Influential observations shown", 5, 100, 20, 5))
selected = st.multiselect("Diagnostic tests", TESTS, default=TESTS)

run_col, clear_col = st.columns(2)
run = run_col.button("Run diagnostic tests", type="primary", use_container_width=True)
clear = clear_col.button("Clear latest diagnostics", type="secondary", use_container_width=True, disabled=latest_name() is None)
if clear:
    removed = clear_latest()
    st.success(f"Removed {removed!r} from the diagnostics page and full export.")
    st.rerun()

if run:
    try:
        if not x:
            raise ValueError("Select at least one explanatory variable.")
        if not selected:
            raise ValueError("Select at least one diagnostic test.")
        if estimator == "WLS" and not weight:
            raise ValueError("Select a WLS weight variable.")
        columns = [y] + x + ([weight] if weight else [])
        sample = clean_frame(df, columns)
        y_data, x_data = sample[y], sample[x]
        if constant:
            x_data = sm.add_constant(x_data, has_constant="add")
        if estimator == "OLS":
            model = sm.OLS(y_data, x_data)
        elif estimator == "WLS":
            if (sample[weight] <= 0).any():
                raise ValueError("WLS weights must be strictly positive.")
            model = sm.WLS(y_data, x_data, weights=sample[weight])
        else:
            model = sm.GLS(y_data, x_data)
        results = model.fit()
        tests, vif, influence = run_suite(results, selected, lag, alpha, sample.index, top_n)
        coefficients = coefficient_table(results)
        summary = results.summary().as_text()
        plot_data = pd.DataFrame({"Actual": y_data, "Fitted": results.fittedvalues, "Residual": results.resid}, index=sample.index)
        std = plot_data["Residual"].std(ddof=1)
        standardized = (plot_data["Residual"] - plot_data["Residual"].mean()) / std if std else pd.Series(0, index=plot_data.index)
        plot_data["Cumulative standardized residual"] = standardized.cumsum() / np.sqrt(max(1, len(standardized)))
        name = unique_name()
        code = export_code(estimator, y, x, constant, weight, selected, lag, alpha)
        st.session_state.results[name] = tests
        st.session_state.results[name + "_coefficients"] = coefficients
        st.session_state.results[name + "_summary"] = summary
        if not vif.empty:
            st.session_state.results[name + "_vif"] = vif
        if not influence.empty:
            st.session_state.results[name + "_influence"] = influence
        st.session_state.code_blocks.append({"name": name, "code": textwrap.dedent(code).strip()})
        st.session_state.settings_log.append({"name": name, "time": datetime.now().isoformat(timespec="seconds"),
            "settings": {"estimator": estimator, "dependent": y, "explanatory": x, "tests": selected,
            "lag": lag, "significance": alpha, "observations": len(sample)}, "summary": summary})
        st.session_state.history.append({"time": datetime.now().isoformat(timespec="seconds"), "action": f"Completed {name}"})
        stack = st.session_state.slot_outputs.setdefault("diagnostics", [])
        if isinstance(stack, str):
            stack = [stack]
            st.session_state.slot_outputs["diagnostics"] = stack
        stack.append(name)
        st.session_state.diagnostic_page_state = {"name": name, "coefficients": coefficients,
            "tests": tests, "vif": vif, "influence": influence, "plot_data": plot_data, "summary": summary}
        st.success("Diagnostic tests completed and added to the main export.")
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

state = st.session_state.diagnostic_page_state
if state:
    st.caption(f"Displayed diagnostic run: `{state['name']}`")
    st.subheader("Model coefficients")
    show_frame(state["coefficients"])
    st.subheader("Diagnostic-test results")
    show_frame(state["tests"])
    if not state["vif"].empty:
        st.subheader("Variance inflation factors")
        show_frame(state["vif"])
    if not state["influence"].empty:
        st.subheader("Influential observations")
        show_frame(state["influence"])
    with st.expander("Model summary"):
        st.text(state["summary"])
    st.subheader("Diagnostic plots")
    choice = st.selectbox("Plot", ["Residuals against fitted values", "Residual histogram",
        "Normal Q-Q plot", "Actual against fitted values", "Cumulative standardized residuals"])
    data = state["plot_data"]
    fig, ax = plt.subplots()
    if choice == "Residuals against fitted values":
        ax.scatter(data["Fitted"], data["Residual"]); ax.axhline(0); ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
    elif choice == "Residual histogram":
        ax.hist(data["Residual"].dropna(), bins="auto"); ax.set_xlabel("Residual"); ax.set_ylabel("Frequency")
    elif choice == "Normal Q-Q plot":
        sm.qqplot(data["Residual"].dropna(), line="45", ax=ax)
    elif choice == "Actual against fitted values":
        ax.scatter(data["Actual"], data["Fitted"]); lo=min(data["Actual"].min(),data["Fitted"].min()); hi=max(data["Actual"].max(),data["Fitted"].max()); ax.plot([lo,hi],[lo,hi]); ax.set_xlabel("Actual"); ax.set_ylabel("Fitted")
    else:
        ax.plot(np.arange(len(data)), data["Cumulative standardized residual"]); ax.axhline(0); ax.set_xlabel("Observation"); ax.set_ylabel("Cumulative standardized residual")
    ax.set_title(choice); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.divider()
st.caption("Interpret diagnostic tests together with model assumptions, sample size, lag choice and economic theory.")
