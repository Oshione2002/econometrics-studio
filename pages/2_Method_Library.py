from __future__ import annotations

from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from method_catalog import CORE_APP_METHODS, METHOD_CATALOG, RUNNABLE_IN_ADVANCED_PAGE
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.statespace.structural import UnobservedComponents

try:
    from linearmodels.panel import PanelOLS, RandomEffects
    PANEL_OK, PANEL_ERROR = True, ""
except Exception as exc:
    PANEL_OK, PANEL_ERROR = False, str(exc)

st.set_page_config(page_title="Econometrics Studio — Method Library", page_icon="📚", layout="wide")


def init_state() -> None:
    defaults = {
        "df": None, "source_filename": None, "dark_mode": False, "results": {},
        "code_blocks": [], "settings_log": [], "history": [], "advanced_runs": [],
        "advanced_display": None, "advanced_chart": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def theme() -> None:
    if not st.session_state.dark_mode:
        plt.rcdefaults()
        return
    plt.rcParams.update({
        "figure.facecolor": "#0e1117", "axes.facecolor": "#111827",
        "savefig.facecolor": "#0e1117", "text.color": "#f3f4f6",
        "axes.labelcolor": "#f3f4f6", "axes.edgecolor": "#6b7280",
        "xtick.color": "#d1d5db", "ytick.color": "#d1d5db",
        "legend.facecolor": "#111827", "legend.labelcolor": "#f3f4f6",
    })
    st.markdown("""
    <style>
    :root{color-scheme:dark;--b:#0e1117;--p:#161b22;--s:#111827;--s2:#1f2937;
    --bd:#374151;--t:#f3f4f6}
    html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
    [data-testid="stMainBlockContainer"],[data-testid="stHeader"]{background:var(--b)!important;color:var(--t)!important}
    [data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:var(--p)!important}
    .stApp,.stApp p,.stApp label,.stApp span,.stApp li,.stApp h1,.stApp h2,.stApp h3,
    [data-testid="stMarkdownContainer"],[data-testid="stWidgetLabel"],[data-testid="stSidebar"] *{color:var(--t)!important}
    div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,input,textarea{
    background:var(--s2)!important;color:var(--t)!important;border-color:#4b5563!important}
    div[data-baseweb="popover"],div[data-baseweb="menu"],ul[role="listbox"],[role="option"]{
    background:var(--s2)!important;color:var(--t)!important}
    .stButton>button,.stDownloadButton>button{border-color:#4b5563!important;color:var(--t)!important}
    .stButton>button[kind="primary"]{background:#2563eb!important}
    .eco-wrap{width:100%;overflow:auto;max-height:600px;border:1px solid var(--bd);
    border-radius:.6rem;background:var(--s);margin:.25rem 0 1rem}
    table.eco{width:max-content;min-width:100%;border-collapse:collapse;color:var(--t)!important;
    background:var(--s)!important;font-size:.88rem}
    table.eco th,table.eco td{padding:.58rem .7rem;border:1px solid #263244;white-space:nowrap}
    table.eco thead th{position:sticky;top:0;background:#1f2937!important}
    </style>""", unsafe_allow_html=True)


def show(value: Any, height: int = 520) -> None:
    frame = value.to_frame() if isinstance(value, pd.Series) else (
        value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    )
    if not st.session_state.dark_mode:
        st.dataframe(frame, use_container_width=True, height=height)
        return
    formatted = frame.map(lambda x: "" if pd.isna(x) else (f"{float(x):,.6f}" if isinstance(x, (float, np.floating)) else str(x)))
    st.markdown(
        f'<div class="eco-wrap" style="max-height:{height}px">{formatted.to_html(classes="eco", border=0, escape=True)}</div>',
        unsafe_allow_html=True,
    )


def unique(prefix: str) -> str:
    base = f"{prefix}_{datetime.now().strftime('%H%M%S')}"
    name, i = base, 2
    while name in st.session_state.results:
        name, i = f"{base}_{i}", i + 1
    return name


def save(prefix: str, table: pd.DataFrame, code: str, settings: dict[str, Any], chart: dict[str, Any] | None = None) -> str:
    name = unique(prefix)
    st.session_state.results[name] = table
    st.session_state.code_blocks.append({"name": name, "code": code.strip()})
    st.session_state.settings_log.append({
        "name": name, "time": datetime.now().isoformat(timespec="seconds"),
        "settings": settings, "summary": "",
    })
    st.session_state.history.append({
        "time": datetime.now().isoformat(timespec="seconds"), "action": f"Completed {name}",
    })
    st.session_state.advanced_runs.append(name)
    st.session_state.advanced_display, st.session_state.advanced_chart = name, chart
    return name


def clear_latest() -> str | None:
    if not st.session_state.advanced_runs:
        return None
    name = st.session_state.advanced_runs.pop()
    st.session_state.results = {k: v for k, v in st.session_state.results.items() if not (k == name or k.startswith(name + "_"))}
    st.session_state.code_blocks = [v for v in st.session_state.code_blocks if v.get("name") != name]
    st.session_state.settings_log = [v for v in st.session_state.settings_log if v.get("name") != name]
    st.session_state.history = [v for v in st.session_state.history if v.get("action") != f"Completed {name}"]
    st.session_state.advanced_display = st.session_state.advanced_runs[-1] if st.session_state.advanced_runs else None
    st.session_state.advanced_chart = None
    return name


def action(key: str, label: str) -> bool:
    run_col, clear_col = st.columns(2)
    run = run_col.button(label, key=f"run_{key}", type="primary", use_container_width=True)
    if clear_col.button("Clear latest advanced result", key=f"clear_{key}",
                        disabled=not st.session_state.advanced_runs, use_container_width=True):
        removed = clear_latest()
        st.success(f"Removed {removed!r} from results and the reproduction export.")
        st.rerun()
    return run


def params(result: Any) -> pd.DataFrame:
    p = pd.Series(result.params)
    out = pd.DataFrame({"Coefficient": p})
    for label, attr in [("Std. Error", "bse"), ("Statistic", "tvalues"), ("P-value", "pvalues")]:
        value = getattr(result, attr, None)
        if value is not None:
            out[label] = pd.Series(value, index=p.index)
    return out


def catalogue_frame() -> pd.DataFrame:
    rows = []
    for family, categories in METHOD_CATALOG.items():
        for category, methods in categories.items():
            for method in methods:
                status = ("Runnable in Method Library" if method in RUNNABLE_IN_ADVANCED_PAGE
                          else "Implemented in core app" if method in CORE_APP_METHODS
                          else "Catalogue / future module")
                rows.append({"Family": family, "Category": category, "Method": method, "Status": status})
    return pd.DataFrame(rows)


def clean(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = data[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def render_catalogue() -> None:
    frame = catalogue_frame()
    c1, c2, c3 = st.columns(3)
    c1.metric("Catalogued methods", len(frame))
    c2.metric("Runnable here", int(frame.Status.eq("Runnable in Method Library").sum()))
    c3.metric("Implemented in core", int(frame.Status.eq("Implemented in core app").sum()))
    query = st.text_input("Search methods", placeholder="GARCH, fixed effects, cointegration...")
    a, b, c = st.columns(3)
    family = a.selectbox("Family", ["All"] + sorted(frame.Family.unique()))
    categories = sorted(frame.loc[frame.Family.eq(family) if family != "All" else frame.index == frame.index, "Category"].unique())
    category = b.selectbox("Category", ["All"] + categories)
    status = c.selectbox("Status", ["All"] + sorted(frame.Status.unique()))
    out = frame.copy()
    if family != "All": out = out[out.Family == family]
    if category != "All": out = out[out.Category == category]
    if status != "All": out = out[out.Status == status]
    if query:
        out = out[out.astype(str).apply(lambda s: s.str.contains(query, case=False, regex=False)).any(axis=1)]
    show(out.reset_index(drop=True), 620)
    st.download_button("Download catalogue CSV", out.to_csv(index=False).encode(),
                       "econometrics_method_catalogue.csv", "text/csv")
    st.info("Catalogue / future module means documented but not yet exposed as a validated estimator.")


def render_ts(data: pd.DataFrame) -> None:
    numeric = list(data.select_dtypes(include=[np.number]).columns)
    method = st.selectbox("Estimator", [
        "AutoReg / AR", "Simple exponential smoothing", "Holt linear trend",
        "Holt-Winters additive seasonality", "Holt-Winters multiplicative seasonality",
        "Unobserved components", "Markov-switching regression",
    ], key="ml_ts_method")
    variable = st.selectbox("Series", numeric, key="ml_ts_y")
    steps = int(st.number_input("Forecast periods", 1, 200, 10, key="ml_ts_steps"))
    series = pd.to_numeric(data[variable], errors="coerce").dropna()
    settings = {"family": "Time series", "method": method, "series": variable, "forecast_steps": steps}
    if method == "AutoReg / AR":
        lag = int(st.number_input("AR lags", 1, 100, 1)); trend = st.selectbox("Trend", ["n", "c", "ct", "ctt"])
        settings.update(lag=lag, trend=trend)
    elif "Holt-Winters" in method:
        period = int(st.number_input("Seasonal periods", 2, 365, 4)); damped = st.checkbox("Damped trend")
        settings.update(period=period, damped=damped)
    elif method == "Holt linear trend":
        damped = st.checkbox("Damped trend"); exponential = st.checkbox("Exponential trend")
        settings.update(damped=damped, exponential=exponential)
    elif method == "Unobserved components":
        level = st.checkbox("Level", True); trend_component = st.checkbox("Trend component", True)
        seasonal = int(st.number_input("Seasonal periods (0 disables)", 0, 365, 0))
        settings.update(level=level, trend=trend_component, seasonal=seasonal)
    elif method == "Markov-switching regression":
        regimes = int(st.number_input("Regimes", 2, 6, 2)); switching_variance = st.checkbox("Switching variance", True)
        settings.update(regimes=regimes, switching_variance=switching_variance)
    if not action("ts", "Estimate time-series model"): return
    if len(series) < 8: raise ValueError("At least 8 valid observations are required.")
    forecast = pd.Series(dtype=float)
    if method == "AutoReg / AR":
        result = AutoReg(series, lags=lag, trend=trend, old_names=False).fit()
        table, forecast = params(result), result.predict(len(series), len(series) + steps - 1)
        code = f"result = AutoReg(pd.to_numeric(data[{variable!r}], errors='coerce').dropna(), lags={lag}, trend={trend!r}, old_names=False).fit()"
    elif method == "Simple exponential smoothing":
        result = SimpleExpSmoothing(series, initialization_method="estimated").fit()
        table, forecast = pd.DataFrame({"Value": pd.Series(result.params, dtype="object")}), result.forecast(steps)
        code = f"result = SimpleExpSmoothing(pd.to_numeric(data[{variable!r}], errors='coerce').dropna(), initialization_method='estimated').fit()"
    elif method == "Holt linear trend":
        result = Holt(series, exponential=exponential, damped_trend=damped, initialization_method="estimated").fit()
        table, forecast = pd.DataFrame({"Value": pd.Series(result.params, dtype="object")}), result.forecast(steps)
        code = f"result = Holt(pd.to_numeric(data[{variable!r}], errors='coerce').dropna(), exponential={exponential!r}, damped_trend={damped!r}, initialization_method='estimated').fit()"
    elif "Holt-Winters" in method:
        seasonal_type = "add" if "additive" in method else "mul"
        result = ExponentialSmoothing(series, trend="add", seasonal=seasonal_type,
                                      seasonal_periods=period, damped_trend=damped,
                                      initialization_method="estimated").fit()
        table, forecast = pd.DataFrame({"Value": pd.Series(result.params, dtype="object")}), result.forecast(steps)
        code = f"result = ExponentialSmoothing(pd.to_numeric(data[{variable!r}], errors='coerce').dropna(), trend='add', seasonal={seasonal_type!r}, seasonal_periods={period}, damped_trend={damped!r}, initialization_method='estimated').fit()"
    elif method == "Unobserved components":
        result = UnobservedComponents(series, level=level, trend=trend_component,
                                      seasonal=seasonal or None, stochastic_level=level,
                                      stochastic_trend=trend_component).fit(disp=False)
        table, forecast = params(result), result.get_forecast(steps).predicted_mean
        code = f"result = UnobservedComponents(pd.to_numeric(data[{variable!r}], errors='coerce').dropna(), level={level!r}, trend={trend_component!r}, seasonal={seasonal or None!r}).fit(disp=False)"
    else:
        result = MarkovRegression(series, k_regimes=regimes, trend="c",
                                  switching_variance=switching_variance).fit(disp=False)
        table, code = params(result), f"result = MarkovRegression(pd.to_numeric(data[{variable!r}], errors='coerce').dropna(), k_regimes={regimes}, trend='c', switching_variance={switching_variance!r}).fit(disp=False)"
    fitted = pd.Series(np.asarray(result.fittedvalues), index=series.index[-len(result.fittedvalues):])
    name = save("Advanced_time_series", table, code + "\nprint(result.summary())", settings,
                {"kind": "forecast", "actual": series, "fitted": fitted,
                 "forecast": pd.Series(forecast), "title": method})
    st.success(f"Completed {name}.")


def render_reg(data: pd.DataFrame) -> None:
    numeric = list(data.select_dtypes(include=[np.number]).columns)
    method = st.selectbox("Estimator", ["Ridge regression", "Lasso", "Elastic net",
                                        "Polynomial regression", "Recursive least squares"], key="ml_reg_method")
    y_name = st.selectbox("Dependent variable", numeric, key="ml_reg_y")
    x_names = st.multiselect("Explanatory variables", [c for c in numeric if c != y_name], key="ml_reg_x")
    constant = st.checkbox("Include intercept", True, key="ml_reg_const")
    settings = {"family": "Regression", "method": method, "dependent": y_name,
                "explanatory": x_names, "constant": constant}
    if method in {"Ridge regression", "Lasso", "Elastic net"}:
        alpha = float(st.number_input("Penalty strength", 0.000001, 1000000.0, 1.0))
        l1 = 0.0 if method == "Ridge regression" else 1.0 if method == "Lasso" else float(st.slider("L1 weight", 0.0, 1.0, .5, .05))
        settings.update(alpha=alpha, l1_weight=l1)
    elif method == "Polynomial regression":
        degree = int(st.number_input("Polynomial degree", 2, 8, 2)); settings["degree"] = degree
    if not action("reg", "Estimate regression"): return
    if not x_names: raise ValueError("Select explanatory variables.")
    sample = clean(data, [y_name] + x_names); y, X = sample[y_name], sample[x_names].copy()
    if method == "Polynomial regression":
        X = pd.DataFrame({(c if p == 1 else f"{c}^{p}"): X[c] ** p
                          for c in x_names for p in range(1, degree + 1)}, index=X.index)
    if constant: X = sm.add_constant(X, has_constant="add")
    if method in {"Ridge regression", "Lasso", "Elastic net"}:
        result = sm.OLS(y, X).fit_regularized(method="elastic_net", alpha=alpha, L1_wt=l1)
        fitted = np.asarray(X) @ np.asarray(result.params)
        table = pd.DataFrame({"Coefficient": pd.Series(result.params, index=X.columns)})
        residual = y.to_numpy() - fitted
        table.loc["[Model] RMSE"] = np.sqrt(np.mean(residual ** 2))
        table.loc["[Model] MAE"] = np.mean(np.abs(residual))
        code = f"result = sm.OLS(y, X).fit_regularized(method='elastic_net', alpha={alpha!r}, L1_wt={l1!r})"
    elif method == "Polynomial regression":
        result = sm.OLS(y, X).fit(); fitted, table = result.fittedvalues, params(result)
        code = f"# Expand {x_names!r} to polynomial degree {degree}, then fit sm.OLS(y, X).fit()"
    else:
        result = RecursiveLS(y, X).fit(); fitted, table = result.fittedvalues, params(result)
        code = "result = RecursiveLS(y, X).fit()"
    name = save("Advanced_regression", table, code + "\nprint(result.params)", settings,
                {"kind": "actual", "actual": y, "fitted": pd.Series(np.asarray(fitted), index=y.index), "title": method})
    st.success(f"Completed {name}.")


def panel_data(data: pd.DataFrame, entity: str, time: str, columns: list[str]) -> pd.DataFrame:
    frame = data[[entity, time] + columns].copy()
    for c in columns: frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.replace([np.inf, -np.inf], np.nan).dropna()


def render_panel(data: pd.DataFrame) -> None:
    if not PANEL_OK:
        st.error(f"linearmodels unavailable: {PANEL_ERROR}"); return
    numeric = list(data.select_dtypes(include=[np.number]).columns)
    method = st.selectbox("Estimator", ["Two-way fixed-effects DiD", "Mundlak correlated random effects",
                                        "Fixed effects with Driscoll-Kraay covariance",
                                        "Common Correlated Effects Pooled"], key="ml_panel_method")
    entity = st.selectbox("Entity identifier", list(data.columns), key="ml_panel_entity")
    time = st.selectbox("Time identifier", [c for c in data.columns if c != entity], key="ml_panel_time")
    y_name = st.selectbox("Dependent variable", [c for c in numeric if c not in {entity, time}], key="ml_panel_y")
    x_names = st.multiselect("Explanatory variables", [c for c in numeric if c not in {entity, time, y_name}], key="ml_panel_x")
    settings = {"family": "Panel data", "method": method, "entity": entity, "time": time,
                "dependent": y_name, "explanatory": x_names}
    if method == "Two-way fixed-effects DiD":
        treatment = st.selectbox("Treatment indicator", [c for c in numeric if c != y_name], key="ml_panel_treat")
        post = st.selectbox("Post indicator", [c for c in numeric if c not in {y_name, treatment}], key="ml_panel_post")
        settings.update(treatment=treatment, post=post)
    if not action("panel", "Estimate panel model"): return
    if method != "Two-way fixed-effects DiD" and not x_names: raise ValueError("Select explanatory variables.")
    if method == "Two-way fixed-effects DiD":
        controls = [c for c in x_names if c not in {treatment, post}]
        sample = panel_data(data, entity, time, list(dict.fromkeys([y_name, treatment, post] + controls)))
        sample["Treatment × Post"] = sample[treatment] * sample[post]
        X = pd.concat([sample[[treatment, post, "Treatment × Post"] + controls].reset_index(drop=True),
                       pd.get_dummies(sample[entity].astype(str), drop_first=True, dtype=float).reset_index(drop=True),
                       pd.get_dummies(sample[time].astype(str), drop_first=True, dtype=float).reset_index(drop=True)], axis=1)
        X = sm.add_constant(X, has_constant="add").astype(float)
        y = sample[y_name].reset_index(drop=True)
        result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sample[entity].reset_index(drop=True)})
        table, fitted = params(result), result.fittedvalues
        code = "# Two-way fixed-effects DiD with treatment, post, interaction, entity dummies and time dummies."
    elif method == "Mundlak correlated random effects":
        sample = panel_data(data, entity, time, [y_name] + x_names)
        means = sample.groupby(entity)[x_names].transform("mean"); means.columns = [f"Mean({c})" for c in x_names]
        work = pd.concat([sample, means], axis=1).set_index([entity, time]).sort_index()
        X = sm.add_constant(work[x_names + list(means.columns)], has_constant="add")
        result = RandomEffects(work[y_name], X).fit(cov_type="robust")
        table = pd.DataFrame({"Coefficient": result.params, "Std. Error": result.std_errors,
                              "Statistic": result.tstats, "P-value": result.pvalues})
        fitted, y = np.asarray(result.fitted_values).ravel(), work[y_name]
        code = "# RandomEffects with entity means of time-varying regressors (Mundlak specification)."
    elif method == "Fixed effects with Driscoll-Kraay covariance":
        sample = panel_data(data, entity, time, [y_name] + x_names)
        work = sample.set_index([entity, time]).sort_index()
        X = sm.add_constant(work[x_names], has_constant="add")
        result = PanelOLS(work[y_name], X, entity_effects=True, time_effects=True, drop_absorbed=True).fit(cov_type="kernel")
        table = pd.DataFrame({"Coefficient": result.params, "Std. Error": result.std_errors,
                              "Statistic": result.tstats, "P-value": result.pvalues})
        fitted, y = np.asarray(result.fitted_values).ravel(), work[y_name]
        code = "# PanelOLS with entity/time effects and kernel (Driscoll-Kraay) covariance."
    else:
        sample = panel_data(data, entity, time, [y_name] + x_names)
        for c in [y_name] + x_names: sample[f"CSMean({c})"] = sample.groupby(time)[c].transform("mean")
        regressors = x_names + [f"CSMean({c})" for c in [y_name] + x_names]
        X = pd.concat([sample[regressors].reset_index(drop=True),
                       pd.get_dummies(sample[entity].astype(str), drop_first=True, dtype=float).reset_index(drop=True)], axis=1)
        X = sm.add_constant(X, has_constant="add").astype(float); y = sample[y_name].reset_index(drop=True)
        result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sample[entity].reset_index(drop=True)})
        table, fitted = params(result), result.fittedvalues
        code = "# CCEP screening regression with cross-sectional averages and entity fixed effects."
    name = save("Advanced_panel", table, code + "\nprint(result.params)", settings,
                {"kind": "actual", "actual": pd.Series(np.asarray(y)), "fitted": pd.Series(np.asarray(fitted).ravel()), "title": method})
    st.success(f"Completed {name}.")


def display_latest() -> None:
    name = st.session_state.advanced_display
    if name and name in st.session_state.results:
        st.subheader(f"Latest advanced result — {name}"); show(st.session_state.results[name])
    chart = st.session_state.advanced_chart
    if not chart: return
    fig, ax = plt.subplots()
    if chart["kind"] == "forecast":
        actual, fitted, forecast = chart["actual"], chart["fitted"], chart["forecast"]
        ax.plot(np.arange(len(actual)), actual.to_numpy(), label="Actual")
        ax.plot(np.arange(len(actual)-len(fitted), len(actual)), fitted.to_numpy(), label="Fitted")
        if len(forecast): ax.plot(np.arange(len(actual), len(actual)+len(forecast)), forecast.to_numpy(), label="Forecast")
        ax.legend()
    else:
        actual, fitted = np.asarray(chart["actual"]), np.asarray(chart["fitted"])
        ax.scatter(actual, fitted, alpha=.65)
        lo, hi = min(np.nanmin(actual), np.nanmin(fitted)), max(np.nanmax(actual), np.nanmax(fitted))
        ax.plot([lo, hi], [lo, hi]); ax.set_xlabel("Actual"); ax.set_ylabel("Fitted")
    ax.set_title(chart["title"]); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    if st.button("Clear advanced chart", use_container_width=True):
        st.session_state.advanced_chart = None; st.rerun()


init_state(); theme()
with st.sidebar:
    st.header("Method Library")
    st.toggle("Dark mode", key="dark_mode")
    if st.session_state.source_filename: st.caption(st.session_state.source_filename)

st.title("📚 Method Library & Advanced Models")
st.caption("Search the full method catalogue and run additional stable estimators. Methods are never labelled operational until a working implementation is present.")
catalog_tab, run_tab, roadmap_tab = st.tabs(["Full method catalogue", "Run advanced methods", "Implementation roadmap"])
with catalog_tab: render_catalogue()
with run_tab:
    data = st.session_state.df
    if data is None:
        st.info("Load a dataset in the main Data workspace first.")
    else:
        family = st.radio("Advanced family", ["Time series", "Regression", "Panel data"], horizontal=True)
        try:
            if family == "Time series": render_ts(data)
            elif family == "Regression": render_reg(data)
            else: render_panel(data)
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
        display_latest()
with roadmap_tab:
    st.subheader("Implementation policy")
    st.write("Stable estimators from statsmodels, linearmodels and arch are added first. Spatial, survival, dynamic-panel, causal, machine-learning and deep-learning packages are introduced only after Python-version and Streamlit Cloud compatibility tests.")
    show(pd.DataFrame([
        ["Core stable", "statsmodels, linearmodels, arch", "Normal cloud deployment"],
        ["Extended statistical", "survival, spatial, dynamic panel, DiD, synthetic control", "Compatibility-tested optional modules"],
        ["Machine learning", "scikit-learn and boosting libraries", "Optional dependency profile"],
        ["Deep learning / foundation", "PyTorch, transformers and hosted APIs", "Separate deployment or external service"],
        ["Research implementations", "methods without maintained packages", "Individually validated experimental modules"],
    ], columns=["Release group", "Scope", "Deployment"]), 360)
    st.warning("The catalogue is not a claim that every method is appropriate for every dataset. Identification, assumptions and diagnostics remain the researcher's responsibility.")
