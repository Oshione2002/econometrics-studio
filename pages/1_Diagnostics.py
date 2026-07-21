from __future__ import annotations

import math
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

st.set_page_config(
    page_title="Econometrics Studio — Post-estimation diagnostics",
    page_icon="🧪",
    layout="wide",
)


def init_state() -> None:
    defaults = {
        "df": None,
        "source_filename": None,
        "history": [],
        "code_blocks": [],
        "results": {},
        "settings_log": [],
        "slot_outputs": {},
        "model_registry": {},
        "diagnostic_runs": {},
        "diagnostic_page_state": None,
        "dark_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_theme() -> None:
    if not st.session_state.dark_mode:
        plt.rcdefaults()
        return

    plt.rcParams.update(
        {
            "figure.facecolor": "#0e1117",
            "axes.facecolor": "#111827",
            "savefig.facecolor": "#0e1117",
            "text.color": "#f3f4f6",
            "axes.labelcolor": "#f3f4f6",
            "axes.edgecolor": "#6b7280",
            "xtick.color": "#d1d5db",
            "ytick.color": "#d1d5db",
            "grid.color": "#374151",
            "legend.facecolor": "#111827",
            "legend.edgecolor": "#4b5563",
            "legend.labelcolor": "#f3f4f6",
        }
    )
    st.markdown(
        """
        <style>
        :root{color-scheme:dark;--bg:#0e1117;--panel:#161b22;--surface:#111827;
        --surface2:#1f2937;--border:#374151;--text:#f3f4f6;--muted:#cbd5e1}
        html,body,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],
        [data-testid="stMainBlockContainer"],[data-testid="stHeader"]{
            background:var(--bg)!important;color:var(--text)!important}
        [data-testid="stSidebar"],[data-testid="stSidebarContent"]{
            background:var(--panel)!important}
        .stApp,.stApp p,.stApp label,.stApp span,.stApp li,.stApp h1,.stApp h2,
        .stApp h3,.stApp h4,[data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],[data-testid="stSidebar"] *{
            color:var(--text)!important}
        [data-testid="stCaptionContainer"],small{color:var(--muted)!important}
        div[data-baseweb="select"]>div,div[data-baseweb="input"]>div,input,textarea{
            background:var(--surface2)!important;color:var(--text)!important;
            border-color:#4b5563!important}
        div[data-baseweb="popover"],div[data-baseweb="menu"],ul[role="listbox"],
        [role="option"]{background:var(--surface2)!important;color:var(--text)!important}
        .stButton>button{border-color:#4b5563!important;color:var(--text)!important}
        .stButton>button[kind="primary"]{background:#2563eb!important}
        [data-testid="stCodeBlock"],pre,code{
            background:var(--surface)!important;color:#e5e7eb!important}
        .eco-wrap{width:100%;overflow:auto;max-height:560px;border:1px solid var(--border);
            border-radius:.6rem;background:var(--surface);margin:.25rem 0 1rem}
        table.eco{width:max-content;min-width:100%;border-collapse:separate;
            border-spacing:0;color:var(--text)!important;background:var(--surface)!important;
            font-size:.88rem}
        table.eco thead th{position:sticky;top:0;background:#1f2937!important;
            color:#f8fafc!important}
        table.eco tbody th{position:sticky;left:0;background:#172033!important;
            color:#e2e8f0!important}
        table.eco th,table.eco td{padding:.62rem .72rem;border-right:1px solid #263244;
            border-bottom:1px solid #263244;white-space:nowrap}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_frame(value: Any) -> None:
    if isinstance(value, pd.Series):
        frame = value.to_frame()
    elif isinstance(value, pd.DataFrame):
        frame = value.copy()
    else:
        frame = pd.DataFrame(value)

    if not st.session_state.dark_mode:
        st.dataframe(frame, use_container_width=True)
        return

    def fmt(item: Any) -> str:
        if pd.isna(item):
            return ""
        if isinstance(item, (float, np.floating)):
            return f"{float(item):,.6f}"
        return str(item)

    formatted = frame.map(fmt)
    html = formatted.to_html(classes="eco", border=0, escape=True)
    st.markdown(f'<div class="eco-wrap">{html}</div>', unsafe_allow_html=True)


def safe_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim > 1 and array.shape[1] == 1:
        array = array[:, 0]
    return array.astype(float).reshape(-1)


def finite_series(value: Any) -> pd.Series:
    series = pd.Series(safe_array(value))
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def decision(p_value: Any, alpha: float) -> str:
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return "See statistic"
    if not np.isfinite(p):
        return "See statistic"
    return "Do not reject null" if p >= alpha else "Reject null"


def add_test(
    rows: list[dict[str, Any]],
    test: str,
    statistic: Any,
    p_value: Any = np.nan,
    null: str = "",
    alpha: float = 0.05,
    note: str = "",
) -> None:
    rows.append(
        {
            "Test": test,
            "Statistic": statistic,
            "P-value": p_value,
            "Null hypothesis": null,
            f"Decision at {alpha:.0%}": decision(p_value, alpha),
            "Note": note,
        }
    )


def model_residuals(entry: dict[str, Any]) -> pd.Series:
    result = entry["result"]
    for attribute in ("resid", "resids", "resid_response"):
        if hasattr(result, attribute):
            try:
                return finite_series(getattr(result, attribute))
            except Exception:
                continue
    raise ValueError("The selected fitted result does not expose usable residuals.")


def model_fitted(entry: dict[str, Any]) -> pd.Series:
    result = entry["result"]
    for attribute in ("fittedvalues", "fitted_values", "predicted_values"):
        if hasattr(result, attribute):
            try:
                return finite_series(getattr(result, attribute))
            except Exception:
                continue
    if hasattr(result, "predict"):
        try:
            return finite_series(result.predict())
        except Exception:
            pass
    raise ValueError("The selected fitted result does not expose fitted values.")


def model_endog(entry: dict[str, Any]) -> pd.Series:
    result = entry["result"]
    model = getattr(result, "model", None)
    if model is not None and hasattr(model, "endog"):
        return finite_series(model.endog)

    sample = entry.get("sample")
    dependent = entry.get("metadata", {}).get("dependent")
    if isinstance(sample, pd.DataFrame) and dependent in sample.columns:
        return finite_series(sample[dependent])
    raise ValueError("The selected fitted result does not expose the dependent variable.")


def model_exog(entry: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    result = entry["result"]
    model = getattr(result, "model", None)
    if model is not None and hasattr(model, "exog"):
        exog = np.asarray(model.exog, dtype=float)
        names = list(getattr(model, "exog_names", []))
        if not names:
            names = [f"x{i}" for i in range(exog.shape[1])]
        return exog, names

    sample = entry.get("sample")
    explanatory = entry.get("metadata", {}).get("explanatory", [])
    if isinstance(sample, pd.DataFrame) and explanatory:
        frame = sample[explanatory].apply(pd.to_numeric, errors="coerce").dropna()
        return frame.to_numpy(dtype=float), list(frame.columns)

    raise ValueError("The selected fitted result does not expose an explanatory-variable matrix.")


def normality_tests(rows: list[dict[str, Any]], residuals: pd.Series, alpha: float) -> None:
    if len(residuals) >= 3:
        jb_stat, jb_p, _, _ = jarque_bera(residuals)
        add_test(
            rows,
            "Jarque-Bera normality",
            jb_stat,
            jb_p,
            "Residuals are normally distributed",
            alpha,
        )
    if len(residuals) >= 8:
        stat, p_value = scipy_stats.normaltest(residuals)
        add_test(
            rows,
            "D'Agostino-Pearson normality",
            stat,
            p_value,
            "Residuals are normally distributed",
            alpha,
        )


def serial_arch_tests(
    rows: list[dict[str, Any]],
    residuals: pd.Series,
    lag: int,
    alpha: float,
) -> None:
    usable_lag = min(max(1, int(lag)), max(1, len(residuals) // 4))
    add_test(
        rows,
        "Durbin-Watson",
        durbin_watson(residuals),
        np.nan,
        "Descriptive statistic; values near 2 indicate little first-order autocorrelation",
        alpha,
    )
    try:
        lb = acorr_ljungbox(residuals, lags=[usable_lag], return_df=True).iloc[-1]
        add_test(
            rows,
            f"Ljung-Box Q({usable_lag})",
            lb["lb_stat"],
            lb["lb_pvalue"],
            "No residual serial correlation through the selected lag",
            alpha,
        )
    except Exception:
        pass
    try:
        lm, lm_p, f_stat, f_p = het_arch(residuals, nlags=usable_lag)
        add_test(rows, f"ARCH LM({usable_lag})", lm, lm_p, "No ARCH effects", alpha)
        add_test(rows, f"ARCH F({usable_lag})", f_stat, f_p, "No ARCH effects", alpha)
    except Exception:
        pass


def linear_diagnostics(
    entry: dict[str, Any],
    lag: int,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    result = entry["result"]
    estimator = entry["estimator"]
    residuals = model_residuals(entry)
    rows: list[dict[str, Any]] = []

    normality_tests(rows, residuals, alpha)
    serial_arch_tests(rows, residuals, lag, alpha)

    exog = None
    names: list[str] = []
    try:
        exog, names = model_exog(entry)
    except Exception:
        pass

    if exog is not None and len(exog) == len(residuals):
        try:
            lm, lm_p, f_stat, f_p = het_breuschpagan(residuals, exog)
            add_test(rows, "Breusch-Pagan LM", lm, lm_p, "Homoskedasticity", alpha)
            add_test(rows, "Breusch-Pagan F", f_stat, f_p, "Homoskedasticity", alpha)
        except Exception:
            pass
        try:
            lm, lm_p, f_stat, f_p = het_white(residuals, exog)
            add_test(rows, "White LM", lm, lm_p, "Homoskedasticity", alpha)
            add_test(rows, "White F", f_stat, f_p, "Homoskedasticity", alpha)
        except Exception:
            pass

    if estimator in {"OLS", "WLS", "GLS"}:
        try:
            lm, lm_p, f_stat, f_p = acorr_breusch_godfrey(result, nlags=int(lag))
            add_test(rows, f"Breusch-Godfrey LM({lag})", lm, lm_p, "No serial correlation", alpha)
            add_test(rows, f"Breusch-Godfrey F({lag})", f_stat, f_p, "No serial correlation", alpha)
        except Exception:
            pass
        try:
            reset = linear_reset(result, power=2, use_f=True)
            add_test(rows, "Ramsey RESET", float(reset.fvalue), float(reset.pvalue), "Correct functional form", alpha)
        except Exception:
            pass
        try:
            stat, p_value = linear_rainbow(result)
            add_test(rows, "Rainbow linearity", stat, p_value, "The relationship is linear", alpha)
        except Exception:
            pass
        try:
            stat, p_value, critical = breaks_cusumolsresid(residuals, ddof=int(getattr(result, "df_model", 0)) + 1)
            add_test(
                rows,
                "CUSUM parameter stability",
                stat,
                p_value,
                "Parameters are stable",
                alpha,
                f"Critical values: {critical}",
            )
        except Exception:
            pass

    vif = pd.DataFrame()
    if exog is not None:
        vif_rows = []
        for index, name in enumerate(names):
            try:
                value = float(variance_inflation_factor(exog, index))
            except Exception:
                value = np.nan
            vif_rows.append(
                {
                    "Variable": name,
                    "VIF": value,
                    "Tolerance": np.nan if not np.isfinite(value) or value == 0 else 1 / value,
                }
            )
        vif = pd.DataFrame(vif_rows).set_index("Variable")

        try:
            condition = float(np.linalg.cond(exog))
            add_test(
                rows,
                "Design-matrix condition number",
                condition,
                np.nan,
                "",
                alpha,
                "Large values indicate potential multicollinearity or scaling problems.",
            )
        except Exception:
            pass

    influence = pd.DataFrame()
    if estimator in {"OLS", "WLS", "GLS"}:
        try:
            influence_object = OLSInfluence(result)
            influence = pd.DataFrame(
                {
                    "Studentized residual": influence_object.resid_studentized_external,
                    "Leverage": influence_object.hat_matrix_diag,
                    "Cook's distance": influence_object.cooks_distance[0],
                    "DFFITS": influence_object.dffits[0],
                }
            )
            influence["Influential"] = (
                (np.abs(influence["Studentized residual"]) > 2)
                | (influence["Cook's distance"] > 4 / max(1, len(influence)))
                | (influence["Leverage"] > 2 * exog.shape[1] / max(1, len(influence)))
            )
            influence = influence.sort_values("Cook's distance", ascending=False).head(30)
        except Exception:
            pass

    return {
        "tests": pd.DataFrame(rows).set_index("Test"),
        "vif": vif,
        "influence": influence,
    }


def binary_diagnostics(
    entry: dict[str, Any],
    alpha: float,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    result = entry["result"]
    observed = model_endog(entry)
    predicted = finite_series(result.predict())
    size = min(len(observed), len(predicted))
    observed = observed.iloc[:size].reset_index(drop=True)
    predicted = predicted.iloc[:size].reset_index(drop=True).clip(1e-9, 1 - 1e-9)
    classified = (predicted >= threshold).astype(int)

    tp = int(((observed == 1) & (classified == 1)).sum())
    tn = int(((observed == 0) & (classified == 0)).sum())
    fp = int(((observed == 0) & (classified == 1)).sum())
    fn = int(((observed == 1) & (classified == 0)).sum())

    confusion = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Observed 0", "Observed 1"],
        columns=["Predicted 0", "Predicted 1"],
    )

    metrics = {
        "Accuracy": (tp + tn) / max(1, size),
        "Sensitivity": tp / max(1, tp + fn),
        "Specificity": tn / max(1, tn + fp),
        "Positive predictive value": tp / max(1, tp + fp),
        "Negative predictive value": tn / max(1, tn + fn),
    }

    positives = observed == 1
    n1 = int(positives.sum())
    n0 = int((~positives).sum())
    if n1 and n0:
        ranks = scipy_stats.rankdata(predicted)
        auc = (ranks[positives].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        metrics["ROC AUC"] = float(auc)

    rows: list[dict[str, Any]] = []
    try:
        groups = min(10, max(3, size // 10))
        grouped = pd.DataFrame({"y": observed, "p": predicted})
        grouped["group"] = pd.qcut(grouped["p"], q=groups, duplicates="drop")
        hl = grouped.groupby("group", observed=False).agg(
            observed=("y", "sum"),
            expected=("p", "sum"),
            count=("y", "size"),
            mean_probability=("p", "mean"),
        )
        variance = hl["expected"] * (1 - hl["expected"] / hl["count"]).clip(lower=1e-9)
        statistic = float((((hl["observed"] - hl["expected"]) ** 2) / variance).sum())
        degrees = max(1, len(hl) - 2)
        p_value = float(scipy_stats.chi2.sf(statistic, degrees))
        add_test(
            rows,
            "Hosmer-Lemeshow goodness of fit",
            statistic,
            p_value,
            "The fitted probabilities agree with grouped observations",
            alpha,
            f"Degrees of freedom: {degrees}",
        )
    except Exception:
        hl = pd.DataFrame()

    pearson = (observed - predicted) / np.sqrt(predicted * (1 - predicted))
    normality_tests(rows, pearson, alpha)

    if hasattr(result, "llr_pvalue"):
        add_test(
            rows,
            "Likelihood-ratio model significance",
            getattr(result, "llr", np.nan),
            result.llr_pvalue,
            "All slope coefficients are jointly zero",
            alpha,
        )

    vif = pd.DataFrame()
    try:
        exog, names = model_exog(entry)
        vif_rows = []
        for index, name in enumerate(names):
            value = float(variance_inflation_factor(exog, index))
            vif_rows.append(
                {
                    "Variable": name,
                    "VIF": value,
                    "Tolerance": np.nan if value == 0 else 1 / value,
                }
            )
        vif = pd.DataFrame(vif_rows).set_index("Variable")
    except Exception:
        pass

    return {
        "tests": pd.DataFrame(rows).set_index("Test"),
        "classification": pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]),
        "confusion": confusion,
        "grouped_fit": hl,
        "vif": vif,
        "plot_data": pd.DataFrame({"Observed": observed, "Probability": predicted, "Pearson residual": pearson}),
    }


def count_diagnostics(
    entry: dict[str, Any],
    lag: int,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    result = entry["result"]
    observed = model_endog(entry)
    predicted = finite_series(result.predict()).clip(lower=1e-9)
    size = min(len(observed), len(predicted))
    observed = observed.iloc[:size].reset_index(drop=True)
    predicted = predicted.iloc[:size].reset_index(drop=True)

    pearson = (observed - predicted) / np.sqrt(predicted)
    pearson_chi2 = float(np.sum(pearson**2))
    df_resid = max(1.0, float(getattr(result, "df_resid", size - len(getattr(result, "params", [])))))
    dispersion = pearson_chi2 / df_resid
    deviance_core = (
        np.where(observed > 0, observed * np.log(observed / predicted), 0)
        - (observed - predicted)
    )
    deviance_residual = np.sign(observed - predicted) * np.sqrt(
        2 * np.clip(deviance_core, a_min=0, a_max=None)
    )

    rows: list[dict[str, Any]] = []
    add_test(
        rows,
        "Pearson goodness of fit",
        pearson_chi2,
        scipy_stats.chi2.sf(pearson_chi2, df_resid),
        "The conditional mean/variance specification fits the counts",
        alpha,
        f"Degrees of freedom: {df_resid:.0f}",
    )
    add_test(
        rows,
        "Dispersion ratio",
        dispersion,
        np.nan,
        "",
        alpha,
        "Values materially above 1 indicate overdispersion; values below 1 indicate underdispersion.",
    )
    if hasattr(result, "llr_pvalue"):
        add_test(
            rows,
            "Likelihood-ratio model significance",
            getattr(result, "llr", np.nan),
            result.llr_pvalue,
            "All slope coefficients are jointly zero",
            alpha,
        )
    normality_tests(rows, deviance_residual, alpha)
    serial_arch_tests(rows, pearson, lag, alpha)

    zero_table = pd.DataFrame(
        {
            "Observed zero proportion": [(observed == 0).mean()],
            "Mean predicted zero probability": [np.exp(-predicted).mean()],
            "Difference": [(observed == 0).mean() - np.exp(-predicted).mean()],
        }
    )

    return {
        "tests": pd.DataFrame(rows).set_index("Test"),
        "zero_counts": zero_table,
        "plot_data": pd.DataFrame(
            {
                "Observed": observed,
                "Fitted": predicted,
                "Pearson residual": pearson,
                "Deviance residual": deviance_residual,
            }
        ),
    }


def time_series_diagnostics(
    entry: dict[str, Any],
    lag: int,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    residuals = model_residuals(entry)
    rows: list[dict[str, Any]] = []
    normality_tests(rows, residuals, alpha)
    serial_arch_tests(rows, residuals, lag, alpha)

    result = entry["result"]
    if entry["family"] == "ardl":
        try:
            lm, lm_p, f_stat, f_p = acorr_breusch_godfrey(result, nlags=lag)
            add_test(rows, f"Breusch-Godfrey LM({lag})", lm, lm_p, "No serial correlation", alpha)
            add_test(rows, f"Breusch-Godfrey F({lag})", f_stat, f_p, "No serial correlation", alpha)
        except Exception:
            pass
        try:
            reset = linear_reset(result, power=2, use_f=True)
            add_test(rows, "Ramsey RESET", float(reset.fvalue), float(reset.pvalue), "Correct functional form", alpha)
        except Exception:
            pass

    fitted = pd.Series(dtype=float)
    actual = pd.Series(dtype=float)
    try:
        fitted = model_fitted(entry)
        actual = model_endog(entry)
    except Exception:
        pass

    size = min(len(residuals), len(fitted), len(actual)) if len(fitted) and len(actual) else len(residuals)
    plot_data = pd.DataFrame({"Residual": residuals.iloc[:size].reset_index(drop=True)})
    if len(fitted) and len(actual):
        plot_data["Fitted"] = fitted.iloc[:size].reset_index(drop=True)
        plot_data["Actual"] = actual.iloc[-size:].reset_index(drop=True)

    return {"tests": pd.DataFrame(rows).set_index("Test"), "plot_data": plot_data}


def multivariate_diagnostics(
    entry: dict[str, Any],
    lag: int,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    result = entry["result"]
    rows: list[dict[str, Any]] = []
    requested_lag = max(1, int(lag))
    try:
        whiteness = result.test_whiteness(nlags=requested_lag)
        add_test(
            rows,
            f"Multivariate residual whiteness ({requested_lag})",
            getattr(whiteness, "test_statistic", np.nan),
            getattr(whiteness, "pvalue", np.nan),
            "No residual serial correlation",
            alpha,
            str(getattr(whiteness, "conclusion_str", "")),
        )
    except Exception as exc:
        add_test(rows, "Multivariate residual whiteness", np.nan, np.nan, "", alpha, f"Unavailable: {exc}")

    try:
        normality = result.test_normality()
        add_test(
            rows,
            "Multivariate normality",
            getattr(normality, "test_statistic", np.nan),
            getattr(normality, "pvalue", np.nan),
            "Residuals are multivariate normal",
            alpha,
            str(getattr(normality, "conclusion_str", "")),
        )
    except Exception as exc:
        add_test(rows, "Multivariate normality", np.nan, np.nan, "", alpha, f"Unavailable: {exc}")

    roots = pd.DataFrame()
    stable = np.nan
    try:
        values = np.asarray(result.roots)
        roots = pd.DataFrame({"Root": values, "Modulus": np.abs(values)})
        stable = bool(result.is_stable(verbose=False))
        add_test(
            rows,
            "Dynamic stability",
            float(np.max(np.abs(values))) if len(values) else np.nan,
            np.nan,
            "All companion roots satisfy the model stability condition",
            alpha,
            f"Stable: {stable}",
        )
    except Exception:
        pass

    residuals = getattr(result, "resid", None)
    residual_frame = pd.DataFrame(residuals) if residuals is not None else pd.DataFrame()

    return {
        "tests": pd.DataFrame(rows).set_index("Test"),
        "roots": roots,
        "residuals": residual_frame,
    }


def panel_diagnostics(
    entry: dict[str, Any],
    lag: int,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    result = entry["result"]
    residuals = model_residuals(entry)
    rows: list[dict[str, Any]] = []
    normality_tests(rows, residuals, alpha)
    serial_arch_tests(rows, residuals, lag, alpha)

    residual_object = getattr(result, "resids", None)
    panel_table = pd.DataFrame()
    if isinstance(residual_object, pd.Series) and isinstance(residual_object.index, pd.MultiIndex):
        try:
            wide = residual_object.unstack(level=0)
            corr = wide.corr(min_periods=3)
            n = corr.shape[0]
            valid = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack()
            t_bar = wide.notna().sum().mean()
            if n > 1 and len(valid):
                cd = math.sqrt(2 * t_bar / (n * (n - 1))) * valid.sum()
                p_value = 2 * scipy_stats.norm.sf(abs(cd))
                add_test(
                    rows,
                    "Pesaran cross-sectional dependence",
                    cd,
                    p_value,
                    "Cross-sectional independence",
                    alpha,
                )

            variances = residual_object.groupby(level=0).var()
            panel_table = variances.to_frame("Residual variance")
            if len(variances) > 1:
                ratio = float(variances.max() / variances.min()) if variances.min() > 0 else np.inf
                add_test(
                    rows,
                    "Groupwise residual-variance ratio",
                    ratio,
                    np.nan,
                    "",
                    alpha,
                    "Large ratios indicate possible groupwise heteroskedasticity.",
                )
        except Exception:
            pass

    return {"tests": pd.DataFrame(rows).set_index("Test"), "entity_variance": panel_table}


def extract_wald_test(value: Any) -> tuple[Any, Any, str]:
    return (
        getattr(value, "stat", getattr(value, "statistic", np.nan)),
        getattr(value, "pval", getattr(value, "pvalue", np.nan)),
        str(getattr(value, "null", "")),
    )


def iv_diagnostics(
    entry: dict[str, Any],
    alpha: float,
) -> dict[str, pd.DataFrame]:
    result = entry["result"]
    rows: list[dict[str, Any]] = []
    first_stage = pd.DataFrame()

    try:
        first_stage = result.first_stage.diagnostics.copy()
    except Exception:
        pass

    for label, attribute in [
        ("Durbin endogeneity", "durbin"),
        ("Wu-Hausman endogeneity", "wu_hausman"),
    ]:
        try:
            test = getattr(result, attribute)()
            statistic, p_value, null = extract_wald_test(test)
            add_test(rows, label, statistic, p_value, null or "Regressors are exogenous", alpha)
        except Exception:
            pass

    for label, attribute in [
        ("Sargan overidentification", "sargan"),
        ("Basmann overidentification", "basmann"),
        ("Anderson-Rubin", "anderson_rubin"),
        ("Wooldridge overidentification", "wooldridge_overid"),
    ]:
        try:
            test = getattr(result, attribute)
            statistic, p_value, null = extract_wald_test(test)
            add_test(
                rows,
                label,
                statistic,
                p_value,
                null or "The overidentifying restrictions are valid",
                alpha,
            )
        except Exception:
            pass

    try:
        residuals = model_residuals(entry)
        normality_tests(rows, residuals, alpha)
    except Exception:
        pass

    return {"tests": pd.DataFrame(rows).set_index("Test"), "first_stage": first_stage}


def volatility_diagnostics(
    entry: dict[str, Any],
    lag: int,
    alpha: float,
) -> dict[str, pd.DataFrame]:
    result = entry["result"]
    standardized = finite_series(result.std_resid)
    rows: list[dict[str, Any]] = []
    normality_tests(rows, standardized, alpha)

    usable_lag = min(max(1, lag), max(1, len(standardized) // 4))
    for label, values in [
        ("Standardized residuals", standardized),
        ("Squared standardized residuals", standardized**2),
    ]:
        try:
            lb = acorr_ljungbox(values, lags=[usable_lag], return_df=True).iloc[-1]
            add_test(
                rows,
                f"Ljung-Box on {label.lower()} ({usable_lag})",
                lb["lb_stat"],
                lb["lb_pvalue"],
                "No remaining serial dependence",
                alpha,
            )
        except Exception:
            pass

    try:
        lm, lm_p, f_stat, f_p = het_arch(standardized, nlags=usable_lag)
        add_test(rows, f"Remaining ARCH LM({usable_lag})", lm, lm_p, "No remaining ARCH effects", alpha)
        add_test(rows, f"Remaining ARCH F({usable_lag})", f_stat, f_p, "No remaining ARCH effects", alpha)
    except Exception:
        pass

    plot_data = pd.DataFrame(
        {
            "Standardized residual": standardized.reset_index(drop=True),
        }
    )
    try:
        volatility = finite_series(result.conditional_volatility)
        size = min(len(plot_data), len(volatility))
        plot_data = plot_data.iloc[:size].copy()
        plot_data["Conditional volatility"] = volatility.iloc[:size].reset_index(drop=True)
    except Exception:
        pass

    return {"tests": pd.DataFrame(rows).set_index("Test"), "plot_data": plot_data}


def run_model_diagnostics(
    entry: dict[str, Any],
    lag: int,
    alpha: float,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    family = entry["family"]
    if family == "linear":
        return linear_diagnostics(entry, lag, alpha)
    if family == "binary":
        return binary_diagnostics(entry, alpha, threshold)
    if family == "count":
        return count_diagnostics(entry, lag, alpha)
    if family in {"ardl", "arima"}:
        return time_series_diagnostics(entry, lag, alpha)
    if family in {"var", "vecm"}:
        return multivariate_diagnostics(entry, lag, alpha)
    if family == "panel":
        return panel_diagnostics(entry, lag, alpha)
    if family == "iv":
        return iv_diagnostics(entry, alpha)
    if family == "volatility":
        return volatility_diagnostics(entry, lag, alpha)
    raise ValueError(f"No diagnostic suite has been defined for model family {family!r}.")


def diagnostic_code(entry: dict[str, Any], lag: int, alpha: float, threshold: float) -> str:
    original = entry.get("code", "")
    family = entry["family"]
    snippets = {
        "linear": f"""
residuals = pd.Series(np.asarray(results.resid)).dropna()
print("Durbin-Watson:", durbin_watson(residuals))
print("Jarque-Bera:", jarque_bera(residuals))
print("Ljung-Box:", acorr_ljungbox(residuals, lags=[{lag}], return_df=True))
try:
    print("Breusch-Godfrey:", acorr_breusch_godfrey(results, nlags={lag}))
except Exception:
    pass
try:
    print("Breusch-Pagan:", het_breuschpagan(residuals, np.asarray(results.model.exog)))
    print("White:", het_white(residuals, np.asarray(results.model.exog)))
except Exception:
    pass
""",
        "binary": f"""
observed = pd.Series(np.asarray(results.model.endog))
probability = pd.Series(np.asarray(results.predict())).clip(1e-9, 1-1e-9)
classification = (probability >= {threshold}).astype(int)
print(pd.crosstab(observed, classification, rownames=["Observed"], colnames=["Predicted"]))
""",
        "count": f"""
observed = pd.Series(np.asarray(results.model.endog))
fitted = pd.Series(np.asarray(results.predict())).clip(lower=1e-9)
pearson_residual = (observed - fitted) / np.sqrt(fitted)
pearson_chi2 = float(np.sum(pearson_residual ** 2))
print("Pearson chi-square:", pearson_chi2)
print("Dispersion ratio:", pearson_chi2 / results.df_resid)
""",
        "ardl": f"""
residuals = pd.Series(np.asarray(results.resid)).dropna()
print("Jarque-Bera:", jarque_bera(residuals))
print("Ljung-Box:", acorr_ljungbox(residuals, lags=[{lag}], return_df=True))
print("ARCH LM:", het_arch(residuals, nlags={lag}))
""",
        "arima": f"""
residuals = pd.Series(np.asarray(results.resid)).dropna()
print("Jarque-Bera:", jarque_bera(residuals))
print("Ljung-Box:", acorr_ljungbox(residuals, lags=[{lag}], return_df=True))
print("ARCH LM:", het_arch(residuals, nlags={lag}))
""",
        "var": f"""
print(results.test_whiteness(nlags={lag}))
print(results.test_normality())
print("Stable:", results.is_stable())
""",
        "vecm": f"""
print(results.test_whiteness(nlags={lag}))
print(results.test_normality())
""",
        "panel": f"""
residuals = pd.Series(np.asarray(results.resids)).dropna()
print("Durbin-Watson:", durbin_watson(residuals))
print("Jarque-Bera:", jarque_bera(residuals))
""",
        "iv": """
print(results.first_stage)
try:
    print(results.wu_hausman())
except Exception:
    pass
try:
    print(results.sargan)
except Exception:
    pass
""",
        "volatility": f"""
standardized_residuals = pd.Series(np.asarray(results.std_resid)).dropna()
print("Jarque-Bera:", jarque_bera(standardized_residuals))
print("Ljung-Box:", acorr_ljungbox(standardized_residuals, lags=[{lag}], return_df=True))
print("Ljung-Box squared:", acorr_ljungbox(standardized_residuals ** 2, lags=[{lag}], return_df=True))
print("ARCH LM:", het_arch(standardized_residuals, nlags={lag}))
""",
    }
    return textwrap.dedent(
        f"""
# Rebuild the selected fitted model exactly as originally estimated.
{original}

# Model-specific post-estimation diagnostics.
diagnostic_alpha = {alpha!r}
{snippets[family]}
"""
    ).strip()


def unique_diagnostic_name(model_name: str) -> str:
    stamp = datetime.now().strftime("%H%M%S")
    candidate = f"{model_name}_Diagnostics_{stamp}"
    index = 2
    while candidate in st.session_state.results:
        candidate = f"{model_name}_Diagnostics_{stamp}_{index}"
        index += 1
    return candidate


def latest_diagnostic_name() -> str | None:
    stack = st.session_state.slot_outputs.get("diagnostics", [])
    if isinstance(stack, str):
        stack = [stack]
        st.session_state.slot_outputs["diagnostics"] = stack
    return stack[-1] if stack else None


def remove_diagnostic(name: str) -> None:
    st.session_state.code_blocks = [
        block for block in st.session_state.code_blocks if block.get("name") != name
    ]
    st.session_state.settings_log = [
        item for item in st.session_state.settings_log if item.get("name") != name
    ]
    st.session_state.history = [
        item
        for item in st.session_state.history
        if item.get("action") != f"Completed {name}"
    ]
    for key in list(st.session_state.results):
        if key == name or key.startswith(name + "_"):
            st.session_state.results.pop(key, None)
    st.session_state.diagnostic_runs.pop(name, None)

    stack = st.session_state.slot_outputs.get("diagnostics", [])
    if isinstance(stack, str):
        stack = [stack]
    st.session_state.slot_outputs["diagnostics"] = [item for item in stack if item != name]
    if not st.session_state.slot_outputs["diagnostics"]:
        st.session_state.slot_outputs.pop("diagnostics", None)

    state = st.session_state.diagnostic_page_state
    if state and state.get("name") == name:
        st.session_state.diagnostic_page_state = None


def register_diagnostics(
    model_name: str,
    entry: dict[str, Any],
    outputs: dict[str, pd.DataFrame],
    lag: int,
    alpha: float,
    threshold: float,
) -> str:
    name = unique_diagnostic_name(model_name)
    tests = outputs.get("tests", pd.DataFrame())
    st.session_state.results[name] = tests
    for key, frame in outputs.items():
        if key == "tests" or key == "plot_data" or frame is None or frame.empty:
            continue
        st.session_state.results[f"{name}_{key}"] = frame

    code = diagnostic_code(entry, lag, alpha, threshold)
    st.session_state.code_blocks.append({"name": name, "code": code})
    st.session_state.settings_log.append(
        {
            "name": name,
            "time": datetime.now().isoformat(timespec="seconds"),
            "settings": {
                "parent_model": model_name,
                "model_family": entry["family"],
                "estimator": entry["estimator"],
                "lag": lag,
                "significance": alpha,
                "classification_threshold": threshold if entry["family"] == "binary" else None,
            },
            "summary": f"Post-estimation diagnostics for {model_name}",
        }
    )
    st.session_state.history.append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "action": f"Completed {name}",
        }
    )
    stack = st.session_state.slot_outputs.setdefault("diagnostics", [])
    if isinstance(stack, str):
        stack = [stack]
        st.session_state.slot_outputs["diagnostics"] = stack
    stack.append(name)
    st.session_state.diagnostic_runs[name] = model_name
    st.session_state.diagnostic_page_state = {
        "name": name,
        "parent_model": model_name,
        "family": entry["family"],
        "estimator": entry["estimator"],
        "outputs": outputs,
    }
    return name


def render_plots(state: dict[str, Any]) -> None:
    outputs = state["outputs"]
    family = state["family"]
    plot_data = outputs.get("plot_data")
    residuals = outputs.get("residuals")

    choices: list[str] = []
    if isinstance(plot_data, pd.DataFrame) and not plot_data.empty:
        if "Residual" in plot_data:
            choices.extend(["Residual sequence", "Residual histogram", "Normal Q-Q plot"])
        if {"Fitted", "Residual"}.issubset(plot_data.columns):
            choices.append("Residuals against fitted")
        if {"Actual", "Fitted"}.issubset(plot_data.columns):
            choices.append("Actual against fitted")
        if "Probability" in plot_data:
            choices.extend(["Predicted-probability histogram", "Observed against probability"])
        if "Pearson residual" in plot_data:
            choices.append("Pearson residual sequence")
        if "Conditional volatility" in plot_data:
            choices.append("Conditional volatility")
        if "Standardized residual" in plot_data:
            choices.extend(["Standardized residual sequence", "Squared standardized residuals"])

    if isinstance(residuals, pd.DataFrame) and not residuals.empty:
        choices.append("Multivariate residual sequences")

    if not choices:
        return

    st.subheader("Diagnostic plots")
    choice = st.selectbox("Plot", choices, key="post_estimation_diagnostic_plot")
    fig, ax = plt.subplots()

    if choice == "Residual sequence":
        ax.plot(plot_data["Residual"].to_numpy())
        ax.axhline(0)
        ax.set_ylabel("Residual")
    elif choice == "Residual histogram":
        ax.hist(plot_data["Residual"].dropna(), bins="auto")
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
    elif choice == "Normal Q-Q plot":
        sm.qqplot(plot_data["Residual"].dropna(), line="45", ax=ax)
    elif choice == "Residuals against fitted":
        ax.scatter(plot_data["Fitted"], plot_data["Residual"])
        ax.axhline(0)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
    elif choice == "Actual against fitted":
        ax.scatter(plot_data["Actual"], plot_data["Fitted"])
        lower = min(plot_data["Actual"].min(), plot_data["Fitted"].min())
        upper = max(plot_data["Actual"].max(), plot_data["Fitted"].max())
        ax.plot([lower, upper], [lower, upper])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Fitted")
    elif choice == "Predicted-probability histogram":
        ax.hist(plot_data["Probability"], bins="auto")
        ax.set_xlabel("Predicted probability")
    elif choice == "Observed against probability":
        ax.scatter(plot_data["Probability"], plot_data["Observed"], alpha=0.5)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed outcome")
    elif choice == "Pearson residual sequence":
        ax.plot(plot_data["Pearson residual"].to_numpy())
        ax.axhline(0)
        ax.set_ylabel("Pearson residual")
    elif choice == "Conditional volatility":
        ax.plot(plot_data["Conditional volatility"].to_numpy())
        ax.set_ylabel("Conditional volatility")
    elif choice == "Standardized residual sequence":
        ax.plot(plot_data["Standardized residual"].to_numpy())
        ax.axhline(0)
        ax.set_ylabel("Standardized residual")
    elif choice == "Squared standardized residuals":
        ax.plot(plot_data["Standardized residual"].to_numpy() ** 2)
        ax.set_ylabel("Squared standardized residual")
    elif choice == "Multivariate residual sequences":
        residuals.plot(ax=ax)
        ax.axhline(0)

    ax.set_title(choice)
    ax.set_xlabel(ax.get_xlabel() or "Observation")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


init_state()
apply_theme()

with st.sidebar:
    st.header("Diagnostics")
    st.toggle(
        "Dark mode",
        key="dark_mode",
        help="The theme setting is shared with the main application.",
    )
    if st.session_state.source_filename:
        st.caption(st.session_state.source_filename)

st.title("🧪 Post-estimation diagnostics")
st.caption(
    "Select a model that was already estimated in Econometrics Studio. "
    "Every test below uses that exact fitted result, sample, residuals, weights, "
    "lags, instruments and estimator settings."
)

registry: dict[str, dict[str, Any]] = st.session_state.model_registry
if not registry:
    st.info(
        "No fitted models are available in this session. Estimate a regression, "
        "time-series, panel, IV or volatility model in the main application first."
    )
    st.stop()

model_names = list(registry)
selected_name = st.selectbox(
    "Previously estimated model",
    model_names,
    format_func=lambda name: (
        f"{name} — {registry[name]['estimator']} "
        f"({registry[name]['family']})"
    ),
    key="selected_post_estimation_model",
)
entry = registry[selected_name]

c1, c2, c3 = st.columns(3)
c1.metric("Estimator", entry["estimator"])
c2.metric("Model family", entry["family"].upper())
sample = entry.get("sample")
try:
    observations = len(sample)
except Exception:
    observations = getattr(entry["result"], "nobs", "—")
c3.metric("Estimation observations", observations)

with st.expander("Exact estimation settings", expanded=False):
    st.json(entry.get("metadata", {}))

lag = int(st.number_input("Diagnostic lag", 1, 100, 1))
alpha = float(
    st.selectbox(
        "Significance level",
        [0.01, 0.05, 0.10],
        index=1,
        format_func=lambda value: f"{value:.0%}",
    )
)
threshold = 0.5
if entry["family"] == "binary":
    threshold = float(st.slider("Classification threshold", 0.05, 0.95, 0.50, 0.05))

run_col, clear_col = st.columns(2)
run_clicked = run_col.button(
    "Run diagnostics for selected fitted model",
    type="primary",
    use_container_width=True,
)
latest = latest_diagnostic_name()
clear_clicked = clear_col.button(
    "Clear latest diagnostic run",
    type="secondary",
    use_container_width=True,
    disabled=latest is None,
)

if clear_clicked and latest:
    remove_diagnostic(latest)
    st.success(f"Removed {latest!r} from the page and the complete export.")
    st.rerun()

if run_clicked:
    try:
        outputs = run_model_diagnostics(entry, lag, alpha, threshold)
        name = register_diagnostics(selected_name, entry, outputs, lag, alpha, threshold)
        st.success(f"Completed {name}. The results are linked to {selected_name!r}.")
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")

state = st.session_state.diagnostic_page_state
if state:
    if state["parent_model"] not in registry:
        st.session_state.diagnostic_page_state = None
    else:
        st.caption(
            f"Displayed run: `{state['name']}` · Parent fitted model: "
            f"`{state['parent_model']}` · Estimator: **{state['estimator']}**"
        )
        outputs = state["outputs"]
        for title, key in [
            ("Diagnostic-test results", "tests"),
            ("Classification metrics", "classification"),
            ("Classification table", "confusion"),
            ("Grouped probability fit", "grouped_fit"),
            ("Count and zero-frequency checks", "zero_counts"),
            ("Variance inflation factors", "vif"),
            ("Influential observations", "influence"),
            ("VAR/VECM roots", "roots"),
            ("Panel entity residual variances", "entity_variance"),
            ("IV first-stage diagnostics", "first_stage"),
        ]:
            frame = outputs.get(key)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                st.subheader(title)
                show_frame(frame)

        render_plots(state)

st.divider()
st.caption(
    "The available tests change automatically with the selected estimator. "
    "A diagnostic test is reported only when it is meaningful and supported by "
    "the exact fitted result."
)
