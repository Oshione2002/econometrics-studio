
from __future__ import annotations

import math
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss

try:
    from arch.unitroot import PhillipsPerron
    PP_AVAILABLE = True
except Exception:
    PP_AVAILABLE = False

try:
    from westerlund_test import WesterlundTest
    WESTERLUND_AVAILABLE = True
    WESTERLUND_IMPORT_ERROR = ""
except Exception as exc:
    WESTERLUND_AVAILABLE = False
    WESTERLUND_IMPORT_ERROR = str(exc)


GENERAL_METHODS = [
    "Data and panel structure",
    "Descriptive statistics",
    "Distribution and normality",
    "Correlation and covariance",
    "Variance inflation factors",
    "Country/entity and time trend graphs",
]

TIME_SERIES_METHODS = [
    "Time-series unit-root tests",
    "Serial-correlation and ARCH screening",
    "ACF and PACF graphs",
]

PANEL_METHODS = [
    "Within-between variation",
    "Cross-sectional dependence tests",
    "Pesaran-Yamagata slope heterogeneity test",
    "CIPS panel unit-root test",
    "Fisher panel unit-root tests",
    "Westerlund panel cointegration test",
    "Residual-based panel cointegration screening",
]


def _clean_numeric(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = data.loc[:, columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def descriptive_statistics_table(data: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    frame = _clean_numeric(data, variables)
    rows: dict[str, dict[str, float]] = {}
    for variable in variables:
        series = frame[variable].dropna()
        if series.empty:
            continue
        jb = scipy_stats.jarque_bera(series)
        rows[variable] = {
            "Observations": float(series.count()),
            "Mean": float(series.mean()),
            "Median": float(series.median()),
            "Maximum": float(series.max()),
            "Minimum": float(series.min()),
            "Std. Deviation": float(series.std(ddof=1)),
            "Variance": float(series.var(ddof=1)),
            "Coefficient of variation": float(series.std(ddof=1) / abs(series.mean()))
            if series.mean() != 0
            else np.nan,
            "Skewness": float(series.skew()),
            "Kurtosis": float(series.kurtosis() + 3),
            "Jarque-Bera": float(jb.statistic),
            "JB P-value": float(jb.pvalue),
        }
    return pd.DataFrame(rows).T


def distribution_diagnostics_table(
    data: pd.DataFrame, variables: list[str], alpha: float = 0.05
) -> pd.DataFrame:
    frame = _clean_numeric(data, variables)
    rows = []
    for variable in variables:
        series = frame[variable].dropna()
        if len(series) < 3:
            continue
        jb = scipy_stats.jarque_bera(series)
        if len(series) >= 8:
            dag = scipy_stats.normaltest(series)
            dag_stat, dag_p = float(dag.statistic), float(dag.pvalue)
        else:
            dag_stat, dag_p = np.nan, np.nan
        if 3 <= len(series) <= 5000:
            shapiro = scipy_stats.shapiro(series)
            sh_stat, sh_p = float(shapiro.statistic), float(shapiro.pvalue)
        else:
            sh_stat, sh_p = np.nan, np.nan
        rows.append(
            {
                "Variable": variable,
                "Observations": len(series),
                "Skewness": float(series.skew()),
                "Excess kurtosis": float(series.kurtosis()),
                "Jarque-Bera": float(jb.statistic),
                "JB P-value": float(jb.pvalue),
                "D'Agostino K²": dag_stat,
                "D'Agostino P-value": dag_p,
                "Shapiro-Wilk": sh_stat,
                "Shapiro P-value": sh_p,
                f"Normal at {alpha:.0%}": "Yes" if float(jb.pvalue) >= alpha else "No",
            }
        )
    return pd.DataFrame(rows).set_index("Variable") if rows else pd.DataFrame()


def correlation_covariance_tables(
    data: pd.DataFrame, variables: list[str], method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _clean_numeric(data, variables)
    return frame.corr(method=method), frame.cov()


def vif_condition_table(data: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _clean_numeric(data, variables).dropna()
    if len(frame) <= len(variables) + 1:
        raise ValueError("The complete-case sample is too small for VIF calculation.")
    design = sm.add_constant(frame, has_constant="add")
    rows = []
    for index, name in enumerate(design.columns):
        if name == "const":
            continue
        value = float(variance_inflation_factor(design.to_numpy(), index))
        rows.append(
            {
                "Variable": name,
                "VIF": value,
                "Tolerance": 1.0 / value if value not in {0.0, np.inf} else 0.0,
                "Flag": "High" if value >= 10 else ("Moderate" if value >= 5 else "Low"),
            }
        )
    scaled = (frame - frame.mean()) / frame.std(ddof=1).replace(0, np.nan)
    scaled = scaled.dropna(axis=1)
    if scaled.shape[1] >= 2:
        _, singular, vt = np.linalg.svd(scaled.to_numpy(), full_matrices=False)
        eigenvalues = singular**2
        maximum = float(eigenvalues.max())
        condition = np.sqrt(maximum / eigenvalues)
        condition_table = pd.DataFrame(
            {
                "Dimension": np.arange(1, len(condition) + 1),
                "Eigenvalue": eigenvalues,
                "Condition index": condition,
            }
        ).set_index("Dimension")
    else:
        condition_table = pd.DataFrame()
    return pd.DataFrame(rows).set_index("Variable"), condition_table


def panel_structure_table(data: pd.DataFrame, entity: str, time: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = data[[entity, time]].dropna().copy()
    duplicate_count = int(panel.duplicated([entity, time]).sum())
    counts = panel.groupby(entity, observed=True)[time].nunique().sort_values()
    periods = panel[time].nunique()
    balanced = bool(len(counts) > 0 and counts.nunique() == 1 and counts.iloc[0] == periods)
    overview = pd.DataFrame(
        {
            "Value": [
                len(data),
                panel[entity].nunique(),
                periods,
                int(counts.min()) if len(counts) else 0,
                float(counts.mean()) if len(counts) else np.nan,
                int(counts.max()) if len(counts) else 0,
                balanced,
                duplicate_count,
                int(data[[entity, time]].isna().any(axis=1).sum()),
            ]
        },
        index=[
            "Rows",
            "Cross-sectional units",
            "Time periods",
            "Minimum periods per unit",
            "Average periods per unit",
            "Maximum periods per unit",
            "Balanced panel",
            "Duplicate entity-time rows",
            "Rows missing entity or time",
        ],
    )
    entity_counts = counts.to_frame("Observed periods")
    entity_counts["Share of full time dimension (%)"] = (
        entity_counts["Observed periods"] / periods * 100 if periods else np.nan
    )
    return overview, entity_counts


def missingness_table(data: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Missing": data.isna().sum(),
            "Percent": data.isna().mean() * 100,
            "Non-missing": data.notna().sum(),
            "Unique": data.nunique(dropna=True),
            "Data type": data.dtypes.astype(str),
        }
    )
    return table.sort_values(["Missing", "Percent"], ascending=False)


def within_between_table(
    data: pd.DataFrame, entity: str, variables: list[str]
) -> pd.DataFrame:
    frame = data[[entity] + variables].copy()
    for variable in variables:
        frame[variable] = pd.to_numeric(frame[variable], errors="coerce")
    rows = []
    for variable in variables:
        valid = frame[[entity, variable]].dropna()
        if valid.empty:
            continue
        values = valid[variable]
        means = valid.groupby(entity, observed=True)[variable].mean()
        within = values - valid.groupby(entity, observed=True)[variable].transform("mean")
        total_var = float(values.var(ddof=1))
        between_var = float(means.var(ddof=1)) if len(means) > 1 else 0.0
        within_var = float(within.var(ddof=1))
        rows.append(
            {
                "Variable": variable,
                "Observations": len(values),
                "Entities": valid[entity].nunique(),
                "Overall mean": float(values.mean()),
                "Overall std. dev.": math.sqrt(total_var) if total_var >= 0 else np.nan,
                "Between std. dev.": math.sqrt(between_var) if between_var >= 0 else np.nan,
                "Within std. dev.": math.sqrt(within_var) if within_var >= 0 else np.nan,
                "Within variance share (%)": 100 * within_var / total_var
                if total_var > 0
                else np.nan,
                "Between variance share (%)": 100 * between_var / total_var
                if total_var > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("Variable") if rows else pd.DataFrame()


def _pairwise_panel_correlations(
    data: pd.DataFrame, entity: str, time: str, variable: str
) -> tuple[list[float], list[int]]:
    pivot = (
        data[[entity, time, variable]]
        .dropna()
        .pivot_table(index=time, columns=entity, values=variable, aggfunc="mean")
        .sort_index()
    )
    correlations: list[float] = []
    overlaps: list[int] = []
    columns = list(pivot.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            pair = pivot[[columns[i], columns[j]]].dropna()
            if len(pair) < 3:
                continue
            rho = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if np.isfinite(rho):
                correlations.append(float(rho))
                overlaps.append(len(pair))
    return correlations, overlaps


def cross_section_dependence_table(
    data: pd.DataFrame, entity: str, time: str, variables: list[str], alpha: float = 0.05
) -> pd.DataFrame:
    rows = []
    n_entities = int(data[entity].nunique(dropna=True))
    pair_df = n_entities * (n_entities - 1) / 2
    if n_entities < 2:
        raise ValueError("At least two cross-sectional units are required.")
    for variable in variables:
        correlations, overlaps = _pairwise_panel_correlations(data, entity, time, variable)
        if not correlations:
            continue
        rho = np.asarray(correlations, dtype=float)
        tij = np.asarray(overlaps, dtype=float)
        pairs = len(rho)
        cd = math.sqrt(2.0 / (n_entities * (n_entities - 1))) * float(
            np.sum(np.sqrt(tij) * rho)
        )
        cd_p = float(2 * scipy_stats.norm.sf(abs(cd)))
        bp_lm = float(np.sum(tij * rho**2))
        bp_p = float(scipy_stats.chi2.sf(bp_lm, df=max(1, pairs)))
        scaled_lm = float(
            np.sum(tij * rho**2 - 1.0) / math.sqrt(max(1.0, 2.0 * pairs))
        )
        scaled_p = float(2 * scipy_stats.norm.sf(abs(scaled_lm)))
        rows.extend(
            [
                {
                    "Variable": variable,
                    "Test": "Pesaran CD",
                    "Statistic": cd,
                    "P-value": cd_p,
                    "Pairs": pairs,
                    "Average pairwise correlation": float(rho.mean()),
                    "Average absolute correlation": float(np.abs(rho).mean()),
                    "Null hypothesis": "Cross-sectional independence",
                    f"Decision at {alpha:.0%}": "Reject" if cd_p < alpha else "Do not reject",
                },
                {
                    "Variable": variable,
                    "Test": "Breusch-Pagan LM",
                    "Statistic": bp_lm,
                    "P-value": bp_p,
                    "Pairs": pairs,
                    "Average pairwise correlation": float(rho.mean()),
                    "Average absolute correlation": float(np.abs(rho).mean()),
                    "Null hypothesis": "Cross-sectional independence",
                    f"Decision at {alpha:.0%}": "Reject" if bp_p < alpha else "Do not reject",
                },
                {
                    "Variable": variable,
                    "Test": "Pesaran scaled LM",
                    "Statistic": scaled_lm,
                    "P-value": scaled_p,
                    "Pairs": pairs,
                    "Average pairwise correlation": float(rho.mean()),
                    "Average absolute correlation": float(np.abs(rho).mean()),
                    "Null hypothesis": "Cross-sectional independence",
                    f"Decision at {alpha:.0%}": "Reject" if scaled_p < alpha else "Do not reject",
                },
            ]
        )
    return (
        pd.DataFrame(rows).set_index(["Variable", "Test"])
        if rows
        else pd.DataFrame()
    )


def pesaran_yamagata_test(
    data: pd.DataFrame,
    entity: str,
    time: str,
    dependent: str,
    regressors: list[str],
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not regressors:
        raise ValueError("Select at least one slope variable.")
    columns = [entity, time, dependent] + regressors
    frame = _clean_numeric(data, [dependent] + regressors)
    frame[entity] = data[entity]
    frame[time] = data[time]
    frame = frame.dropna(subset=columns).sort_values([entity, time])
    k = len(regressors)
    unit_data: list[dict[str, Any]] = []
    xtx_sum = np.zeros((k, k), dtype=float)
    xty_sum = np.zeros(k, dtype=float)

    for unit, group in frame.groupby(entity, observed=True):
        if len(group) <= k + 2:
            continue
        x = group[regressors].to_numpy(dtype=float)
        y = group[dependent].to_numpy(dtype=float)
        x_demeaned = x - x.mean(axis=0)
        y_demeaned = y - y.mean()
        if np.linalg.matrix_rank(x_demeaned) < k:
            continue
        xtx = x_demeaned.T @ x_demeaned
        beta = np.linalg.solve(xtx, x_demeaned.T @ y_demeaned)
        xtx_sum += xtx
        xty_sum += x_demeaned.T @ y_demeaned
        unit_data.append(
            {"unit": unit, "n": len(group), "x": x_demeaned, "y": y_demeaned, "xtx": xtx, "beta": beta}
        )

    n_units = len(unit_data)
    if n_units < 3:
        raise ValueError("At least three estimable cross-sectional units are required.")
    if np.linalg.matrix_rank(xtx_sum) < k:
        raise ValueError("The pooled slope design matrix is singular.")
    beta_fe = np.linalg.solve(xtx_sum, xty_sum)
    swamy = 0.0
    detail_rows = []
    for item in unit_data:
        residual_fe = item["y"] - item["x"] @ beta_fe
        sigma2 = float(residual_fe @ residual_fe / max(1, item["n"] - 1))
        diff = item["beta"] - beta_fe
        d_i = float(diff.T @ item["xtx"] @ diff / max(sigma2, 1e-12))
        swamy += d_i
        row = {"Entity": item["unit"], "Periods": item["n"], "Weighted dispersion": d_i}
        for idx, variable in enumerate(regressors):
            row[f"Slope: {variable}"] = float(item["beta"][idx])
        detail_rows.append(row)

    t_bar = float(np.mean([item["n"] for item in unit_data]))
    delta = math.sqrt(n_units) * (swamy / n_units - k) / math.sqrt(2 * k)
    adjusted_variance = 2 * k * max(t_bar - k - 1, 1e-9) / max(t_bar + 1, 1e-9)
    delta_adj = math.sqrt(n_units) * (swamy / n_units - k) / math.sqrt(adjusted_variance)
    delta_p = float(2 * scipy_stats.norm.sf(abs(delta)))
    delta_adj_p = float(2 * scipy_stats.norm.sf(abs(delta_adj)))
    summary = pd.DataFrame(
        [
            {
                "Test": "Pesaran-Yamagata Delta",
                "Statistic": delta,
                "P-value": delta_p,
                "Swamy dispersion": swamy,
                "Cross-sectional units": n_units,
                "Average periods": t_bar,
                "Slope coefficients tested": k,
                "Null hypothesis": "Slope homogeneity",
                f"Decision at {alpha:.0%}": "Reject" if delta_p < alpha else "Do not reject",
            },
            {
                "Test": "Pesaran-Yamagata adjusted Delta",
                "Statistic": delta_adj,
                "P-value": delta_adj_p,
                "Swamy dispersion": swamy,
                "Cross-sectional units": n_units,
                "Average periods": t_bar,
                "Slope coefficients tested": k,
                "Null hypothesis": "Slope homogeneity",
                f"Decision at {alpha:.0%}": "Reject" if delta_adj_p < alpha else "Do not reject",
            },
        ]
    ).set_index("Test")
    details = pd.DataFrame(detail_rows).set_index("Entity")
    return summary, details


def _cadf_tstat(
    y: pd.Series,
    y_bar: pd.Series,
    lags: int,
    trend: str,
) -> float:
    aligned = pd.concat([y.rename("y"), y_bar.rename("ybar")], axis=1).dropna()
    dy = aligned["y"].diff()
    dybar = aligned["ybar"].diff()
    design = pd.DataFrame(
        {
            "dy": dy,
            "y_lag": aligned["y"].shift(1),
            "ybar_lag": aligned["ybar"].shift(1),
            "dybar": dybar,
        },
        index=aligned.index,
    )
    for lag in range(1, lags + 1):
        design[f"dy_lag_{lag}"] = dy.shift(lag)
        design[f"dybar_lag_{lag}"] = dybar.shift(lag)
    if trend in {"ct", "ctt"}:
        design["trend"] = np.arange(1, len(design) + 1, dtype=float)
    if trend == "ctt":
        design["trend_sq"] = design["trend"] ** 2
    design = design.dropna()
    if len(design) <= design.shape[1] + 2:
        return np.nan
    x_columns = [column for column in design.columns if column != "dy"]
    x = design[x_columns]
    if trend != "n":
        x = sm.add_constant(x, has_constant="add")
    result = sm.OLS(design["dy"], x).fit()
    return float(result.tvalues["y_lag"])


def cips_panel_unit_root_test(
    data: pd.DataFrame,
    entity: str,
    time: str,
    variable: str,
    lags: int = 1,
    trend: str = "c",
    simulations: int = 99,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = (
        data[[entity, time, variable]]
        .dropna()
        .pivot_table(index=time, columns=entity, values=variable, aggfunc="mean")
        .sort_index()
    )
    if pivot.shape[1] < 3 or pivot.shape[0] < max(12, lags + 8):
        raise ValueError("CIPS requires at least three units and a sufficient time dimension.")
    ybar = pivot.mean(axis=1, skipna=True)
    rows = []
    tstats = []
    for unit in pivot.columns:
        stat = _cadf_tstat(pivot[unit], ybar, lags, trend)
        if np.isfinite(stat):
            rows.append({"Entity": unit, "CADF t-statistic": stat, "Observations": int(pivot[unit].count())})
            tstats.append(stat)
    if len(tstats) < 3:
        raise ValueError("Too few entity-specific CADF regressions could be estimated.")
    cips = float(np.mean(tstats))

    rng = np.random.default_rng(seed)
    simulated: list[float] = []
    t_count, n_count = pivot.shape
    for _ in range(int(simulations)):
        innovations = rng.standard_normal((t_count, n_count))
        simulated_panel = pd.DataFrame(np.cumsum(innovations, axis=0), index=pivot.index)
        simulated_bar = simulated_panel.mean(axis=1)
        stats = [
            _cadf_tstat(simulated_panel[column], simulated_bar, lags, trend)
            for column in simulated_panel.columns
        ]
        stats = [value for value in stats if np.isfinite(value)]
        if stats:
            simulated.append(float(np.mean(stats)))
    if not simulated:
        raise RuntimeError("The CIPS null simulation failed.")
    null = np.asarray(simulated)
    pvalue = float((1 + np.sum(null <= cips)) / (1 + len(null)))
    critical_1, critical_5, critical_10 = np.quantile(null, [0.01, 0.05, 0.10])
    summary = pd.DataFrame(
        [
            {
                "Variable": variable,
                "CIPS statistic": cips,
                "Simulation P-value": pvalue,
                "1% simulated critical": critical_1,
                "5% simulated critical": critical_5,
                "10% simulated critical": critical_10,
                "Entities": len(tstats),
                "Time periods": t_count,
                "Lags": lags,
                "Deterministic terms": trend,
                "Null hypothesis": "All panel units contain a unit root",
                f"Decision at {alpha:.0%}": "Reject" if pvalue < alpha else "Do not reject",
            }
        ]
    ).set_index("Variable")
    return summary, pd.DataFrame(rows).set_index("Entity")


def fisher_panel_unit_root_test(
    data: pd.DataFrame,
    entity: str,
    time: str,
    variables: list[str],
    method: str = "ADF",
    lags: int = 1,
    trend: str = "c",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    individual_rows = []
    for variable in variables:
        pvalues = []
        for unit, group in data[[entity, time, variable]].dropna().groupby(entity, observed=True):
            series = pd.to_numeric(group.sort_values(time)[variable], errors="coerce").dropna()
            if len(series) < max(10, lags + 6):
                continue
            try:
                if method == "ADF":
                    result = adfuller(series, maxlag=lags, regression=trend, autolag=None)
                    statistic, pvalue = float(result[0]), float(result[1])
                elif method == "Phillips-Perron":
                    if not PP_AVAILABLE:
                        raise ImportError("Phillips-Perron requires the arch package.")
                    pp_trend = "n" if trend == "n" else ("ct" if trend in {"ct", "ctt"} else "c")
                    result = PhillipsPerron(series, lags=lags, trend=pp_trend)
                    statistic, pvalue = float(result.stat), float(result.pvalue)
                else:
                    raise ValueError("Unknown Fisher panel unit-root method.")
                pvalues.append(max(pvalue, np.finfo(float).tiny))
                individual_rows.append(
                    {
                        "Variable": variable,
                        "Entity": unit,
                        "Method": method,
                        "Statistic": statistic,
                        "P-value": pvalue,
                        "Observations": len(series),
                    }
                )
            except Exception:
                continue
        if not pvalues:
            continue
        fisher_stat = float(-2 * np.sum(np.log(pvalues)))
        fisher_p = float(scipy_stats.chi2.sf(fisher_stat, 2 * len(pvalues)))
        inv_normal = float(np.sum(scipy_stats.norm.ppf(pvalues)) / math.sqrt(len(pvalues)))
        inv_p = float(scipy_stats.norm.cdf(inv_normal))
        summary_rows.extend(
            [
                {
                    "Variable": variable,
                    "Combined test": f"Fisher {method} chi-square",
                    "Statistic": fisher_stat,
                    "P-value": fisher_p,
                    "Entities": len(pvalues),
                    "Null hypothesis": "All panel units contain a unit root",
                    f"Decision at {alpha:.0%}": "Reject" if fisher_p < alpha else "Do not reject",
                },
                {
                    "Variable": variable,
                    "Combined test": f"Inverse-normal {method}",
                    "Statistic": inv_normal,
                    "P-value": inv_p,
                    "Entities": len(pvalues),
                    "Null hypothesis": "All panel units contain a unit root",
                    f"Decision at {alpha:.0%}": "Reject" if inv_p < alpha else "Do not reject",
                },
            ]
        )
    summary = (
        pd.DataFrame(summary_rows).set_index(["Variable", "Combined test"])
        if summary_rows
        else pd.DataFrame()
    )
    details = (
        pd.DataFrame(individual_rows).set_index(["Variable", "Entity"])
        if individual_rows
        else pd.DataFrame()
    )
    return summary, details


def time_series_unit_root_table(
    data: pd.DataFrame,
    variables: list[str],
    method: str,
    lags: int,
    trend: str,
    alpha: float,
) -> pd.DataFrame:
    rows = []
    for variable in variables:
        series = pd.to_numeric(data[variable], errors="coerce").dropna()
        if len(series) < max(10, lags + 6):
            continue
        if method == "ADF":
            result = adfuller(series, maxlag=lags, regression=trend, autolag=None)
            stat, pvalue, used_lag = result[0], result[1], result[2]
            null = "Unit root"
        elif method == "KPSS":
            kpss_trend = "ct" if trend in {"ct", "ctt"} else "c"
            result = kpss(series, regression=kpss_trend, nlags=lags)
            stat, pvalue, used_lag = result[0], result[1], result[2]
            null = "Stationarity"
        elif method == "Phillips-Perron":
            if not PP_AVAILABLE:
                raise ImportError("Phillips-Perron requires the arch package.")
            pp_trend = "n" if trend == "n" else ("ct" if trend in {"ct", "ctt"} else "c")
            result = PhillipsPerron(series, lags=lags, trend=pp_trend)
            stat, pvalue, used_lag = result.stat, result.pvalue, result.lags
            null = "Unit root"
        else:
            raise ValueError("Unknown unit-root method.")
        reject = pvalue < alpha
        rows.append(
            {
                "Variable": variable,
                "Method": method,
                "Statistic": float(stat),
                "P-value": float(pvalue),
                "Lags": int(used_lag),
                "Observations": len(series),
                "Null hypothesis": null,
                f"Decision at {alpha:.0%}": "Reject" if reject else "Do not reject",
            }
        )
    return pd.DataFrame(rows).set_index(["Variable", "Method"]) if rows else pd.DataFrame()


def serial_arch_table(
    data: pd.DataFrame, variables: list[str], lag: int, alpha: float
) -> pd.DataFrame:
    rows = []
    for variable in variables:
        series = pd.to_numeric(data[variable], errors="coerce").dropna()
        if len(series) <= lag + 5:
            continue
        lb = acorr_ljungbox(series, lags=[lag], return_df=True).iloc[-1]
        arch = het_arch(series - series.mean(), nlags=lag)
        rows.extend(
            [
                {
                    "Variable": variable,
                    "Test": f"Ljung-Box Q({lag})",
                    "Statistic": float(lb["lb_stat"]),
                    "P-value": float(lb["lb_pvalue"]),
                    "Null hypothesis": "No serial correlation through selected lag",
                    f"Decision at {alpha:.0%}": "Reject" if lb["lb_pvalue"] < alpha else "Do not reject",
                },
                {
                    "Variable": variable,
                    "Test": f"ARCH LM({lag})",
                    "Statistic": float(arch[0]),
                    "P-value": float(arch[1]),
                    "Null hypothesis": "No ARCH effects through selected lag",
                    f"Decision at {alpha:.0%}": "Reject" if arch[1] < alpha else "Do not reject",
                },
            ]
        )
    return pd.DataFrame(rows).set_index(["Variable", "Test"]) if rows else pd.DataFrame()


def residual_panel_cointegration_test(
    data: pd.DataFrame,
    entity: str,
    time: str,
    dependent: str,
    regressors: list[str],
    lags: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not regressors:
        raise ValueError("Select at least one long-run regressor.")
    pvalues = []
    details = []
    for unit, group in data[[entity, time, dependent] + regressors].dropna().groupby(entity, observed=True):
        group = group.sort_values(time)
        if len(group) <= len(regressors) + lags + 6:
            continue
        y = pd.to_numeric(group[dependent], errors="coerce")
        x = _clean_numeric(group, regressors)
        model = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()
        residual = model.resid
        try:
            adf = adfuller(residual, maxlag=lags, regression="n", autolag=None)
        except Exception:
            continue
        pvalue = float(adf[1])
        pvalues.append(max(pvalue, np.finfo(float).tiny))
        details.append(
            {
                "Entity": unit,
                "Residual ADF statistic": float(adf[0]),
                "P-value": pvalue,
                "Observations": len(group),
            }
        )
    if not pvalues:
        raise ValueError("No entity-specific cointegrating regressions could be estimated.")
    fisher = float(-2 * np.sum(np.log(pvalues)))
    fisher_p = float(scipy_stats.chi2.sf(fisher, 2 * len(pvalues)))
    summary = pd.DataFrame(
        [
            {
                "Test": "Fisher residual ADF panel cointegration screen",
                "Statistic": fisher,
                "P-value": fisher_p,
                "Entities": len(pvalues),
                "Null hypothesis": "No cointegration in all cross-sectional units",
                f"Decision at {alpha:.0%}": "Reject" if fisher_p < alpha else "Do not reject",
            }
        ]
    ).set_index("Test")
    return summary, pd.DataFrame(details).set_index("Entity")


def run_westerlund_test(
    data: pd.DataFrame,
    entity: str,
    time: str,
    dependent: str,
    regressors: list[str],
    lags: int,
    leads: int,
    constant: bool,
    trend: bool,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, Any]:
    if not WESTERLUND_AVAILABLE:
        raise ImportError(
            "The optional Westerlund package is unavailable. "
            f"Deployment message: {WESTERLUND_IMPORT_ERROR}"
        )
    if not regressors:
        raise ValueError("Select at least one long-run regressor.")
    test = WesterlundTest(
        data=data[[entity, time, dependent] + regressors].dropna().copy(),
        y_var=dependent,
        x_vars=regressors,
        id_var=entity,
        time_var=time,
        lags=lags,
        leads=leads,
        constant=constant,
        trend=trend,
        bootstrap=int(bootstrap),
        seed=int(seed),
    )
    result = test.run()
    raw = getattr(result, "test_stats", result)
    if isinstance(raw, pd.DataFrame):
        table = raw.copy()
    elif isinstance(raw, dict):
        table = pd.DataFrame([raw]).T
        table.columns = ["Value"]
    else:
        try:
            table = pd.DataFrame(raw)
        except Exception:
            table = pd.DataFrame({"Result": [str(raw)]})
    return table, result


def _store_and_show(
    *,
    slot: str,
    prefix: str,
    result: Any,
    code: str,
    settings: dict[str, Any],
    register_output: Callable[..., None],
    make_unique_name: Callable[[str], str],
) -> str:
    name = make_unique_name(prefix)
    register_output(name, result, code, settings)
    st.session_state["preliminary_display_name"] = name
    return name


def _show_registered_result(
    display_dataframe: Callable[..., None],
) -> None:
    name = st.session_state.get("preliminary_display_name")
    if not name or name not in st.session_state.results:
        return
    result = st.session_state.results[name]
    st.caption(f"Displayed preliminary result: `{name}`")
    if isinstance(result, pd.DataFrame):
        display_dataframe(result, use_container_width=True)
    elif isinstance(result, pd.Series):
        display_dataframe(result.to_frame("Value"), use_container_width=True)
    else:
        st.text(str(result))
    for key, title in [
        (name + "_details", "Detailed results"),
        (name + "_secondary", "Additional results"),
    ]:
        extra = st.session_state.results.get(key)
        if isinstance(extra, pd.DataFrame) and not extra.empty:
            st.subheader(title)
            display_dataframe(extra, use_container_width=True)


def render_preliminary_workspace(
    data: pd.DataFrame,
    *,
    display_dataframe: Callable[..., None],
    analysis_action_buttons: Callable[..., bool],
    register_output: Callable[..., None],
    make_unique_name: Callable[[str], str],
    display_exception: Callable[[Exception], None],
) -> None:
    st.caption(
        "General, time-series and panel-data screening before estimation. "
        "Each retained result is added to generated code and the complete export."
    )
    numeric = list(data.select_dtypes(include=[np.number]).columns)
    if not numeric:
        st.warning("No numeric variables are available.")
        return

    category = st.radio(
        "Preliminary-analysis family",
        ["General and descriptive", "Time series", "Panel data"],
        key="active_preliminary_family",
        horizontal=True,
    )
    methods = (
        GENERAL_METHODS
        if category == "General and descriptive"
        else TIME_SERIES_METHODS
        if category == "Time series"
        else PANEL_METHODS
    )
    method = st.selectbox("Preliminary method", methods, key="active_preliminary_method")
    alpha = float(
        st.selectbox(
            "Significance level",
            [0.01, 0.05, 0.10],
            index=1,
            format_func=lambda value: f"{value:.0%}",
            key="prelim_alpha",
        )
    )

    entity = time = None
    if category == "Panel data" or method in {
        "Data and panel structure",
        "Country/entity and time trend graphs",
    }:
        with st.expander("Panel identifiers", expanded=category == "Panel data"):
            entity = st.selectbox(
                "Country/entity identifier",
                list(data.columns),
                key="prelim_entity",
            )
            time_options = [column for column in data.columns if column != entity]
            time = st.selectbox("Time identifier", time_options, key="prelim_time")

    try:
        if method == "Data and panel structure":
            run = analysis_action_buttons(
                "Create structure report",
                "prelim_structure",
                clear_label="Clear latest structure report",
            )
            if run:
                overview, counts = panel_structure_table(data, entity, time)
                missing = missingness_table(data)
                name = _store_and_show(
                    slot="prelim_structure",
                    prefix="Preliminary_structure",
                    result=overview,
                    code=(
                        "from preliminary import panel_structure_table, missingness_table\n"
                        f"overview, entity_counts = panel_structure_table(data, {entity!r}, {time!r})\n"
                        "missingness = missingness_table(data)\n"
                        "print(overview)\nprint(entity_counts)\nprint(missingness)"
                    ),
                    settings={"method": method, "entity": entity, "time": time},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )
                st.session_state.results[name + "_details"] = counts
                st.session_state.results[name + "_secondary"] = missing

        elif method == "Descriptive statistics":
            variables = st.multiselect("Variables", numeric, default=numeric[: min(8, len(numeric))])
            if analysis_action_buttons(
                "Run descriptive statistics",
                "prelim_descriptive",
                clear_label="Clear latest descriptive statistics",
            ):
                if not variables:
                    raise ValueError("Select at least one variable.")
                table = descriptive_statistics_table(data, variables)
                _store_and_show(
                    slot="prelim_descriptive",
                    prefix="Preliminary_descriptive",
                    result=table,
                    code=(
                        "from preliminary import descriptive_statistics_table\n"
                        f"preliminary_descriptive = descriptive_statistics_table(data, {variables!r})\n"
                        "print(preliminary_descriptive)"
                    ),
                    settings={"method": method, "variables": variables},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )

        elif method == "Distribution and normality":
            variables = st.multiselect("Variables", numeric, default=numeric[: min(8, len(numeric))])
            if analysis_action_buttons(
                "Run distribution diagnostics",
                "prelim_distribution",
                clear_label="Clear latest distribution diagnostics",
            ):
                if not variables:
                    raise ValueError("Select at least one variable.")
                table = distribution_diagnostics_table(data, variables, alpha)
                _store_and_show(
                    slot="prelim_distribution",
                    prefix="Preliminary_distribution",
                    result=table,
                    code=(
                        "from preliminary import distribution_diagnostics_table\n"
                        f"distribution_diagnostics = distribution_diagnostics_table(data, {variables!r}, {alpha!r})\n"
                        "print(distribution_diagnostics)"
                    ),
                    settings={"method": method, "variables": variables, "alpha": alpha},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )

        elif method == "Correlation and covariance":
            variables = st.multiselect("Variables", numeric, default=numeric[: min(8, len(numeric))])
            corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
            if analysis_action_buttons(
                "Run correlation and covariance",
                "prelim_correlation",
                clear_label="Clear latest correlation/covariance",
            ):
                if len(variables) < 2:
                    raise ValueError("Select at least two variables.")
                corr, cov = correlation_covariance_tables(data, variables, corr_method)
                name = _store_and_show(
                    slot="prelim_correlation",
                    prefix="Preliminary_correlation",
                    result=corr,
                    code=(
                        "from preliminary import correlation_covariance_tables\n"
                        f"correlation, covariance = correlation_covariance_tables(data, {variables!r}, {corr_method!r})\n"
                        "print(correlation)\nprint(covariance)"
                    ),
                    settings={"method": method, "variables": variables, "correlation": corr_method},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )
                st.session_state.results[name + "_secondary"] = cov

        elif method == "Variance inflation factors":
            variables = st.multiselect("Explanatory variables", numeric, default=numeric[: min(6, len(numeric))])
            if analysis_action_buttons(
                "Calculate VIF and condition indices",
                "prelim_vif",
                clear_label="Clear latest VIF result",
            ):
                if len(variables) < 2:
                    raise ValueError("Select at least two explanatory variables.")
                vif, condition = vif_condition_table(data, variables)
                name = _store_and_show(
                    slot="prelim_vif",
                    prefix="Preliminary_VIF",
                    result=vif,
                    code=(
                        "from preliminary import vif_condition_table\n"
                        f"vif, condition_indices = vif_condition_table(data, {variables!r})\n"
                        "print(vif)\nprint(condition_indices)"
                    ),
                    settings={"method": method, "variables": variables},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )
                st.session_state.results[name + "_secondary"] = condition

        elif method == "Country/entity and time trend graphs":
            variable = st.selectbox("Variable", numeric)
            graph = st.selectbox(
                "Trend graph",
                [
                    "All country/entity lines",
                    "Selected countries/entities",
                    "Cross-sectional mean and median",
                    "Country/entity means",
                    "Time-period box plots",
                ],
            )
            selected_entities = []
            if graph == "Selected countries/entities":
                values = list(pd.Series(data[entity].dropna().unique()).sort_values())
                selected_entities = st.multiselect("Countries/entities", values, default=values[: min(6, len(values))])
            draw_col, clear_col = st.columns(2)
            if draw_col.button("Draw preliminary graph", type="primary", use_container_width=True):
                if graph == "Selected countries/entities" and not selected_entities:
                    raise ValueError("Select at least one country/entity.")
                st.session_state.preliminary_chart_state = {
                    "variable": variable,
                    "entity": entity,
                    "time": time,
                    "graph": graph,
                    "entities": selected_entities,
                }
            if clear_col.button(
                "Clear preliminary graph",
                use_container_width=True,
                disabled=st.session_state.get("preliminary_chart_state") is None,
            ):
                st.session_state.preliminary_chart_state = None

        elif method == "Time-series unit-root tests":
            variables = st.multiselect("Series", numeric, default=numeric[: min(5, len(numeric))])
            test = st.selectbox("Test", ["ADF", "Phillips-Perron", "KPSS"])
            lags = int(st.number_input("Lag/bandwidth", 0, 100, 1))
            trend = st.selectbox("Deterministic terms", ["c", "ct", "ctt", "n"])
            if analysis_action_buttons(
                "Run time-series unit-root tests",
                "prelim_ts_unit_root",
                clear_label="Clear latest time-series unit-root result",
            ):
                if not variables:
                    raise ValueError("Select at least one series.")
                table = time_series_unit_root_table(data, variables, test, lags, trend, alpha)
                _store_and_show(
                    slot="prelim_ts_unit_root",
                    prefix="Preliminary_time_series_unit_root",
                    result=table,
                    code=(
                        "from preliminary import time_series_unit_root_table\n"
                        f"unit_root_results = time_series_unit_root_table(data, {variables!r}, {test!r}, {lags}, {trend!r}, {alpha!r})\n"
                        "print(unit_root_results)"
                    ),
                    settings={"method": method, "variables": variables, "test": test, "lags": lags, "trend": trend},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )

        elif method == "Serial-correlation and ARCH screening":
            variables = st.multiselect("Series", numeric, default=numeric[: min(5, len(numeric))])
            lag = int(st.number_input("Maximum lag", 1, 100, 1))
            if analysis_action_buttons(
                "Run serial-correlation and ARCH tests",
                "prelim_serial_arch",
                clear_label="Clear latest serial/ARCH result",
            ):
                if not variables:
                    raise ValueError("Select at least one series.")
                table = serial_arch_table(data, variables, lag, alpha)
                _store_and_show(
                    slot="prelim_serial_arch",
                    prefix="Preliminary_serial_ARCH",
                    result=table,
                    code=(
                        "from preliminary import serial_arch_table\n"
                        f"serial_arch_results = serial_arch_table(data, {variables!r}, {lag}, {alpha!r})\n"
                        "print(serial_arch_results)"
                    ),
                    settings={"method": method, "variables": variables, "lag": lag},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )

        elif method == "ACF and PACF graphs":
            variable = st.selectbox("Series", numeric)
            lags = int(st.number_input("Displayed lags", 1, 100, 20))
            draw_col, clear_col = st.columns(2)
            if draw_col.button("Draw ACF/PACF", type="primary", use_container_width=True):
                st.session_state.preliminary_chart_state = {
                    "graph": "ACF and PACF",
                    "variable": variable,
                    "lags": lags,
                }
            if clear_col.button(
                "Clear ACF/PACF",
                use_container_width=True,
                disabled=st.session_state.get("preliminary_chart_state") is None,
            ):
                st.session_state.preliminary_chart_state = None

        elif method == "Within-between variation":
            variables = st.multiselect("Variables", numeric, default=numeric[: min(8, len(numeric))])
            if analysis_action_buttons(
                "Calculate within-between variation",
                "prelim_within_between",
                clear_label="Clear latest within-between result",
            ):
                if not variables:
                    raise ValueError("Select at least one variable.")
                table = within_between_table(data, entity, variables)
                _store_and_show(
                    slot="prelim_within_between",
                    prefix="Preliminary_within_between",
                    result=table,
                    code=(
                        "from preliminary import within_between_table\n"
                        f"within_between = within_between_table(data, {entity!r}, {variables!r})\n"
                        "print(within_between)"
                    ),
                    settings={"method": method, "entity": entity, "time": time, "variables": variables},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )

        elif method == "Cross-sectional dependence tests":
            variables = st.multiselect("Variables", numeric, default=numeric[: min(5, len(numeric))])
            if analysis_action_buttons(
                "Run cross-sectional dependence tests",
                "prelim_cd",
                clear_label="Clear latest dependence tests",
            ):
                if not variables:
                    raise ValueError("Select at least one variable.")
                table = cross_section_dependence_table(data, entity, time, variables, alpha)
                _store_and_show(
                    slot="prelim_cd",
                    prefix="Preliminary_cross_section_dependence",
                    result=table,
                    code=(
                        "from preliminary import cross_section_dependence_table\n"
                        f"cd_results = cross_section_dependence_table(data, {entity!r}, {time!r}, {variables!r}, {alpha!r})\n"
                        "print(cd_results)"
                    ),
                    settings={"method": method, "entity": entity, "time": time, "variables": variables},
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )

        elif method == "Pesaran-Yamagata slope heterogeneity test":
            dependent = st.selectbox("Dependent variable", numeric)
            regressors = st.multiselect("Slope variables", [value for value in numeric if value != dependent])
            st.caption(
                "The test compares entity-specific within slopes with the pooled fixed-effects slope. "
                "The intercept is allowed to differ across entities."
            )
            if analysis_action_buttons(
                "Run Pesaran-Yamagata test",
                "prelim_slope_heterogeneity",
                clear_label="Clear latest slope-homogeneity result",
            ):
                summary, details = pesaran_yamagata_test(
                    data, entity, time, dependent, regressors, alpha
                )
                name = _store_and_show(
                    slot="prelim_slope_heterogeneity",
                    prefix="Preliminary_Pesaran_Yamagata",
                    result=summary,
                    code=(
                        "from preliminary import pesaran_yamagata_test\n"
                        f"slope_test, entity_slopes = pesaran_yamagata_test(data, {entity!r}, {time!r}, {dependent!r}, {regressors!r}, {alpha!r})\n"
                        "print(slope_test)\nprint(entity_slopes)"
                    ),
                    settings={
                        "method": method,
                        "entity": entity,
                        "time": time,
                        "dependent": dependent,
                        "regressors": regressors,
                    },
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )
                st.session_state.results[name + "_details"] = details

        elif method == "CIPS panel unit-root test":
            variable = st.selectbox("Panel variable", numeric)
            lags = int(st.number_input("CADF lags", 0, 10, 1))
            trend = st.selectbox("Deterministic terms", ["c", "ct", "ctt", "n"])
            simulations = int(st.number_input("Null simulations", 49, 499, 99, 50))
            seed = int(st.number_input("Random seed", 0, 999999, 42))
            if analysis_action_buttons(
                "Run CIPS panel unit-root test",
                "prelim_cips",
                clear_label="Clear latest CIPS result",
            ):
                summary, details = cips_panel_unit_root_test(
                    data, entity, time, variable, lags, trend, simulations, seed, alpha
                )
                name = _store_and_show(
                    slot="prelim_cips",
                    prefix="Preliminary_CIPS",
                    result=summary,
                    code=(
                        "from preliminary import cips_panel_unit_root_test\n"
                        f"cips, cadf_details = cips_panel_unit_root_test(data, {entity!r}, {time!r}, {variable!r}, {lags}, {trend!r}, {simulations}, {seed}, {alpha!r})\n"
                        "print(cips)\nprint(cadf_details)"
                    ),
                    settings={
                        "method": method,
                        "entity": entity,
                        "time": time,
                        "variable": variable,
                        "lags": lags,
                        "trend": trend,
                        "simulations": simulations,
                    },
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )
                st.session_state.results[name + "_details"] = details

        elif method == "Fisher panel unit-root tests":
            variables = st.multiselect("Panel variables", numeric, default=numeric[: min(5, len(numeric))])
            test = st.selectbox("Individual test", ["ADF", "Phillips-Perron"])
            lags = int(st.number_input("Lag/bandwidth", 0, 20, 1))
            trend = st.selectbox("Deterministic terms", ["c", "ct", "ctt", "n"])
            if analysis_action_buttons(
                "Run Fisher panel unit-root tests",
                "prelim_fisher_unit_root",
                clear_label="Clear latest Fisher unit-root result",
            ):
                if not variables:
                    raise ValueError("Select at least one variable.")
                summary, details = fisher_panel_unit_root_test(
                    data, entity, time, variables, test, lags, trend, alpha
                )
                name = _store_and_show(
                    slot="prelim_fisher_unit_root",
                    prefix="Preliminary_Fisher_unit_root",
                    result=summary,
                    code=(
                        "from preliminary import fisher_panel_unit_root_test\n"
                        f"fisher_panel, individual_tests = fisher_panel_unit_root_test(data, {entity!r}, {time!r}, {variables!r}, {test!r}, {lags}, {trend!r}, {alpha!r})\n"
                        "print(fisher_panel)\nprint(individual_tests)"
                    ),
                    settings={
                        "method": method,
                        "entity": entity,
                        "time": time,
                        "variables": variables,
                        "test": test,
                        "lags": lags,
                        "trend": trend,
                    },
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )
                st.session_state.results[name + "_details"] = details

        elif method == "Westerlund panel cointegration test":
            dependent = st.selectbox("Dependent variable", numeric)
            regressors = st.multiselect("Long-run regressors", [value for value in numeric if value != dependent])
            c1, c2, c3 = st.columns(3)
            lags = int(c1.number_input("Short-run lags", 0, 10, 1))
            leads = int(c2.number_input("Short-run leads", 0, 10, 0))
            bootstrap = int(c3.number_input("Bootstrap replications", 0, 999, 99, 50))
            constant = st.checkbox("Include constant", value=True)
            trend = st.checkbox("Include trend", value=False)
            seed = int(st.number_input("Random seed", 0, 999999, 42))
            if not WESTERLUND_AVAILABLE:
                st.warning(
                    "The optional Westerlund package is not currently importable. "
                    "The deployment will install it from requirements.txt."
                )
            if analysis_action_buttons(
                "Run Westerlund panel cointegration test",
                "prelim_westerlund",
                clear_label="Clear latest Westerlund result",
            ):
                table, raw = run_westerlund_test(
                    data,
                    entity,
                    time,
                    dependent,
                    regressors,
                    lags,
                    leads,
                    constant,
                    trend,
                    bootstrap,
                    seed,
                )
                _store_and_show(
                    slot="prelim_westerlund",
                    prefix="Preliminary_Westerlund",
                    result=table,
                    code=(
                        "from westerlund_test import WesterlundTest\n"
                        f"test = WesterlundTest(data=data, y_var={dependent!r}, x_vars={regressors!r}, "
                        f"id_var={entity!r}, time_var={time!r}, lags={lags}, leads={leads}, "
                        f"constant={constant!r}, trend={trend!r}, bootstrap={bootstrap}, seed={seed})\n"
                        "westerlund_result = test.run()\nprint(westerlund_result)"
                    ),
                    settings={
                        "method": method,
                        "entity": entity,
                        "time": time,
                        "dependent": dependent,
                        "regressors": regressors,
                        "lags": lags,
                        "leads": leads,
                        "bootstrap": bootstrap,
                    },
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )

        elif method == "Residual-based panel cointegration screening":
            dependent = st.selectbox("Dependent variable", numeric)
            regressors = st.multiselect("Long-run regressors", [value for value in numeric if value != dependent])
            lags = int(st.number_input("Residual ADF lags", 0, 20, 1))
            if analysis_action_buttons(
                "Run residual-based panel cointegration screen",
                "prelim_residual_cointegration",
                clear_label="Clear latest residual cointegration result",
            ):
                summary, details = residual_panel_cointegration_test(
                    data, entity, time, dependent, regressors, lags, alpha
                )
                name = _store_and_show(
                    slot="prelim_residual_cointegration",
                    prefix="Preliminary_residual_cointegration",
                    result=summary,
                    code=(
                        "from preliminary import residual_panel_cointegration_test\n"
                        f"cointegration_screen, individual_residual_adf = residual_panel_cointegration_test(data, {entity!r}, {time!r}, {dependent!r}, {regressors!r}, {lags}, {alpha!r})\n"
                        "print(cointegration_screen)\nprint(individual_residual_adf)"
                    ),
                    settings={
                        "method": method,
                        "entity": entity,
                        "time": time,
                        "dependent": dependent,
                        "regressors": regressors,
                        "lags": lags,
                    },
                    register_output=register_output,
                    make_unique_name=make_unique_name,
                )
                st.session_state.results[name + "_details"] = details

    except Exception as exc:
        display_exception(exc)

    _show_registered_result(display_dataframe)

    chart = st.session_state.get("preliminary_chart_state")
    if chart:
        try:
            if chart.get("graph") == "ACF and PACF":
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

                series = pd.to_numeric(data[chart["variable"]], errors="coerce").dropna()
                max_lag = min(int(chart["lags"]), max(1, len(series) // 2 - 1))
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                plot_acf(series, lags=max_lag, ax=axes[0])
                plot_pacf(series, lags=max_lag, ax=axes[1], method="ywm")
                axes[0].set_title(f"ACF: {chart['variable']}")
                axes[1].set_title(f"PACF: {chart['variable']}")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                variable = chart["variable"]
                entity_name = chart["entity"]
                time_name = chart["time"]
                graph = chart["graph"]
                work = data[[entity_name, time_name, variable]].dropna().copy()
                work[variable] = pd.to_numeric(work[variable], errors="coerce")
                work = work.dropna().sort_values([entity_name, time_name])
                fig, ax = plt.subplots(figsize=(11, 6))
                if graph == "All country/entity lines":
                    for unit, group in work.groupby(entity_name, observed=True):
                        ax.plot(group[time_name], group[variable], alpha=0.45, linewidth=1)
                    ax.set_title(f"{variable}: all {entity_name} trends")
                elif graph == "Selected countries/entities":
                    selected = chart["entities"]
                    for unit, group in work[work[entity_name].isin(selected)].groupby(entity_name, observed=True):
                        ax.plot(group[time_name], group[variable], label=str(unit))
                    ax.legend()
                    ax.set_title(f"{variable}: selected {entity_name} trends")
                elif graph == "Cross-sectional mean and median":
                    aggregate = work.groupby(time_name, observed=True)[variable].agg(["mean", "median"])
                    ax.plot(aggregate.index, aggregate["mean"], label="Mean")
                    ax.plot(aggregate.index, aggregate["median"], label="Median")
                    ax.legend()
                    ax.set_title(f"{variable}: cross-sectional time trend")
                elif graph == "Country/entity means":
                    means = work.groupby(entity_name, observed=True)[variable].mean().sort_values()
                    means.plot.barh(ax=ax)
                    ax.set_title(f"Average {variable} by {entity_name}")
                else:
                    ordered_times = sorted(work[time_name].unique())
                    groups = [
                        work.loc[work[time_name] == value, variable].dropna().to_numpy()
                        for value in ordered_times
                    ]
                    ax.boxplot(groups, labels=[str(value) for value in ordered_times])
                    ax.tick_params(axis="x", rotation=90)
                    ax.set_title(f"{variable}: distribution by {time_name}")
                ax.set_xlabel(str(time_name if "time" in graph.lower() or "line" in graph.lower() else entity_name))
                ax.set_ylabel(variable)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        except Exception as exc:
            display_exception(exc)
