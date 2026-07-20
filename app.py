
from __future__ import annotations

import hashlib
import io
import json
import platform
from pathlib import Path
import re
import sys
import textwrap
import traceback
import warnings
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import statsmodels.api as sm
import streamlit as st
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_arch,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen

try:
    from arch import arch_model
    from arch.unitroot import DFGLS, PhillipsPerron, ZivotAndrews
    ARCH_AVAILABLE = True
    ARCH_IMPORT_ERROR = ""
except Exception as exc:
    ARCH_AVAILABLE = False
    ARCH_IMPORT_ERROR = str(exc)

try:
    from linearmodels.iv import IV2SLS, IVGMM, IVLIML
    from linearmodels.panel import (
        BetweenOLS,
        FamaMacBeth,
        FirstDifferenceOLS,
        PanelOLS,
        PooledOLS,
        RandomEffects,
    )
    LINEARMODELS_AVAILABLE = True
    LINEARMODELS_IMPORT_ERROR = ""
except Exception as exc:
    LINEARMODELS_AVAILABLE = False
    LINEARMODELS_IMPORT_ERROR = str(exc)


APP_VERSION = "1.2.0"
DEFAULT_DATA_FILE = "cleaned_dataset.csv"

st.set_page_config(
    page_title="Econometrics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    defaults = {
        "df": None,
        "original_df": None,
        "file_signature": None,
        "source_filename": None,
        "history": [],
        "code_blocks": [],
        "results": {},
        "settings_log": [],
        "slot_outputs": {},
        "dark_mode": False,
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_project() -> None:
    for key in [
        "df",
        "original_df",
        "file_signature",
        "source_filename",
        "history",
        "code_blocks",
        "results",
        "settings_log",
        "slot_outputs",
        "_active_registration_slot",
        "last_error",
    ]:
        st.session_state.pop(key, None)
    init_state()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=[np.number]).columns)


def clean_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.loc[:, columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def safe_name(value: str, max_length: int = 31) -> str:
    value = re.sub(r"[\[\]:*?/\\]", "_", str(value)).strip()
    return (value or "result")[:max_length]


def make_unique_name(prefix: str) -> str:
    stamp = datetime.now().strftime("%H%M%S")
    existing = st.session_state.results
    candidate = f"{prefix}_{stamp}"
    index = 2
    while candidate in existing:
        candidate = f"{prefix}_{stamp}_{index}"
        index += 1
    return candidate


def add_history(message: str) -> None:
    st.session_state.history.append(
        {
            "time": datetime.now().isoformat(timespec="seconds"),
            "action": message,
        }
    )


def register_output(
    name: str,
    result: Any,
    code: str,
    settings: dict[str, Any] | None = None,
    summary: str | None = None,
) -> None:
    st.session_state.results[name] = result
    if code.strip():
        st.session_state.code_blocks.append(
            {
                "name": name,
                "code": textwrap.dedent(code).strip(),
            }
        )
    st.session_state.settings_log.append(
        {
            "name": name,
            "time": datetime.now().isoformat(timespec="seconds"),
            "settings": settings or {},
            "summary": summary or "",
        }
    )
    slot = st.session_state.pop("_active_registration_slot", None)
    if slot:
        stack = st.session_state.slot_outputs.setdefault(slot, [])
        if isinstance(stack, str):
            stack = [stack]
            st.session_state.slot_outputs[slot] = stack
        stack.append(name)

    add_history(f"Completed {name}")


def _replace_text_in_object(value: Any, old: str, new: str) -> Any:
    """Recursively update variable references stored in settings and history."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_text_in_object(item, old, new) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_text_in_object(item, old, new) for item in value)
    if isinstance(value, dict):
        return {
            _replace_text_in_object(key, old, new): _replace_text_in_object(item, old, new)
            for key, item in value.items()
        }
    return value


def _renamed_axis(axis: pd.Index, old: str, new: str) -> pd.Index:
    def rename_label(label: Any) -> Any:
        if isinstance(label, tuple):
            return tuple(rename_label(item) for item in label)
        if isinstance(label, str):
            return label.replace(old, new)
        return label

    labels = [rename_label(label) for label in axis]
    names = [rename_label(name) for name in axis.names]
    if isinstance(axis, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(labels, names=names)
    return pd.Index(labels, name=names[0] if names else None)


def _rename_result_labels(value: Any, old: str, new: str) -> Any:
    """Rename variable labels in saved tables and textual summaries."""
    if isinstance(value, pd.DataFrame):
        updated = value.copy()
        updated.index = _renamed_axis(updated.index, old, new)
        updated.columns = _renamed_axis(updated.columns, old, new)
        return updated
    if isinstance(value, pd.Series):
        updated = value.copy()
        updated.index = _renamed_axis(updated.index, old, new)
        if isinstance(updated.name, str):
            updated.name = updated.name.replace(old, new)
        return updated
    if isinstance(value, str):
        return value.replace(old, new)
    return value


def rename_variable_everywhere(old: str, new: str) -> None:
    """Rename a current-data column and update recorded reproducibility objects."""
    st.session_state.df = st.session_state.df.rename(columns={old: new})

    old_literal = repr(old)
    new_literal = repr(new)
    for block in st.session_state.code_blocks:
        block["code"] = block["code"].replace(old_literal, new_literal)

    st.session_state.settings_log = _replace_text_in_object(
        st.session_state.settings_log, old, new
    )
    st.session_state.history = _replace_text_in_object(
        st.session_state.history, old, new
    )

    st.session_state.results = {
        key: _rename_result_labels(value, old, new)
        for key, value in st.session_state.results.items()
    }

    for key in list(st.session_state.keys()):
        if key.startswith("last_"):
            st.session_state[key] = _rename_result_labels(
                st.session_state[key], old, new
            )

    rename_code = f"data = data.rename(columns={{{old!r}: {new!r}}})"
    register_output(
        make_unique_name("Rename_variable"),
        pd.DataFrame(
            {"Previous variable name": [old], "New variable name": [new]}
        ),
        rename_code,
        {
            "operation": "Rename variable",
            "previous_name": old,
            "new_name": new,
        },
    )


def registered_analysis_names() -> list[str]:
    """Return registered analysis/operation identifiers in creation order."""
    names: list[str] = []
    for entry in st.session_state.settings_log:
        name = entry.get("name")
        if name and name not in names:
            names.append(name)
    return names


def _record_owner(record_name: str, registered_names: list[str]) -> str | None:
    """Find the registered analysis that owns a result or code-block name."""
    matches = [
        name
        for name in registered_names
        if record_name == name or record_name.startswith(name + "_")
    ]
    return max(matches, key=len) if matches else None


def remove_recorded_analyses(selected_names: list[str]) -> int:
    """
    Remove selected analyses from code, result files, settings and exported history.

    The current working dataset is intentionally left unchanged.
    """
    selected = set(selected_names)
    if not selected:
        return 0

    all_names = registered_analysis_names()

    st.session_state.code_blocks = [
        block
        for block in st.session_state.code_blocks
        if _record_owner(str(block.get("name", "")), all_names) not in selected
    ]

    st.session_state.results = {
        key: value
        for key, value in st.session_state.results.items()
        if _record_owner(str(key), all_names) not in selected
    }

    st.session_state.settings_log = [
        entry
        for entry in st.session_state.settings_log
        if entry.get("name") not in selected
    ]

    completed_actions = {f"Completed {name}" for name in selected}
    st.session_state.history = [
        item
        for item in st.session_state.history
        if item.get("action") not in completed_actions
    ]

    for slot, stack in list(st.session_state.slot_outputs.items()):
        if isinstance(stack, str):
            stack = [stack]
        remaining = [name for name in stack if name not in selected]
        if remaining:
            st.session_state.slot_outputs[slot] = remaining
        else:
            st.session_state.slot_outputs.pop(slot, None)

    for key in list(st.session_state.keys()):
        if key.startswith("last_"):
            del st.session_state[key]

    return len(selected)


def _slot_stack(slot: str) -> list[str]:
    stack = st.session_state.slot_outputs.get(slot, [])
    if isinstance(stack, str):
        stack = [stack]
        st.session_state.slot_outputs[slot] = stack
    return stack


def latest_slot_output(slot: str) -> str | None:
    stack = _slot_stack(slot)
    return stack[-1] if stack else None


def clear_latest_slot_output(slot: str) -> str | None:
    current = latest_slot_output(slot)
    if current is None:
        return None
    remove_recorded_analyses([current])
    stack = _slot_stack(slot)
    if current in stack:
        stack.remove(current)
    if not stack:
        st.session_state.slot_outputs.pop(slot, None)
    return current


def render_slot_clear(
    slot: str,
    label: str = "Clear latest result",
    *,
    key: str | None = None,
) -> None:
    current = latest_slot_output(slot)
    clicked = st.button(
        label,
        key=key or f"clear_{slot}",
        disabled=current is None,
        use_container_width=True,
        type="secondary",
    )
    if current:
        st.caption(f"Latest recorded item: `{current}`")
    if clicked:
        removed = clear_latest_slot_output(slot)
        st.session_state["local_clear_notice"] = (
            f"Removed {removed!r} from generated code and the full export."
        )
        st.rerun()


def analysis_action_buttons(
    run_label: str,
    slot: str,
    *,
    clear_label: str = "Clear latest result",
    primary: bool = True,
) -> bool:
    run_col, clear_col = st.columns(2)
    run_clicked = run_col.button(
        run_label,
        key=f"run_{slot}",
        type="primary" if primary else "secondary",
        use_container_width=True,
    )
    if run_clicked:
        st.session_state["_active_registration_slot"] = slot

    with clear_col:
        render_slot_clear(
            slot,
            clear_label,
            key=f"clear_{slot}_near_action",
        )
    return run_clicked


def apply_dynamic_theme(dark_mode: bool) -> None:
    if not dark_mode:
        return

    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }

        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background-color: #0e1117 !important;
            color: #f3f4f6 !important;
        }

        [data-testid="stSidebar"] {
            background-color: #161b22 !important;
            border-right: 1px solid #30363d !important;
        }

        [data-testid="stSidebar"] *,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stApp p, .stApp label, .stApp span,
        .stApp [data-testid="stMarkdownContainer"] {
            color: #f3f4f6;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"],
        input, textarea {
            background-color: #1f2937 !important;
            color: #f9fafb !important;
            border-color: #4b5563 !important;
        }

        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[role="listbox"] {
            background-color: #1f2937 !important;
            color: #f9fafb !important;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stMetric"],
        div[data-testid="stExpander"],
        div[data-testid="stAlert"] {
            background-color: #161b22 !important;
            color: #f3f4f6 !important;
            border-color: #30363d !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            background-color: #161b22 !important;
            border-radius: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            color: #d1d5db !important;
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            border-bottom-color: #60a5fa !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-color: #4b5563 !important;
        }

        code, pre, .stCodeBlock {
            background-color: #111827 !important;
            color: #e5e7eb !important;
        }

        hr {
            border-color: #30363d !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def parameter_table(results: Any) -> pd.DataFrame:
    params = pd.Series(results.params)
    bse = pd.Series(results.bse, index=params.index)
    pvalues = pd.Series(results.pvalues, index=params.index)
    statistic = getattr(results, "tvalues", None)
    if statistic is None:
        statistic = getattr(results, "zvalues", np.full(len(params), np.nan))
    statistic = pd.Series(statistic, index=params.index)
    conf = results.conf_int()
    if not isinstance(conf, pd.DataFrame):
        conf = pd.DataFrame(conf, index=params.index)
    conf.columns = ["CI Lower", "CI Upper"]
    return pd.DataFrame(
        {
            "Coefficient": params,
            "Std. Error": bse,
            "Statistic": statistic,
            "P-value": pvalues,
            "CI Lower": conf["CI Lower"],
            "CI Upper": conf["CI Upper"],
        }
    )


def linearmodels_parameter_table(results: Any) -> pd.DataFrame:
    conf = results.conf_int()
    return pd.DataFrame(
        {
            "Coefficient": results.params,
            "Std. Error": results.std_errors,
            "Statistic": results.tstats,
            "P-value": results.pvalues,
            "CI Lower": conf.iloc[:, 0],
            "CI Upper": conf.iloc[:, 1],
        }
    )


def summary_text(results: Any) -> str:
    summary = getattr(results, "summary", None)
    if callable(summary):
        summary = summary()
    if hasattr(summary, "as_text"):
        return summary.as_text()
    return str(summary)


def dataframe_to_excel_bytes(frames: dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        used: set[str] = set()
        for raw_name, frame in frames.items():
            name = safe_name(raw_name)
            base_name = name
            counter = 2
            while name in used:
                suffix = f"_{counter}"
                name = f"{base_name[:31-len(suffix)]}{suffix}"
                counter += 1
            used.add(name)
            frame.to_excel(writer, sheet_name=name)
    return output.getvalue()


def generated_analysis_script() -> str:
    header = f"""# Reproducible analysis generated by Econometrics Studio {APP_VERSION}.\n# Generated: {datetime.now().isoformat(timespec="seconds")}\n# Place cleaned_dataset.csv in the same directory before running this script.\n\nimport warnings\nwarnings.filterwarnings("ignore")\n\nimport numpy as np\nimport pandas as pd\nimport scipy.stats as scipy_stats\nimport statsmodels.api as sm\nfrom statsmodels.stats.diagnostic import (\n    acorr_breusch_godfrey, het_arch, het_breuschpagan, het_white, linear_reset\n)\nfrom statsmodels.stats.outliers_influence import variance_inflation_factor\nfrom statsmodels.stats.stattools import durbin_watson, jarque_bera\nfrom statsmodels.tsa.api import VAR\nfrom statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order\nfrom statsmodels.tsa.arima.model import ARIMA\nfrom statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss\nfrom statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen\n\ntry:\n    from arch import arch_model\n    from arch.unitroot import DFGLS, PhillipsPerron, ZivotAndrews\nexcept ImportError:\n    pass\n\ntry:\n    from linearmodels.iv import IV2SLS, IVGMM, IVLIML\n    from linearmodels.panel import (\n        BetweenOLS, FamaMacBeth, FirstDifferenceOLS,\n        PanelOLS, PooledOLS, RandomEffects\n    )\nexcept ImportError:\n    pass\n\ndata = pd.read_csv("{DEFAULT_DATA_FILE}")\n"""
    blocks = []
    for entry in st.session_state.code_blocks:
        blocks.append(
            "\n\n# " + "=" * 78 + "\n"
            + f"# {entry['name']}\n"
            + "# " + "=" * 78 + "\n"
            + entry["code"]
        )
    return header + "".join(blocks) + "\n"


def build_reproduction_zip() -> bytes:
    buffer = io.BytesIO()
    frames: dict[str, pd.DataFrame] = {}
    texts: dict[str, str] = {}

    for name, result in st.session_state.results.items():
        if isinstance(result, pd.DataFrame):
            frames[name] = result
        elif isinstance(result, pd.Series):
            frames[name] = result.to_frame("Value")
        else:
            texts[name] = str(result)

    config = {
        "application": "Econometrics Studio",
        "application_version": APP_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "source_filename": st.session_state.source_filename,
        "history": st.session_state.history,
        "analyses": st.session_state.settings_log,
    }

    readme = f"""# Econometrics Studio reproduction package

Generated with Econometrics Studio {APP_VERSION}.

## Files

- `analysis.py`: generated Python code for the analyses performed.
- `cleaned_dataset.csv`: the current dataset, including transformations.
- `results.xlsx`: tabular results.
- `text_results/`: textual model summaries.
- `analysis_configuration.json`: analysis history and settings.
- `requirements.txt`: required packages.

## Run

```bash
pip install -r requirements.txt
python analysis.py
```

Review model assumptions and data definitions before publication.
"""

    requirements = Path(__file__).with_name("requirements.txt")
    requirement_text = (
        requirements.read_text(encoding="utf-8")
        if requirements.exists()
        else "streamlit\npandas\nnumpy\nscipy\nstatsmodels\narch\nlinearmodels\nmatplotlib\nopenpyxl\nxlsxwriter\n"
    )

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("analysis.py", generated_analysis_script())
        archive.writestr("analysis_configuration.json", json.dumps(config, indent=2, default=str))
        archive.writestr("requirements.txt", requirement_text)
        archive.writestr("README.md", readme)

        if st.session_state.df is not None:
            archive.writestr(
                DEFAULT_DATA_FILE,
                st.session_state.df.to_csv(index=False).encode("utf-8"),
            )

        if frames:
            archive.writestr("results.xlsx", dataframe_to_excel_bytes(frames))

        for name, text in texts.items():
            archive.writestr(f"text_results/{safe_name(name, 80)}.txt", text)

    return buffer.getvalue()


def display_exception(exc: Exception) -> None:
    st.session_state.pop("_active_registration_slot", None)
    st.error(f"{type(exc).__name__}: {exc}")
    st.session_state.last_error = traceback.format_exc()
    with st.expander("Technical details"):
        st.code(st.session_state.last_error, language="text")


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    records = {}
    for column in df.columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        jb_stat, jb_p, _, _ = jarque_bera(series)
        records[column] = {
            "Observations": series.count(),
            "Mean": series.mean(),
            "Median": series.median(),
            "Maximum": series.max(),
            "Minimum": series.min(),
            "Std. Deviation": series.std(ddof=1),
            "Variance": series.var(ddof=1),
            "Skewness": series.skew(),
            "Kurtosis": series.kurtosis() + 3,
            "Jarque-Bera": jb_stat,
            "JB P-value": jb_p,
        }
    return pd.DataFrame(records)


def ols_diagnostics(results: Any, nlags: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    resid = np.asarray(results.resid)
    exog = np.asarray(results.model.exog)

    rows.append(("Durbin-Watson", durbin_watson(resid), np.nan, "Descriptive statistic"))

    jb_stat, jb_p, _, _ = jarque_bera(resid)
    rows.append(("Jarque-Bera normality", jb_stat, jb_p, "Residuals are normally distributed"))

    try:
        lm, lm_p, fval, f_p = het_breuschpagan(resid, exog)
        rows.append(("Breusch-Pagan LM", lm, lm_p, "Homoskedasticity"))
        rows.append(("Breusch-Pagan F", fval, f_p, "Homoskedasticity"))
    except Exception:
        pass

    try:
        lm, lm_p, fval, f_p = het_white(resid, exog)
        rows.append(("White LM", lm, lm_p, "Homoskedasticity"))
        rows.append(("White F", fval, f_p, "Homoskedasticity"))
    except Exception:
        pass

    try:
        lm, lm_p, fval, f_p = acorr_breusch_godfrey(results, nlags=nlags)
        rows.append(("Breusch-Godfrey LM", lm, lm_p, "No serial correlation"))
        rows.append(("Breusch-Godfrey F", fval, f_p, "No serial correlation"))
    except Exception:
        pass

    try:
        lm, lm_p, fval, f_p = het_arch(resid, nlags=nlags)
        rows.append(("ARCH LM", lm, lm_p, "No ARCH effects"))
        rows.append(("ARCH F", fval, f_p, "No ARCH effects"))
    except Exception:
        pass

    try:
        reset = linear_reset(results, power=2, use_f=True)
        rows.append(("Ramsey RESET", float(reset.fvalue), float(reset.pvalue), "Correct functional form"))
    except Exception:
        pass

    diagnostic = pd.DataFrame(rows, columns=["Test", "Statistic", "P-value", "Null hypothesis"])
    diagnostic["Decision at 5%"] = np.where(
        diagnostic["P-value"].isna(),
        "Not applicable",
        np.where(diagnostic["P-value"] >= 0.05, "Do not reject null", "Reject null"),
    )

    vif_rows = []
    names = list(results.model.exog_names)
    for index, name in enumerate(names):
        try:
            vif_rows.append((name, variance_inflation_factor(exog, index)))
        except Exception:
            vif_rows.append((name, np.nan))
    vif = pd.DataFrame(vif_rows, columns=["Variable", "VIF"])
    return diagnostic, vif


def require_data() -> pd.DataFrame | None:
    df = st.session_state.df
    if df is None:
        st.info("Upload a dataset in the Data tab first.")
        return None
    return df


def unit_root_result(
    series: pd.Series,
    method: str,
    trend: str,
    max_lag: int | None,
    autolag: str,
) -> dict[str, Any]:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 10:
        raise ValueError("At least 10 valid observations are required.")

    if method == "ADF":
        result = adfuller(
            series,
            maxlag=max_lag,
            regression=trend,
            autolag=None if autolag == "None" else autolag,
        )
        return {
            "Statistic": result[0],
            "P-value": result[1],
            "Lags": result[2],
            "Observations": result[3],
            "1% Critical": result[4].get("1%"),
            "5% Critical": result[4].get("5%"),
            "10% Critical": result[4].get("10%"),
        }

    if method == "KPSS":
        kpss_trend = "ct" if trend in {"ct", "ctt"} else "c"
        stat, pvalue, lags, critical = kpss(
            series,
            regression=kpss_trend,
            nlags="auto" if max_lag is None else max_lag,
        )
        return {
            "Statistic": stat,
            "P-value": pvalue,
            "Lags": lags,
            "Observations": len(series),
            "1% Critical": critical.get("1%"),
            "5% Critical": critical.get("5%"),
            "10% Critical": critical.get("10%"),
        }

    if not ARCH_AVAILABLE:
        raise ImportError(f"The arch package is unavailable: {ARCH_IMPORT_ERROR}")

    arch_trend = "n" if trend == "n" else ("ct" if trend in {"ct", "ctt"} else "c")
    if method == "Phillips-Perron":
        test = PhillipsPerron(series, lags=max_lag, trend=arch_trend)
    elif method == "DF-GLS":
        if arch_trend == "n":
            arch_trend = "c"
        test = DFGLS(series, lags=max_lag, trend=arch_trend)
    elif method == "Zivot-Andrews":
        if arch_trend == "n":
            arch_trend = "c"
        test = ZivotAndrews(series, lags=max_lag, trend=arch_trend)
    else:
        raise ValueError(f"Unknown method: {method}")

    critical = test.critical_values
    return {
        "Statistic": test.stat,
        "P-value": test.pvalue,
        "Lags": getattr(test, "lags", np.nan),
        "Observations": test.nobs,
        "1% Critical": critical.get("1%"),
        "5% Critical": critical.get("5%"),
        "10% Critical": critical.get("10%"),
    }


init_state()

st.title("📊 Econometrics Studio")
st.caption(
    "A no-terminal interface for data preparation, econometric estimation, diagnostics, "
    "and reproducible Python-code export."
)

with st.sidebar:
    st.header("Project")
    st.write(f"App version: **{APP_VERSION}**")
    st.write(f"Python: **{platform.python_version()}**")
    st.toggle(
        "Dark mode",
        key="dark_mode",
        help="Switch between the default light appearance and a dark interface.",
    )

    if st.session_state.df is not None:
        st.success(
            f"{len(st.session_state.df):,} rows × "
            f"{len(st.session_state.df.columns):,} columns"
        )
        st.caption(st.session_state.source_filename or "Uploaded dataset")

    st.subheader("Optional engines")
    st.write("arch:", "✅" if ARCH_AVAILABLE else "❌")
    st.write("linearmodels:", "✅" if LINEARMODELS_AVAILABLE else "❌")

    if st.button("Reset entire project", type="secondary", use_container_width=True):
        clear_project()
        st.rerun()

    with st.expander("Analysis history"):
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                st.caption(f"{item['time']} — {item['action']}")
        else:
            st.caption("No analysis has been run.")

apply_dynamic_theme(bool(st.session_state.dark_mode))

if notice := st.session_state.pop("local_clear_notice", None):
    st.success(notice)

tabs = st.tabs(
    [
        "1. Data",
        "2. Transform",
        "3. Descriptive",
        "4. Regression",
        "5. Time Series",
        "6. Panel & IV",
        "7. Volatility",
        "8. Export",
    ]
)

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
with tabs[0]:
    st.header("Data manager")
    uploaded = st.file_uploader(
        "Upload CSV, Excel or Stata data",
        type=["csv", "xlsx", "xls", "dta"],
        key="main_upload",
    )

    if uploaded is not None:
        raw = uploaded.getvalue()
        signature = hashlib.sha256(raw).hexdigest()
        extension = uploaded.name.rsplit(".", 1)[-1].lower()
        sheet_name: str | int = 0

        if extension in {"xlsx", "xls"}:
            workbook = pd.ExcelFile(io.BytesIO(raw))
            sheet_name = st.selectbox("Excel sheet", workbook.sheet_names)
            signature = f"{signature}:{sheet_name}"

        if signature != st.session_state.file_signature:
            try:
                if extension == "csv":
                    loaded = pd.read_csv(io.BytesIO(raw))
                elif extension in {"xlsx", "xls"}:
                    loaded = pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name)
                elif extension == "dta":
                    loaded = pd.read_stata(io.BytesIO(raw))
                else:
                    raise ValueError("Unsupported file type.")

                loaded.columns = [str(column).strip() for column in loaded.columns]
                st.session_state.df = loaded.copy()
                st.session_state.original_df = loaded.copy()
                st.session_state.file_signature = signature
                st.session_state.source_filename = uploaded.name
                st.session_state.history = []
                st.session_state.code_blocks = []
                st.session_state.results = {}
                st.session_state.settings_log = []
                add_history(f"Uploaded {uploaded.name}")
                st.success("Dataset loaded.")
            except Exception as exc:
                display_exception(exc)

    df = require_data()
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Numeric columns", len(numeric_columns(df)))
        c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        st.subheader("Preview")
        st.dataframe(df, use_container_width=True, height=420)

        st.subheader("Rename variables")
        st.caption(
            "Rename one variable at a time. Recorded code, settings and saved "
            "result labels are updated to use the new name."
        )
        with st.form("rename_variable_form", clear_on_submit=True):
            rename_from = st.selectbox(
                "Current variable name",
                list(df.columns),
                key="rename_from_variable",
            )
            rename_to = st.text_input(
                "New variable name",
                placeholder="Enter a unique variable name",
            )
            rename_submitted = st.form_submit_button(
                "Rename variable",
                type="primary",
            )

        if rename_submitted:
            try:
                clean_new_name = rename_to.strip()
                if not clean_new_name:
                    raise ValueError("Enter a new variable name.")
                if clean_new_name == rename_from:
                    raise ValueError("The new name is the same as the current name.")
                if clean_new_name in df.columns:
                    raise ValueError(
                        f"A variable named {clean_new_name!r} already exists."
                    )
                if "\n" in clean_new_name or "\r" in clean_new_name:
                    raise ValueError("Variable names cannot contain line breaks.")

                st.session_state["_active_registration_slot"] = "rename_variable"
                rename_variable_everywhere(rename_from, clean_new_name)
                st.session_state["data_notice"] = (
                    f"Renamed {rename_from!r} to {clean_new_name!r}."
                )
                st.rerun()
            except Exception as exc:
                display_exception(exc)

        if notice := st.session_state.pop("data_notice", None):
            st.success(notice)

        render_slot_clear(
            "rename_variable",
            "Clear latest rename from export",
            key="clear_rename_variable_data_tab",
        )

        info_col, missing_col = st.columns(2)
        with info_col:
            st.subheader("Column information")
            info = pd.DataFrame(
                {
                    "Data type": df.dtypes.astype(str),
                    "Non-missing": df.notna().sum(),
                    "Unique": df.nunique(dropna=True),
                }
            )
            st.dataframe(info, use_container_width=True)

        with missing_col:
            st.subheader("Missing values")
            missing = pd.DataFrame(
                {
                    "Missing": df.isna().sum(),
                    "Percent": df.isna().mean() * 100,
                }
            ).sort_values("Missing", ascending=False)
            st.dataframe(missing, use_container_width=True)

        st.subheader("Sort and sample")
        sort_col, ascending = st.columns([3, 1])
        sort_by = sort_col.selectbox("Sort by", ["None"] + list(df.columns))
        ascending_value = ascending.checkbox("Ascending", value=True)
        if st.button("Apply sorting"):
            if sort_by != "None":
                st.session_state.df = df.sort_values(sort_by, ascending=ascending_value).reset_index(drop=True)
                add_history(f"Sorted data by {sort_by}")
                st.rerun()

        if st.button("Restore originally uploaded data"):
            st.session_state.df = st.session_state.original_df.copy()
            add_history("Restored original uploaded data")
            st.rerun()

        st.download_button(
            "Download current dataset as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=DEFAULT_DATA_FILE,
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# 2. TRANSFORM
# ---------------------------------------------------------------------------
with tabs[1]:
    st.header("Variable transformations")
    df = require_data()
    if df is not None:
        numeric = numeric_columns(df)
        if not numeric:
            st.warning("No numeric variables are available.")
        else:
            operation = st.selectbox(
                "Transformation",
                [
                    "Natural logarithm",
                    "Log one plus x",
                    "Signed logarithm",
                    "Inverse hyperbolic sine",
                    "First difference",
                    "Percentage change",
                    "Growth rate (%)",
                    "Lag",
                    "Lead",
                    "Standardize (z-score)",
                    "Min-max scale",
                    "Square",
                    "Interaction",
                ],
            )
            source = st.selectbox("Source variable", numeric)
            second = None
            periods = 1
            if operation in {"Lag", "Lead", "First difference", "Percentage change", "Growth rate (%)"}:
                periods = st.number_input("Periods", min_value=1, max_value=100, value=1, step=1)
            if operation == "Interaction":
                second = st.selectbox("Second variable", [c for c in numeric if c != source])

            default_prefix = {
                "Natural logarithm": "LN",
                "Log one plus x": "LN1P",
                "Signed logarithm": "SLOG",
                "Inverse hyperbolic sine": "IHS",
                "First difference": "D",
                "Percentage change": "PCT",
                "Growth rate (%)": "GR",
                "Lag": "L",
                "Lead": "F",
                "Standardize (z-score)": "Z",
                "Min-max scale": "MM",
                "Square": "SQ",
                "Interaction": "INT",
            }[operation]
            default_name = (
                f"{default_prefix}_{source}_{second}" if second else f"{default_prefix}_{source}"
            )
            new_name = st.text_input("New variable name", value=default_name)

            if analysis_action_buttons(
                "Apply transformation",
                "transformation",
                clear_label="Clear latest transformation",
            ):
                try:
                    series = pd.to_numeric(df[source], errors="coerce")
                    code = ""
                    if operation == "Natural logarithm":
                        transformed = np.where(series > 0, np.log(series), np.nan)
                        code = f"data[{new_name!r}] = np.where(data[{source!r}] > 0, np.log(data[{source!r}]), np.nan)"
                    elif operation == "Log one plus x":
                        transformed = np.where(series > -1, np.log1p(series), np.nan)
                        code = f"data[{new_name!r}] = np.where(data[{source!r}] > -1, np.log1p(data[{source!r}]), np.nan)"
                    elif operation == "Signed logarithm":
                        transformed = np.sign(series) * np.log1p(np.abs(series))
                        code = f"data[{new_name!r}] = np.sign(data[{source!r}]) * np.log1p(np.abs(data[{source!r}]))"
                    elif operation == "Inverse hyperbolic sine":
                        transformed = np.arcsinh(series)
                        code = f"data[{new_name!r}] = np.arcsinh(data[{source!r}])"
                    elif operation == "First difference":
                        transformed = series.diff(int(periods))
                        code = f"data[{new_name!r}] = data[{source!r}].diff({int(periods)})"
                    elif operation == "Percentage change":
                        transformed = series.pct_change(int(periods))
                        code = f"data[{new_name!r}] = data[{source!r}].pct_change({int(periods)})"
                    elif operation == "Growth rate (%)":
                        transformed = series.pct_change(int(periods)) * 100
                        code = f"data[{new_name!r}] = data[{source!r}].pct_change({int(periods)}) * 100"
                    elif operation == "Lag":
                        transformed = series.shift(int(periods))
                        code = f"data[{new_name!r}] = data[{source!r}].shift({int(periods)})"
                    elif operation == "Lead":
                        transformed = series.shift(-int(periods))
                        code = f"data[{new_name!r}] = data[{source!r}].shift(-{int(periods)})"
                    elif operation == "Standardize (z-score)":
                        transformed = (series - series.mean()) / series.std(ddof=1)
                        code = f"data[{new_name!r}] = (data[{source!r}] - data[{source!r}].mean()) / data[{source!r}].std(ddof=1)"
                    elif operation == "Min-max scale":
                        denominator = series.max() - series.min()
                        if denominator == 0:
                            raise ValueError("The selected variable is constant.")
                        transformed = (series - series.min()) / denominator
                        code = f"data[{new_name!r}] = (data[{source!r}] - data[{source!r}].min()) / (data[{source!r}].max() - data[{source!r}].min())"
                    elif operation == "Square":
                        transformed = series ** 2
                        code = f"data[{new_name!r}] = data[{source!r}] ** 2"
                    elif operation == "Interaction":
                        transformed = series * pd.to_numeric(df[second], errors="coerce")
                        code = f"data[{new_name!r}] = data[{source!r}] * data[{second!r}]"
                    else:
                        raise ValueError("Unknown transformation.")

                    st.session_state.df[new_name] = transformed
                    register_output(
                        make_unique_name(f"Transform_{new_name}"),
                        pd.DataFrame({new_name: transformed}),
                        code,
                        {"operation": operation, "source": source, "new_name": new_name},
                    )
                    st.success(f"Created {new_name}.")
                    st.rerun()
                except Exception as exc:
                    display_exception(exc)

        st.divider()
        st.subheader("Missing-value treatment")
        missing_columns = st.multiselect(
            "Columns",
            list(df.columns),
            default=[c for c in df.columns if df[c].isna().any()][:5],
        )
        treatment = st.selectbox(
            "Treatment",
            ["Drop rows", "Fill with mean", "Fill with median", "Forward fill", "Backward fill", "Linear interpolation"],
        )
        if analysis_action_buttons(
            "Apply missing-value treatment",
            "missing_values",
            clear_label="Clear latest missing-value operation",
        ):
            try:
                if not missing_columns:
                    raise ValueError("Select at least one column.")
                work = st.session_state.df.copy()
                if treatment == "Drop rows":
                    work = work.dropna(subset=missing_columns).reset_index(drop=True)
                    code = f"data = data.dropna(subset={missing_columns!r}).reset_index(drop=True)"
                elif treatment == "Fill with mean":
                    for c in missing_columns:
                        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(pd.to_numeric(work[c], errors="coerce").mean())
                    code = f"for column in {missing_columns!r}:\n    data[column] = pd.to_numeric(data[column], errors='coerce').fillna(pd.to_numeric(data[column], errors='coerce').mean())"
                elif treatment == "Fill with median":
                    for c in missing_columns:
                        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(pd.to_numeric(work[c], errors="coerce").median())
                    code = f"for column in {missing_columns!r}:\n    data[column] = pd.to_numeric(data[column], errors='coerce').fillna(pd.to_numeric(data[column], errors='coerce').median())"
                elif treatment == "Forward fill":
                    work[missing_columns] = work[missing_columns].ffill()
                    code = f"data[{missing_columns!r}] = data[{missing_columns!r}].ffill()"
                elif treatment == "Backward fill":
                    work[missing_columns] = work[missing_columns].bfill()
                    code = f"data[{missing_columns!r}] = data[{missing_columns!r}].bfill()"
                else:
                    work[missing_columns] = work[missing_columns].interpolate(method="linear")
                    code = f"data[{missing_columns!r}] = data[{missing_columns!r}].interpolate(method='linear')"

                st.session_state.df = work
                register_output(
                    make_unique_name("Missing_values"),
                    work[missing_columns].isna().sum().to_frame("Remaining missing"),
                    code,
                    {"columns": missing_columns, "treatment": treatment},
                )
                st.success("Missing-value treatment applied.")
                st.rerun()
            except Exception as exc:
                display_exception(exc)

# ---------------------------------------------------------------------------
# 3. DESCRIPTIVE
# ---------------------------------------------------------------------------
with tabs[2]:
    st.header("Descriptive analysis")
    df = require_data()
    if df is not None:
        numeric = numeric_columns(df)
        selected = st.multiselect("Variables", numeric, default=numeric[: min(6, len(numeric))])

        col_a, col_b = st.columns(2)
        with col_a:
            if analysis_action_buttons(
                "Run descriptive statistics",
                "descriptive_statistics",
                clear_label="Clear latest descriptive result",
            ):
                try:
                    if not selected:
                        raise ValueError("Select at least one variable.")
                    frame = clean_numeric_frame(df, selected)
                    table = descriptive_statistics(frame)
                    name = make_unique_name("Descriptive_statistics")
                    code = f"""
variables = {selected!r}
sample = data[variables].apply(pd.to_numeric, errors="coerce")
descriptive = sample.describe().T
descriptive["Median"] = sample.median()
descriptive["Variance"] = sample.var(ddof=1)
descriptive["Skewness"] = sample.skew()
descriptive["Kurtosis"] = sample.kurtosis() + 3
print(descriptive)
"""
                    register_output(name, table, code, {"variables": selected})
                    st.session_state["last_descriptive"] = table
                except Exception as exc:
                    display_exception(exc)

        with col_b:
            corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
            if analysis_action_buttons(
                "Run correlation matrix",
                "correlation",
                clear_label="Clear latest correlation result",
                primary=False,
            ):
                try:
                    if len(selected) < 2:
                        raise ValueError("Select at least two variables.")
                    frame = clean_numeric_frame(df, selected)
                    table = frame.corr(method=corr_method)
                    name = make_unique_name("Correlation")
                    code = f"""
variables = {selected!r}
sample = data[variables].apply(pd.to_numeric, errors="coerce").dropna()
correlation = sample.corr(method={corr_method!r})
print(correlation)
"""
                    register_output(name, table, code, {"variables": selected, "method": corr_method})
                    st.session_state["last_correlation"] = table
                except Exception as exc:
                    display_exception(exc)

        if "last_descriptive" in st.session_state:
            st.subheader("Descriptive statistics")
            st.dataframe(st.session_state.last_descriptive, use_container_width=True)

        if "last_correlation" in st.session_state:
            st.subheader("Correlation matrix")
            st.dataframe(st.session_state.last_correlation, use_container_width=True)

        st.subheader("Charts")
        chart_type = st.selectbox("Chart", ["Time/line plot", "Histogram", "Scatter plot", "Box plot"])
        if chart_type == "Scatter plot":
            x_chart = st.selectbox("X variable", numeric, key="scatter_x")
            y_chart = st.selectbox("Y variable", numeric, index=min(1, len(numeric)-1), key="scatter_y")
            if st.button("Draw chart"):
                fig, ax = plt.subplots()
                ax.scatter(df[x_chart], df[y_chart])
                ax.set_xlabel(x_chart)
                ax.set_ylabel(y_chart)
                ax.set_title(f"{y_chart} against {x_chart}")
                st.pyplot(fig)
                plt.close(fig)
        else:
            chart_vars = st.multiselect("Chart variables", numeric, default=numeric[:1], key="chart_vars")
            if st.button("Draw chart"):
                if not chart_vars:
                    st.warning("Select at least one variable.")
                else:
                    fig, ax = plt.subplots()
                    if chart_type == "Time/line plot":
                        df[chart_vars].plot(ax=ax)
                    elif chart_type == "Histogram":
                        df[chart_vars].plot.hist(ax=ax, alpha=0.6)
                    else:
                        df[chart_vars].plot.box(ax=ax)
                    ax.set_title(chart_type)
                    st.pyplot(fig)
                    plt.close(fig)

# ---------------------------------------------------------------------------
# 4. REGRESSION
# ---------------------------------------------------------------------------
with tabs[3]:
    st.header("Cross-sectional and general regression")
    df = require_data()
    if df is not None:
        numeric = numeric_columns(df)
        method = st.selectbox(
            "Estimator",
            [
                "OLS",
                "WLS",
                "GLS",
                "Robust linear model",
                "Quantile regression",
                "Logit",
                "Probit",
                "Poisson",
                "Negative binomial",
            ],
        )
        y = st.selectbox("Dependent variable", numeric, key="reg_y")
        x = st.multiselect("Explanatory variables", [c for c in numeric if c != y], key="reg_x")
        add_constant = st.checkbox("Include intercept", value=True, key="reg_const")
        cov_type = st.selectbox(
            "Covariance estimator",
            ["nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC"],
            help="HAC is available for OLS/WLS/GLS and requires a maximum lag.",
        )
        hac_lags = st.number_input("HAC/diagnostic maximum lag", 1, 50, 1)
        weight_column = None
        quantile = 0.5
        if method == "WLS":
            weight_column = st.selectbox("Weight variable", [c for c in numeric if c not in [y] + x])
        if method == "Quantile regression":
            quantile = st.slider("Quantile", 0.05, 0.95, 0.50, 0.05)

        if analysis_action_buttons(
            "Estimate regression",
            "regression",
            clear_label="Clear latest regression",
        ):
            try:
                if not x:
                    raise ValueError("Select at least one explanatory variable.")
                columns = [y] + x + ([weight_column] if weight_column else [])
                sample = clean_numeric_frame(df, columns)
                y_data = sample[y]
                x_data = sample[x]
                if add_constant:
                    x_data = sm.add_constant(x_data, has_constant="add")

                fit_kwargs = {}
                if cov_type != "nonrobust" and method in {"OLS", "WLS", "GLS"}:
                    fit_kwargs["cov_type"] = cov_type
                    if cov_type == "HAC":
                        fit_kwargs["cov_kwds"] = {"maxlags": int(hac_lags)}

                if method == "OLS":
                    model = sm.OLS(y_data, x_data)
                    results = model.fit(**fit_kwargs)
                    model_code = "sm.OLS(y_data, x_data)"
                elif method == "WLS":
                    weights = sample[weight_column]
                    if (weights <= 0).any():
                        raise ValueError("WLS weights must be strictly positive.")
                    model = sm.WLS(y_data, x_data, weights=weights)
                    results = model.fit(**fit_kwargs)
                    model_code = f"sm.WLS(y_data, x_data, weights=sample[{weight_column!r}])"
                elif method == "GLS":
                    model = sm.GLS(y_data, x_data)
                    results = model.fit(**fit_kwargs)
                    model_code = "sm.GLS(y_data, x_data)"
                elif method == "Robust linear model":
                    model = sm.RLM(y_data, x_data, M=sm.robust.norms.HuberT())
                    results = model.fit()
                    model_code = "sm.RLM(y_data, x_data, M=sm.robust.norms.HuberT())"
                elif method == "Quantile regression":
                    model = sm.QuantReg(y_data, x_data)
                    results = model.fit(q=float(quantile))
                    model_code = f"sm.QuantReg(y_data, x_data)"
                elif method == "Logit":
                    if not set(pd.unique(y_data)).issubset({0, 1}):
                        raise ValueError("Logit requires a binary dependent variable coded 0 and 1.")
                    model = sm.Logit(y_data, x_data)
                    results = model.fit(disp=False)
                    model_code = "sm.Logit(y_data, x_data)"
                elif method == "Probit":
                    if not set(pd.unique(y_data)).issubset({0, 1}):
                        raise ValueError("Probit requires a binary dependent variable coded 0 and 1.")
                    model = sm.Probit(y_data, x_data)
                    results = model.fit(disp=False)
                    model_code = "sm.Probit(y_data, x_data)"
                elif method == "Poisson":
                    model = sm.Poisson(y_data, x_data)
                    results = model.fit(disp=False)
                    model_code = "sm.Poisson(y_data, x_data)"
                elif method == "Negative binomial":
                    model = sm.NegativeBinomial(y_data, x_data)
                    results = model.fit(disp=False)
                    model_code = "sm.NegativeBinomial(y_data, x_data)"
                else:
                    raise ValueError("Unknown estimator.")

                table = parameter_table(results)
                summary = summary_text(results)
                st.subheader("Coefficient table")
                st.dataframe(table, use_container_width=True)
                st.subheader("Model summary")
                st.text(summary)

                code = f"""
variables = {[y] + x + ([weight_column] if weight_column else [])!r}
sample = data[variables].apply(pd.to_numeric, errors="coerce").dropna()
y_data = sample[{y!r}]
x_data = sample[{x!r}]
{"x_data = sm.add_constant(x_data, has_constant='add')" if add_constant else ""}
model = {model_code}
"""
                if method == "Quantile regression":
                    code += f"\nresults = model.fit(q={float(quantile)!r})"
                elif method in {"Logit", "Probit", "Poisson", "Negative binomial"}:
                    code += "\nresults = model.fit(disp=False)"
                elif fit_kwargs:
                    code += f"\nresults = model.fit(**{fit_kwargs!r})"
                else:
                    code += "\nresults = model.fit()"
                code += "\nprint(results.summary())"

                result_name = make_unique_name(method.replace(" ", "_"))
                register_output(
                    result_name,
                    table,
                    code,
                    {
                        "method": method,
                        "dependent": y,
                        "explanatory": x,
                        "constant": add_constant,
                        "cov_type": cov_type,
                        "observations": len(sample),
                    },
                    summary,
                )
                st.session_state.results[result_name + "_summary"] = summary

                if method in {"OLS", "WLS", "GLS"}:
                    diagnostic, vif = ols_diagnostics(results, int(hac_lags))
                    st.subheader("Diagnostics")
                    st.dataframe(diagnostic, use_container_width=True)
                    st.subheader("Variance inflation factors")
                    st.dataframe(vif, use_container_width=True)
                    st.session_state.results[result_name + "_diagnostics"] = diagnostic
                    st.session_state.results[result_name + "_vif"] = vif
            except Exception as exc:
                display_exception(exc)

# ---------------------------------------------------------------------------
# 5. TIME SERIES
# ---------------------------------------------------------------------------
with tabs[4]:
    st.header("Time-series econometrics")
    df = require_data()
    if df is not None:
        numeric = numeric_columns(df)
        ts_tabs = st.tabs(["Unit roots", "ARDL/UECM", "ARIMA", "VAR/VECM", "Granger"])

        with ts_tabs[0]:
            st.subheader("Unit-root and stationarity tests")
            test_method = st.selectbox(
                "Test",
                ["ADF", "Phillips-Perron", "KPSS", "DF-GLS", "Zivot-Andrews"],
            )
            test_vars = st.multiselect("Variables", numeric, default=numeric[: min(3, len(numeric))], key="ur_vars")
            difference = st.selectbox("Transformation before testing", ["Level", "First difference", "Second difference"])
            trend = st.selectbox(
                "Deterministic terms",
                ["c", "ct", "ctt", "n"],
                format_func=lambda v: {
                    "c": "Constant",
                    "ct": "Constant and trend",
                    "ctt": "Constant, trend and quadratic trend",
                    "n": "None",
                }[v],
            )
            auto_lag = st.selectbox("ADF autolag", ["AIC", "BIC", "t-stat", "None"])
            automatic_lag = st.checkbox("Select lag/bandwidth automatically", value=True)
            max_lag = None if automatic_lag else int(st.number_input("Lag/bandwidth", 0, 100, 1))

            if analysis_action_buttons(
                "Run unit-root tests",
                "unit_root",
                clear_label="Clear latest unit-root result",
            ):
                try:
                    if not test_vars:
                        raise ValueError("Select at least one variable.")
                    rows = []
                    diff_order = {"Level": 0, "First difference": 1, "Second difference": 2}[difference]
                    for variable in test_vars:
                        series = pd.to_numeric(df[variable], errors="coerce")
                        for _ in range(diff_order):
                            series = series.diff()
                        result = unit_root_result(series, test_method, trend, max_lag, auto_lag)
                        result["Variable"] = variable
                        result["Form"] = difference
                        if test_method == "KPSS":
                            result["Decision at 5%"] = (
                                "Stationary" if result["P-value"] >= 0.05 else "Non-stationary"
                            )
                        else:
                            result["Decision at 5%"] = (
                                "Stationary" if result["P-value"] < 0.05 else "Non-stationary"
                            )
                        rows.append(result)
                    table = pd.DataFrame(rows).set_index(["Variable", "Form"])
                    st.dataframe(table, use_container_width=True)

                    code = f"""
variables = {test_vars!r}
difference_order = {diff_order}
unit_root_rows = []
for variable in variables:
    series = pd.to_numeric(data[variable], errors="coerce").dropna()
    for _ in range(difference_order):
        series = series.diff().dropna()
"""
                    if test_method == "ADF":
                        code += f"""
    test = adfuller(
        series,
        maxlag={max_lag!r},
        regression={trend!r},
        autolag={None if auto_lag == "None" else auto_lag!r},
    )
    unit_root_rows.append({{
        "Variable": variable,
        "Statistic": test[0],
        "P-value": test[1],
        "Lags": test[2],
        "Observations": test[3],
    }})
"""
                    elif test_method == "KPSS":
                        kpss_trend = "ct" if trend in {"ct", "ctt"} else "c"
                        code += f"""
    test = kpss(series, regression={kpss_trend!r}, nlags={"'auto'" if max_lag is None else max_lag})
    unit_root_rows.append({{
        "Variable": variable,
        "Statistic": test[0],
        "P-value": test[1],
        "Lags": test[2],
    }})
"""
                    else:
                        class_name = {
                            "Phillips-Perron": "PhillipsPerron",
                            "DF-GLS": "DFGLS",
                            "Zivot-Andrews": "ZivotAndrews",
                        }[test_method]
                        arch_trend = "n" if trend == "n" else ("ct" if trend in {"ct", "ctt"} else "c")
                        if class_name in {"DFGLS", "ZivotAndrews"} and arch_trend == "n":
                            arch_trend = "c"
                        code += f"""
    test = {class_name}(series, lags={max_lag!r}, trend={arch_trend!r})
    unit_root_rows.append({{
        "Variable": variable,
        "Statistic": test.stat,
        "P-value": test.pvalue,
        "Lags": getattr(test, "lags", np.nan),
        "Observations": test.nobs,
    }})
"""
                    code += "\nunit_root_results = pd.DataFrame(unit_root_rows)\nprint(unit_root_results)"
                    register_output(
                        make_unique_name(test_method.replace("-", "_")),
                        table,
                        code,
                        {
                            "test": test_method,
                            "variables": test_vars,
                            "difference": difference,
                            "trend": trend,
                            "lag": max_lag,
                        },
                    )
                except Exception as exc:
                    display_exception(exc)

        with ts_tabs[1]:
            st.subheader("ARDL, UECM and PSS bounds test")
            y = st.selectbox("Dependent variable", numeric, key="ardl_y")
            x = st.multiselect("Explanatory variables", [c for c in numeric if c != y], key="ardl_x")
            selection_mode = st.radio("Lag specification", ["Automatic selection", "Manual"], horizontal=True)
            trend = st.selectbox("Trend", ["c", "n", "ct", "t"], key="ardl_trend")
            causal = st.checkbox("Causal ARDL: exclude contemporaneous X terms", value=False)

            order_dict: dict[str, int] = {}
            if selection_mode == "Automatic selection":
                max_p = int(st.number_input("Maximum dependent-variable lag", 1, 12, 2))
                max_q = int(st.number_input("Maximum explanatory-variable lag", 1, 12, 2))
                ic = st.selectbox("Selection criterion", ["bic", "aic"])
            else:
                p_lag = int(st.number_input("Dependent-variable lag", 1, 20, 1))
                for variable in x:
                    order_dict[variable] = int(
                        st.number_input(
                            f"Lag order for {variable}",
                            1,
                            20,
                            1,
                            key=f"ardl_lag_{variable}",
                        )
                    )
                ic = "manual"

            run_bounds = st.checkbox("Estimate UECM and bounds test", value=True)
            bounds_case = st.selectbox("PSS bounds-test case", [1, 2, 3, 4, 5], index=2)

            if analysis_action_buttons(
                "Estimate ARDL",
                "ardl",
                clear_label="Clear latest ARDL result",
            ):
                try:
                    if not x:
                        raise ValueError("Select at least one explanatory variable.")
                    sample = clean_numeric_frame(df, [y] + x)
                    endog = sample[y]
                    exog = sample[x]

                    if selection_mode == "Automatic selection":
                        selection = ardl_select_order(
                            endog,
                            maxlag=max_p,
                            exog=exog,
                            maxorder=max_q,
                            trend=trend,
                            causal=causal,
                            ic=ic,
                        )
                        model = selection.model
                        selection_code = f"""
selection = ardl_select_order(
    endog,
    maxlag={max_p},
    exog=exog,
    maxorder={max_q},
    trend={trend!r},
    causal={causal!r},
    ic={ic!r},
)
model = selection.model
"""
                    else:
                        model = ARDL(
                            endog=endog,
                            lags=p_lag,
                            exog=exog,
                            order=order_dict,
                            trend=trend,
                            causal=causal,
                        )
                        selection_code = f"""
model = ARDL(
    endog=endog,
    lags={p_lag},
    exog=exog,
    order={order_dict!r},
    trend={trend!r},
    causal={causal!r},
)
"""
                    results = model.fit()
                    table = parameter_table(results)
                    text = summary_text(results)
                    st.write("Selected order:", model.ardl_order)
                    st.dataframe(table, use_container_width=True)
                    st.text(text)

                    code = f"""
sample = data[{[y] + x!r}].apply(pd.to_numeric, errors="coerce").dropna()
endog = sample[{y!r}]
exog = sample[{x!r}]
{selection_code}
results = model.fit()
print("ARDL order:", model.ardl_order)
print(results.summary())
"""
                    result_name = make_unique_name("ARDL")
                    register_output(
                        result_name,
                        table,
                        code,
                        {
                            "dependent": y,
                            "explanatory": x,
                            "order": model.ardl_order,
                            "mode": selection_mode,
                            "trend": trend,
                            "causal": causal,
                        },
                        text,
                    )
                    st.session_state.results[result_name + "_summary"] = text

                    if run_bounds:
                        try:
                            uecm_model = UECM.from_ardl(model)
                            uecm_results = uecm_model.fit()
                            bounds = uecm_results.bounds_test(case=int(bounds_case))
                            bounds_table = pd.DataFrame(
                                {
                                    "F-statistic": [bounds.stat],
                                    "Lower-bound p-value": [bounds.p_values["lower"]],
                                    "Upper-bound p-value": [bounds.p_values["upper"]],
                                }
                            )
                            st.subheader("Bounds test")
                            st.dataframe(bounds_table, use_container_width=True)
                            st.subheader("Bounds critical values")
                            st.dataframe(bounds.crit_vals, use_container_width=True)
                            st.subheader("Normalized cointegrating relationship")
                            st.text(uecm_results.ci_summary().as_text())
                            st.subheader("UECM results")
                            st.text(summary_text(uecm_results))

                            st.session_state.results[result_name + "_bounds"] = bounds_table
                            st.session_state.results[result_name + "_bounds_critical"] = bounds.crit_vals
                            st.session_state.results[result_name + "_UECM_summary"] = summary_text(uecm_results)
                            st.session_state.code_blocks.append(
                                {
                                    "name": result_name + "_UECM_bounds",
                                    "code": textwrap.dedent(
                                        f"""
uecm_model = UECM.from_ardl(model)
uecm_results = uecm_model.fit()
bounds = uecm_results.bounds_test(case={int(bounds_case)})
print(uecm_results.summary())
print(uecm_results.ci_summary())
print(bounds)
"""
                                    ).strip(),
                                }
                            )
                        except Exception as exc:
                            st.warning(
                                "The ARDL was estimated, but UECM conversion or bounds testing failed. "
                                "UECM requires every included variable to have at least one contiguous lag."
                            )
                            st.caption(str(exc))
                except Exception as exc:
                    display_exception(exc)

        with ts_tabs[2]:
            st.subheader("ARIMA and seasonal ARIMA")
            y = st.selectbox("Series", numeric, key="arima_y")
            exog_vars = st.multiselect("Optional exogenous variables", [c for c in numeric if c != y], key="arima_x")
            p = int(st.number_input("AR order p", 0, 20, 1))
            d = int(st.number_input("Difference order d", 0, 5, 0))
            q = int(st.number_input("MA order q", 0, 20, 0))
            seasonal = st.checkbox("Use seasonal order")
            if seasonal:
                P = int(st.number_input("Seasonal AR P", 0, 10, 0))
                D = int(st.number_input("Seasonal difference D", 0, 3, 0))
                Q = int(st.number_input("Seasonal MA Q", 0, 10, 0))
                s = int(st.number_input("Seasonal period s", 2, 365, 4))
                seasonal_order = (P, D, Q, s)
            else:
                seasonal_order = (0, 0, 0, 0)
            arima_trend = st.selectbox("Trend", ["None", "n", "c", "t", "ct"])

            if analysis_action_buttons(
                "Estimate ARIMA",
                "arima",
                clear_label="Clear latest ARIMA result",
            ):
                try:
                    columns = [y] + exog_vars
                    sample = clean_numeric_frame(df, columns)
                    exog = sample[exog_vars] if exog_vars else None
                    trend_arg = None if arima_trend == "None" else arima_trend
                    model = ARIMA(
                        sample[y],
                        exog=exog,
                        order=(p, d, q),
                        seasonal_order=seasonal_order,
                        trend=trend_arg,
                    )
                    results = model.fit()
                    table = parameter_table(results)
                    text = summary_text(results)
                    st.dataframe(table, use_container_width=True)
                    st.text(text)

                    code = f"""
sample = data[{columns!r}].apply(pd.to_numeric, errors="coerce").dropna()
exog = sample[{exog_vars!r}] if {bool(exog_vars)!r} else None
model = ARIMA(
    sample[{y!r}],
    exog=exog,
    order={(p, d, q)!r},
    seasonal_order={seasonal_order!r},
    trend={trend_arg!r},
)
results = model.fit()
print(results.summary())
"""
                    name = make_unique_name("ARIMA")
                    register_output(
                        name,
                        table,
                        code,
                        {
                            "series": y,
                            "exog": exog_vars,
                            "order": (p, d, q),
                            "seasonal_order": seasonal_order,
                            "trend": trend_arg,
                        },
                        text,
                    )
                    st.session_state.results[name + "_summary"] = text
                except Exception as exc:
                    display_exception(exc)

        with ts_tabs[3]:
            st.subheader("VAR and VECM")
            multivariate_method = st.radio("Model", ["VAR", "Johansen test", "VECM"], horizontal=True)
            variables = st.multiselect("Variables", numeric, default=numeric[: min(3, len(numeric))], key="multi_ts_vars")
            if multivariate_method == "VAR":
                automatic = st.checkbox("Select VAR lag by information criterion", value=True)
                maxlags = int(st.number_input("Maximum lag", 1, 20, 2, key="var_lags"))
                var_ic = st.selectbox("Criterion", ["aic", "bic", "hqic", "fpe"])
                var_trend = st.selectbox("Trend", ["c", "ct", "ctt", "n"], key="var_trend")
            elif multivariate_method == "Johansen test":
                det_order = int(st.selectbox("Deterministic order", [-1, 0, 1], index=1))
                k_ar_diff = int(st.number_input("Lagged differences", 1, 20, 1, key="johansen_lags"))
            else:
                k_ar_diff = int(st.number_input("Lagged differences", 1, 20, 1, key="vecm_lags"))
                coint_rank = int(st.number_input("Cointegration rank", 1, max(1, len(variables)-1), 1))
                deterministic = st.selectbox("Deterministic specification", ["n", "co", "ci", "lo", "li", "colo", "cili"])

            if analysis_action_buttons(
                f"Run {multivariate_method}",
                "multivariate_time_series",
                clear_label="Clear latest VAR/VECM result",
            ):
                try:
                    if len(variables) < 2:
                        raise ValueError("Select at least two variables.")
                    sample = clean_numeric_frame(df, variables)
                    if multivariate_method == "VAR":
                        model = VAR(sample)
                        results = model.fit(
                            maxlags=maxlags,
                            ic=var_ic if automatic else None,
                            trend=var_trend,
                        )
                        text = summary_text(results)
                        roots = pd.DataFrame({"Root": results.roots, "Modulus": np.abs(results.roots)})
                        st.text(text)
                        st.dataframe(roots, use_container_width=True)
                        st.write("Stable:", results.is_stable(verbose=False))
                        code = f"""
sample = data[{variables!r}].apply(pd.to_numeric, errors="coerce").dropna()
model = VAR(sample)
results = model.fit(
    maxlags={maxlags},
    ic={var_ic!r} if {automatic!r} else None,
    trend={var_trend!r},
)
print(results.summary())
print("Stable:", results.is_stable())
"""
                        name = make_unique_name("VAR")
                        register_output(name, roots, code, {"variables": variables}, text)
                        st.session_state.results[name + "_summary"] = text
                    elif multivariate_method == "Johansen test":
                        result = coint_johansen(sample, det_order=det_order, k_ar_diff=k_ar_diff)
                        table = pd.DataFrame(
                            {
                                "Trace statistic": result.lr1,
                                "90% critical": result.cvt[:, 0],
                                "95% critical": result.cvt[:, 1],
                                "99% critical": result.cvt[:, 2],
                                "Max-eigen statistic": result.lr2,
                                "Max-eigen 90%": result.cvm[:, 0],
                                "Max-eigen 95%": result.cvm[:, 1],
                                "Max-eigen 99%": result.cvm[:, 2],
                            },
                            index=[f"r ≤ {i}" for i in range(len(variables))],
                        )
                        st.dataframe(table, use_container_width=True)
                        code = f"""
sample = data[{variables!r}].apply(pd.to_numeric, errors="coerce").dropna()
result = coint_johansen(sample, det_order={det_order}, k_ar_diff={k_ar_diff})
print(result.lr1)
print(result.cvt)
print(result.lr2)
print(result.cvm)
"""
                        register_output(make_unique_name("Johansen"), table, code, {"variables": variables})
                    else:
                        if coint_rank >= len(variables):
                            raise ValueError("Cointegration rank must be less than the number of variables.")
                        model = VECM(
                            sample,
                            k_ar_diff=k_ar_diff,
                            coint_rank=coint_rank,
                            deterministic=deterministic,
                        )
                        results = model.fit()
                        text = summary_text(results)
                        alpha = pd.DataFrame(results.alpha, index=variables)
                        beta = pd.DataFrame(results.beta, index=variables)
                        st.text(text)
                        st.subheader("Adjustment coefficients (alpha)")
                        st.dataframe(alpha, use_container_width=True)
                        st.subheader("Cointegrating vectors (beta)")
                        st.dataframe(beta, use_container_width=True)
                        code = f"""
sample = data[{variables!r}].apply(pd.to_numeric, errors="coerce").dropna()
model = VECM(
    sample,
    k_ar_diff={k_ar_diff},
    coint_rank={coint_rank},
    deterministic={deterministic!r},
)
results = model.fit()
print(results.summary())
"""
                        name = make_unique_name("VECM")
                        register_output(name, alpha, code, {"variables": variables}, text)
                        st.session_state.results[name + "_beta"] = beta
                        st.session_state.results[name + "_summary"] = text
                except Exception as exc:
                    display_exception(exc)

        with ts_tabs[4]:
            st.subheader("Pairwise Granger causality")
            target = st.selectbox("Target variable", numeric, key="granger_y")
            cause = st.selectbox("Potential causal variable", [c for c in numeric if c != target], key="granger_x")
            maxlag = int(st.number_input("Maximum lag", 1, 20, 2, key="granger_lags"))

            if analysis_action_buttons(
                "Run Granger causality",
                "granger",
                clear_label="Clear latest Granger result",
            ):
                try:
                    sample = clean_numeric_frame(df, [target, cause])
                    result = grangercausalitytests(sample[[target, cause]], maxlag=maxlag, verbose=False)
                    rows = []
                    for lag, values in result.items():
                        tests = values[0]
                        ssr_f = tests["ssr_ftest"]
                        ssr_chi = tests["ssr_chi2test"]
                        lr = tests["lrtest"]
                        params_f = tests["params_ftest"]
                        rows.append(
                            {
                                "Lag": lag,
                                "SSR F": ssr_f[0],
                                "SSR F p-value": ssr_f[1],
                                "SSR Chi-square": ssr_chi[0],
                                "SSR Chi-square p-value": ssr_chi[1],
                                "LR statistic": lr[0],
                                "LR p-value": lr[1],
                                "Parameter F": params_f[0],
                                "Parameter F p-value": params_f[1],
                            }
                        )
                    table = pd.DataFrame(rows).set_index("Lag")
                    st.dataframe(table, use_container_width=True)
                    code = f"""
sample = data[{[target, cause]!r}].apply(pd.to_numeric, errors="coerce").dropna()
granger_results = grangercausalitytests(
    sample[[{target!r}, {cause!r}]],
    maxlag={maxlag},
    verbose=False,
)
"""
                    register_output(
                        make_unique_name("Granger"),
                        table,
                        code,
                        {"target": target, "cause": cause, "maxlag": maxlag},
                    )
                except Exception as exc:
                    display_exception(exc)

# ---------------------------------------------------------------------------
# 6. PANEL AND IV
# ---------------------------------------------------------------------------
with tabs[5]:
    st.header("Panel-data and instrumental-variable models")
    df = require_data()
    if df is not None:
        panel_tabs = st.tabs(["Panel models", "Instrumental variables"])

        with panel_tabs[0]:
            if not LINEARMODELS_AVAILABLE:
                st.error(f"Install linearmodels to use panel estimators: {LINEARMODELS_IMPORT_ERROR}")
            else:
                numeric = numeric_columns(df)
                entity = st.selectbox("Entity identifier", list(df.columns), key="panel_entity")
                time = st.selectbox("Time identifier", [c for c in df.columns if c != entity], key="panel_time")
                y = st.selectbox("Dependent variable", [c for c in numeric if c not in [entity, time]], key="panel_y")
                x = st.multiselect(
                    "Explanatory variables",
                    [c for c in numeric if c not in [entity, time, y]],
                    key="panel_x",
                )
                panel_method = st.selectbox(
                    "Panel estimator",
                    ["Pooled OLS", "Fixed effects", "Random effects", "First difference", "Between", "Fama-MacBeth"],
                )
                entity_effects = st.checkbox("Entity effects", value=True)
                time_effects = st.checkbox("Time effects", value=False)
                panel_constant = st.checkbox("Include intercept", value=True)
                panel_cov = st.selectbox("Covariance", ["unadjusted", "robust", "clustered"])
                cluster_entity = st.checkbox("Cluster by entity", value=True)
                cluster_time = st.checkbox("Cluster by time", value=False)

                if analysis_action_buttons(
                    "Estimate panel model",
                    "panel_model",
                    clear_label="Clear latest panel result",
                ):
                    try:
                        if not x:
                            raise ValueError("Select at least one explanatory variable.")
                        columns = [entity, time, y] + x
                        sample = df[columns].copy()
                        for column in [y] + x:
                            sample[column] = pd.to_numeric(sample[column], errors="coerce")
                        sample = sample.dropna().set_index([entity, time]).sort_index()
                        y_data = sample[y]
                        x_data = sample[x]
                        if panel_constant:
                            x_data = sm.add_constant(x_data, has_constant="add")

                        if panel_method == "Pooled OLS":
                            model = PooledOLS(y_data, x_data)
                            model_code = "PooledOLS(y_data, x_data)"
                        elif panel_method == "Fixed effects":
                            model = PanelOLS(
                                y_data,
                                x_data,
                                entity_effects=entity_effects,
                                time_effects=time_effects,
                                drop_absorbed=True,
                            )
                            model_code = f"PanelOLS(y_data, x_data, entity_effects={entity_effects!r}, time_effects={time_effects!r}, drop_absorbed=True)"
                        elif panel_method == "Random effects":
                            model = RandomEffects(y_data, x_data)
                            model_code = "RandomEffects(y_data, x_data)"
                        elif panel_method == "First difference":
                            if panel_constant:
                                x_data = sample[x]
                            model = FirstDifferenceOLS(y_data, x_data)
                            model_code = "FirstDifferenceOLS(y_data, x_data)"
                        elif panel_method == "Between":
                            model = BetweenOLS(y_data, x_data)
                            model_code = "BetweenOLS(y_data, x_data)"
                        else:
                            model = FamaMacBeth(y_data, x_data)
                            model_code = "FamaMacBeth(y_data, x_data)"

                        fit_kwargs: dict[str, Any] = {"cov_type": panel_cov}
                        if panel_cov == "clustered":
                            fit_kwargs.update(
                                cluster_entity=cluster_entity,
                                cluster_time=cluster_time,
                            )
                        results = model.fit(**fit_kwargs)
                        table = linearmodels_parameter_table(results)
                        text = str(results.summary)
                        st.dataframe(table, use_container_width=True)
                        st.text(text)

                        code = f"""
columns = {columns!r}
sample = data[columns].copy()
for column in {[y] + x!r}:
    sample[column] = pd.to_numeric(sample[column], errors="coerce")
sample = sample.dropna().set_index([{entity!r}, {time!r}]).sort_index()
y_data = sample[{y!r}]
x_data = sample[{x!r}]
{"x_data = sm.add_constant(x_data, has_constant='add')" if panel_constant and panel_method != "First difference" else ""}
model = {model_code}
results = model.fit(**{fit_kwargs!r})
print(results.summary)
"""
                        name = make_unique_name(panel_method.replace(" ", "_"))
                        register_output(
                            name,
                            table,
                            code,
                            {
                                "method": panel_method,
                                "entity": entity,
                                "time": time,
                                "dependent": y,
                                "explanatory": x,
                            },
                            text,
                        )
                        st.session_state.results[name + "_summary"] = text
                    except Exception as exc:
                        display_exception(exc)

        with panel_tabs[1]:
            if not LINEARMODELS_AVAILABLE:
                st.error(f"Install linearmodels to use IV estimators: {LINEARMODELS_IMPORT_ERROR}")
            else:
                numeric = numeric_columns(df)
                iv_method = st.selectbox("IV estimator", ["2SLS", "LIML", "IV-GMM"])
                y = st.selectbox("Dependent variable", numeric, key="iv_y")
                exog = st.multiselect("Exogenous regressors", [c for c in numeric if c != y], key="iv_exog")
                endog = st.multiselect("Endogenous regressors", [c for c in numeric if c not in [y] + exog], key="iv_endog")
                instruments = st.multiselect(
                    "Excluded instruments",
                    [c for c in numeric if c not in [y] + exog + endog],
                    key="iv_instruments",
                )
                iv_constant = st.checkbox("Include intercept", value=True, key="iv_constant")
                iv_cov = st.selectbox("IV covariance", ["unadjusted", "robust", "kernel", "clustered"])

                if analysis_action_buttons(
                    "Estimate IV model",
                    "iv_model",
                    clear_label="Clear latest IV result",
                ):
                    try:
                        if not endog or not instruments:
                            raise ValueError("Select endogenous regressors and excluded instruments.")
                        columns = [y] + exog + endog + instruments
                        sample = clean_numeric_frame(df, columns)
                        y_data = sample[y]
                        exog_data = sample[exog] if exog else pd.DataFrame(index=sample.index)
                        if iv_constant:
                            exog_data = sm.add_constant(exog_data, has_constant="add")
                        endog_data = sample[endog]
                        instrument_data = sample[instruments]

                        if iv_method == "2SLS":
                            model = IV2SLS(y_data, exog_data, endog_data, instrument_data)
                            model_code = "IV2SLS(y_data, exog_data, endog_data, instrument_data)"
                        elif iv_method == "LIML":
                            model = IVLIML(y_data, exog_data, endog_data, instrument_data)
                            model_code = "IVLIML(y_data, exog_data, endog_data, instrument_data)"
                        else:
                            model = IVGMM(y_data, exog_data, endog_data, instrument_data)
                            model_code = "IVGMM(y_data, exog_data, endog_data, instrument_data)"

                        results = model.fit(cov_type=iv_cov)
                        table = linearmodels_parameter_table(results)
                        text = str(results.summary)
                        st.dataframe(table, use_container_width=True)
                        st.text(text)

                        code = f"""
sample = data[{columns!r}].apply(pd.to_numeric, errors="coerce").dropna()
y_data = sample[{y!r}]
exog_data = sample[{exog!r}] if {bool(exog)!r} else pd.DataFrame(index=sample.index)
{"exog_data = sm.add_constant(exog_data, has_constant='add')" if iv_constant else ""}
endog_data = sample[{endog!r}]
instrument_data = sample[{instruments!r}]
model = {model_code}
results = model.fit(cov_type={iv_cov!r})
print(results.summary)
"""
                        name = make_unique_name(iv_method.replace("-", "_"))
                        register_output(
                            name,
                            table,
                            code,
                            {
                                "method": iv_method,
                                "dependent": y,
                                "exogenous": exog,
                                "endogenous": endog,
                                "instruments": instruments,
                            },
                            text,
                        )
                        st.session_state.results[name + "_summary"] = text
                    except Exception as exc:
                        display_exception(exc)

# ---------------------------------------------------------------------------
# 7. VOLATILITY
# ---------------------------------------------------------------------------
with tabs[6]:
    st.header("Financial volatility models")
    df = require_data()
    if df is not None:
        if not ARCH_AVAILABLE:
            st.error(f"Install arch to use volatility models: {ARCH_IMPORT_ERROR}")
        else:
            numeric = numeric_columns(df)
            y = st.selectbox("Series", numeric, key="garch_y")
            rescale = st.checkbox("Multiply series by 100 before estimation", value=False)
            mean_model = st.selectbox("Mean model", ["Constant", "Zero", "AR"])
            lags = int(st.number_input("Mean-model AR lags", 0, 20, 0))
            vol_model = st.selectbox("Volatility model", ["GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH"])
            p = int(st.number_input("Volatility p", 1, 10, 1))
            o = int(st.number_input("Asymmetry o", 0, 10, 0))
            q = int(st.number_input("Volatility q", 0, 10, 1))
            distribution = st.selectbox("Error distribution", ["normal", "t", "skewt", "ged"])

            if analysis_action_buttons(
                "Estimate volatility model",
                "volatility",
                clear_label="Clear latest volatility result",
            ):
                try:
                    series = pd.to_numeric(df[y], errors="coerce").dropna()
                    if rescale:
                        series = series * 100
                    model = arch_model(
                        series,
                        mean=mean_model,
                        lags=lags if mean_model == "AR" else 0,
                        vol=vol_model,
                        p=p,
                        o=o,
                        q=q,
                        dist=distribution,
                        rescale=False,
                    )
                    results = model.fit(disp="off")
                    table = pd.DataFrame(
                        {
                            "Coefficient": results.params,
                            "Std. Error": results.std_err,
                            "Statistic": results.tvalues,
                            "P-value": results.pvalues,
                        }
                    )
                    text = str(results.summary())
                    st.dataframe(table, use_container_width=True)
                    st.text(text)

                    fig, ax = plt.subplots()
                    results.conditional_volatility.plot(ax=ax)
                    ax.set_title("Conditional volatility")
                    st.pyplot(fig)
                    plt.close(fig)

                    code = f"""
series = pd.to_numeric(data[{y!r}], errors="coerce").dropna()
{"series = series * 100" if rescale else ""}
model = arch_model(
    series,
    mean={mean_model!r},
    lags={lags if mean_model == "AR" else 0},
    vol={vol_model!r},
    p={p},
    o={o},
    q={q},
    dist={distribution!r},
    rescale=False,
)
results = model.fit(disp="off")
print(results.summary())
"""
                    name = make_unique_name(vol_model)
                    register_output(
                        name,
                        table,
                        code,
                        {
                            "series": y,
                            "mean": mean_model,
                            "volatility": vol_model,
                            "p": p,
                            "o": o,
                            "q": q,
                            "distribution": distribution,
                        },
                        text,
                    )
                    st.session_state.results[name + "_summary"] = text
                except Exception as exc:
                    display_exception(exc)

# ---------------------------------------------------------------------------
# 8. EXPORT
# ---------------------------------------------------------------------------
with tabs[7]:
    st.header("Export and reproducibility")
    st.write(
        "The generated script contains the transformations and analyses performed through this app. "
        "It is intended to reproduce the displayed results using the exported cleaned dataset."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Recorded analyses", len(st.session_state.code_blocks))
    c2.metric("Saved outputs", len(st.session_state.results))
    c3.metric("History entries", len(st.session_state.history))

    if notice := st.session_state.pop("export_notice", None):
        st.success(notice)

    st.subheader("Manage recorded analyses")
    st.caption(
        "Removing an item deletes its generated code, saved result tables and "
        "text, settings, and related completion entry from the full export. "
        "The current working dataset is not changed."
    )

    analysis_names = registered_analysis_names()
    if analysis_names:
        analyses_to_remove = st.multiselect(
            "Select analyses or recorded operations to remove",
            analysis_names,
            key="analyses_to_remove",
        )
        remove_col, clear_col = st.columns(2)

        if remove_col.button(
            "Remove selected from export",
            disabled=not analyses_to_remove,
            use_container_width=True,
        ):
            removed_count = remove_recorded_analyses(analyses_to_remove)
            st.session_state["export_notice"] = (
                f"Removed {removed_count} selected item(s) from the full export."
            )
            st.rerun()

        confirm_clear_all = clear_col.checkbox(
            "Confirm clearing all recorded work",
            key="confirm_clear_all_export",
        )
        if clear_col.button(
            "Clear all analyses from export",
            disabled=not confirm_clear_all,
            use_container_width=True,
            type="secondary",
        ):
            removed_count = remove_recorded_analyses(analysis_names)
            st.session_state["export_notice"] = (
                f"Cleared {removed_count} recorded item(s) from the full export."
            )
            st.rerun()
    else:
        st.info("There are no recorded analyses or operations to remove.")

    script = generated_analysis_script()
    st.subheader("Generated Python code")
    st.code(script, language="python")

    st.download_button(
        "Download analysis.py",
        data=script,
        file_name="analysis.py",
        mime="text/x-python",
        use_container_width=True,
    )

    if st.session_state.df is not None:
        st.download_button(
            "Download cleaned_dataset.csv",
            data=st.session_state.df.to_csv(index=False).encode("utf-8"),
            file_name=DEFAULT_DATA_FILE,
            mime="text/csv",
            use_container_width=True,
        )

    if st.session_state.results:
        frame_results = {
            name: value
            for name, value in st.session_state.results.items()
            if isinstance(value, (pd.DataFrame, pd.Series))
        }
        if frame_results:
            st.download_button(
                "Download tabular results.xlsx",
                data=dataframe_to_excel_bytes(
                    {
                        name: value if isinstance(value, pd.DataFrame) else value.to_frame("Value")
                        for name, value in frame_results.items()
                    }
                ),
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    st.download_button(
        "Download complete reproduction package (.zip)",
        data=build_reproduction_zip(),
        file_name="econometric_reproduction_package.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    st.subheader("Recorded settings")
    if st.session_state.settings_log:
        st.json(st.session_state.settings_log)
    else:
        st.info("Run an analysis to populate the export package.")

st.divider()
st.caption(
    "Econometrics Studio does not replace methodological judgment. Check variable definitions, "
    "identification assumptions, integration orders, sample size, model stability and diagnostics "
    "before reporting results."
)
