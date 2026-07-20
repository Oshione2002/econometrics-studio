<div align="center">

# 📊 Econometrics Studio

### Econometric analysis without terminal commands

**Upload data · transform variables · estimate models · run diagnostics · export reproducible Python code**

<br />

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Version](https://img.shields.io/badge/version-1.3.0-1F4E78?style=for-the-badge)
![License](https://img.shields.io/badge/license-research%20use-2EA44F?style=for-the-badge)

<br />

A browser-based econometrics workspace that turns Python's statistical ecosystem into a guided graphical interface. Build, estimate, diagnose and export a complete reproducible analysis without writing terminal commands.

</div>

---

## ✨ Why Econometrics Studio?

<table>
<tr>
<td width="33%" valign="top">

### 🧭 Guided workflow
Upload a dataset, select variables and configure estimators through clear forms and menus.

</td>
<td width="33%" valign="top">

### 🧪 Serious econometrics
Run cross-sectional, time-series, panel, instrumental-variable and volatility models.

</td>
<td width="33%" valign="top">

### ♻️ Reproducible by design
Download the exact generated Python code, cleaned data, results and model settings.

</td>
</tr>
</table>

## 🚀 Highlights in version 1.3.0

- **Persistent navigation** keeps you on the same workspace and sub-method after clearing results, changing dark mode or triggering a rerun
- **Reliable dark tables** replace the light DataFrame canvas with high-contrast scrollable tables in dark mode
- **Readable recorded settings** use a dark-compatible JSON code view
- **Persistent charts with a Clear chart button** so figures remain visible until deliberately removed
- **Immediate clear buttons** beside each analysis action
- **Variable renaming** with recorded-code and result-label updates
- **Selective export control** for removing unwanted analyses
- **Complete reproduction package** containing code, data, outputs and settings

## 🧰 Econometric coverage

| Area | Included methods |
|---|---|
| **Data management** | CSV, Excel and Stata upload; sorting; missing-value inspection and treatment |
| **Transformations** | Logs, signed logs, differences, growth rates, lags, leads, scaling and interactions |
| **Descriptive analysis** | Summary statistics, normality measures, correlations and persistent charts |
| **Regression** | OLS, WLS, GLS, robust regression, quantile regression, logit, probit and count models |
| **Unit roots** | ADF, Phillips–Perron, KPSS, DF-GLS and Zivot–Andrews |
| **Time series** | ARDL, UECM, PSS bounds test, ARIMA/SARIMA, VAR, Johansen, VECM and Granger causality |
| **Panel data** | Pooled OLS, fixed effects, random effects, first differences, between and Fama–MacBeth |
| **Endogeneity** | 2SLS, LIML and IV-GMM |
| **Volatility** | ARCH, GARCH, EGARCH, FIGARCH, APARCH and HARCH |
| **Diagnostics** | Serial correlation, heteroskedasticity, ARCH effects, normality, RESET, VIF and stability checks |

## 🖥️ Interface workflow

```text
Upload data
    ↓
Prepare and rename variables
    ↓
Choose an econometric method
    ↓
Estimate and inspect diagnostics
    ↓
Clear unwanted analyses or charts immediately
    ↓
Export code, cleaned data, tables and settings
```

## ⚡ Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Oshione2002/econometrics-studio.git
cd econometrics-studio
```

### 2. Create an optional virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the application

```bash
streamlit run app.py
```

Streamlit opens the application in your browser.

## ☁️ Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud.
2. Choose **Create app**.
3. Select this GitHub repository.
4. Set the entry point to `app.py`.
5. Click **Deploy**.

The repository already contains `.streamlit/config.toml` and `requirements.txt`.

## 📦 Reproduction package

Every retained analysis can be exported as a ZIP containing:

```text
econometric_reproduction_package/
├── analysis.py
├── cleaned_dataset.csv
├── results.xlsx
├── analysis_configuration.json
├── requirements.txt
├── README.md
└── text_results/
```

Use the clear button beside an analysis, or the export manager, to remove work that should not appear in the final package.

## 🧱 Technology

<div align="center">

| Interface | Data | Econometrics | Volatility | Panel and IV | Export |
|---|---|---|---|---|---|
| Streamlit | pandas / NumPy | statsmodels | arch | linearmodels | openpyxl / XlsxWriter |

</div>

## ⚠️ Research responsibility

Econometrics Studio assists with computation and reproducibility; it does not replace methodological judgment. Before reporting results, verify:

- variable definitions and measurement;
- identification assumptions;
- integration orders and cointegration requirements;
- sample size and lag structure;
- residual diagnostics and model stability; and
- whether the estimator is appropriate for the research design.

---

<div align="center">

### Built for transparent, reproducible econometric research

**Econometrics Studio · Version 1.3.0**

</div>
