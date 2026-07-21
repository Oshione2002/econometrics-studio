<div align="center">

# 📊 Econometrics Studio

### Econometric analysis without terminal commands

**Upload data · transform variables · estimate models · run model-specific diagnostics · export reproducible Python code**

<br />

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.49%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Version](https://img.shields.io/badge/version-1.5.0-1F4E78?style=for-the-badge)
![License](https://img.shields.io/badge/license-research%20use-2EA44F?style=for-the-badge)

<br />

A browser-based econometrics workspace that turns Python's statistical ecosystem into a guided graphical interface. Build, estimate, diagnose and export a reproducible analysis without writing terminal commands.

</div>

---

## ✨ Why Econometrics Studio?

<table>
<tr>
<td width="33%" valign="top">

### 🧭 Guided workflow
Upload a dataset, prepare variables and configure estimators through structured forms and menus.

</td>
<td width="33%" valign="top">

### 🧪 Model-aware diagnostics
Diagnostic tests are generated from the exact fitted estimation result and automatically adapt to the estimator.

</td>
<td width="33%" valign="top">

### ♻️ Reproducible by design
Download the generated Python code, cleaned data, result tables and recorded model settings.

</td>
</tr>
</table>

## 🚀 Highlights in version 1.5.0

- **True post-estimation diagnostics** use the exact saved fitted model instead of estimating a duplicate model.
- **Estimator-specific diagnostic suites** adapt to linear, binary, count, ARDL/ARIMA, VAR/VECM, panel, IV and volatility models.
- **Linked diagnostic exports** attach each diagnostic run to its parent estimation.
- **Automatic cleanup** removes linked diagnostics when their parent model is cleared.
- **No duplicate regression diagnostics** inside the estimation page.
- **Persistent navigation and charts** remain in place after reruns and clear actions.
- **Light and dark modes** cover tables, settings, charts, forms and result displays.
- **Selective export control** removes unwanted work from the final reproduction package.

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
| **Diagnostics** | Exact-result residual, specification, stability, classification, count, first-stage, overidentification and volatility diagnostics |

## 🧪 Post-estimation diagnostic workflow

Open **Diagnostics** from Streamlit's page navigation after estimating a model.

```text
Estimate a model
      ↓
The exact fitted result is saved in the session
      ↓
Open Diagnostics
      ↓
Select the previously estimated model
      ↓
Run only tests appropriate for that estimator
      ↓
Export or clear the linked diagnostic run
```

<table>
<tr>
<td width="33%" valign="top">

### Linear models
Normality, serial correlation, heteroskedasticity, ARCH effects, RESET, Rainbow, CUSUM, VIF, condition number and influence measures.

</td>
<td width="33%" valign="top">

### Discrete and count models
Classification metrics, ROC AUC, grouped probability fit, Pearson and deviance residuals, dispersion and zero-count checks.

</td>
<td width="33%" valign="top">

### Time-series and volatility
Ljung–Box, ARCH effects, residual normality, VAR/VECM whiteness and stability, and standardized GARCH residual tests.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### Panel models
Residual behaviour, cross-sectional dependence and entity-level residual-variance checks based on the fitted panel result.

</td>
<td width="50%" valign="top" colspan="2">

### Instrumental variables
First-stage diagnostics, endogeneity tests and available overidentification tests from the fitted IV result.

</td>
</tr>
</table>

Diagnostic runs remain visible after reruns, support dark mode, include appropriate plots and have a **Clear latest diagnostic run** button.

## 🖥️ Interface workflow

```text
Upload data
    ↓
Prepare and rename variables
    ↓
Choose an econometric method
    ↓
Estimate the model
    ↓
Run model-specific post-estimation diagnostics
    ↓
Clear unwanted analyses, diagnostics or charts
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

Streamlit opens the application in your browser. Estimate a model in the main workspace, then select **Diagnostics** from the page navigation.

## ☁️ Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud.
2. Choose **Create app**.
3. Select this GitHub repository.
4. Set the entry point to `app.py`.
5. Click **Deploy**.

The repository contains `.streamlit/config.toml`, `requirements.txt` and the Streamlit `pages` directory.

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

Use the clear control beside an analysis, the **Clear latest diagnostic run** button, or the export manager to remove work that should not appear in the final package.

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
- whether the estimator and diagnostic tests are appropriate for the research design.

---

<div align="center">

### Built for transparent, reproducible econometric research

**Econometrics Studio · Version 1.5.0**

</div>
