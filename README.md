# Econometrics Studio

A Streamlit application for running common econometric procedures without typing model commands in a terminal.

## Included

- CSV, Excel and Stata upload
- Variable transformations and missing-value treatment
- Descriptive statistics and correlations
- OLS, WLS, GLS, robust regression, quantile regression
- Logit, probit, Poisson and negative binomial
- ADF, Phillips-Perron, KPSS, DF-GLS and Zivot-Andrews
- ARDL lag selection, UECM and PSS bounds testing
- ARIMA/SARIMA, VAR, VECM and Granger causality
- Panel pooled OLS, fixed effects, random effects, first difference, between and Fama-MacBeth
- 2SLS, LIML and IV-GMM
- ARCH/GARCH-family volatility models
- Export of the complete generated Python analysis code, cleaned data, results and settings

## Run locally

1. Install Python 3.11 or 3.12.
2. Put the project files in one folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

The app opens in your browser.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt` and the `.streamlit` folder.
3. In Streamlit Community Cloud, create an app from the repository.
4. Select `app.py` as the entry point.

## Reproducibility

Every completed analysis adds a code block to the Export tab. The complete ZIP contains:

- `analysis.py`
- `cleaned_dataset.csv`
- `results.xlsx`
- text summaries
- analysis settings and history
- `requirements.txt`

## Important

This is a broad working foundation, not a verified implementation of every econometric estimator ever published. Every method still requires appropriate data, assumptions, identification and diagnostic review.
