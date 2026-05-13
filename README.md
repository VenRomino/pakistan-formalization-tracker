# Pakistan Informal Economy Formalization Tracker

**A data science portfolio project measuring how quickly Pakistan's informal economy
is (or isn't) entering formal economic systems — 2015 to 2025.**

All data sourced directly from official SBP quarterly reports, FBR year books,
and PBS Labour Force Surveys. No Kaggle. No third-party aggregators.

---

## The Research Question

Pakistan's informal economy accounts for roughly 70% of employment. Over the past
decade, the country has simultaneously seen an explosion in fintech adoption and a
series of tax broadening drives. The question this project asks:

> *Are digital payments, mobile banking, and SME lending actually pulling people
> into the formal economic system — or is the progress surface-level?*

---

## Key Findings

**1. Digital payments grew 144× in ten years. Informal employment moved 0.9 percentage points.**

From ₨905B in 2015 to ₨130,657B in 2025 (mobile + internet banking, annualised from
SBP Q3 data). Over the same period, non-agricultural informal employment fell from
73.0% to 72.1% (PBS Labour Force Survey). The gap between financial access and
structural formalization is the central finding of this project.

**2. By 2025, Pakistan had 96 million mobile digital finance users — and 5.2 million tax filers.**

Bank mobile apps: 22.6M · Branchless banking apps (JazzCash, Easypaisa): 68.5M ·
EMI e-wallets: 5.3M · FBR active filers: ~5.2M. Only 5.4% of mobile digital users
are registered tax filers. Digital identity is not automatically converting into
fiscal identity.

**3. The 2022 FBR filer spike was largely procedural.**

Registered filers jumped from 3.04M (2021) to 6.30M (2022) — more than doubling
in one year — due to mandatory filing requirements linked to lower withholding tax
rates on banking transactions. The Auditor General subsequently found that a 76%
increase in filers produced only 30% more revenue. Filer counts fell back to 5.22M
by 2024, confirming the 2022 surge was driven by incentive structure, not genuine
economic formalization.

**4. 89% of retail payments are digital by volume — but 71% of value remains OTC.**

As of Q3FY25, digital channels dominate by transaction count but large-value payments
still flow through bank branches and branchless banking agents. People are using
phones to pay bills; they are not yet using formal credit or investment channels
in meaningful numbers.

**5. Cross-correlation between mobile banking and tax filers is strong but symmetric.**

Pearson r ≈ 0.88 at lags −2 to +2 years — suggesting co-movement but no clear
directional lead. Both series may be responding to common drivers (GDP growth,
urbanisation, policy) rather than one causing the other. A Granger causality test
with a longer series would be required to resolve this.

---

## Composite Formalization Index

| Indicator | Weight | Source | Status |
|---|---|---|---|
| FBR Registered Return Filers | 30% | FBR Year Books | Real |
| Bank Mobile Banking Users | 25% | SBP PSR Quarterly | Real |
| Digital Payment Volume | 20% | SBP PSR Quarterly | Real |
| SME Share of Bank Credit | 15% | SBP BSC Reports | Estimated |
| Informal Employment Share (inv.) | 10% | PBS Labour Force Survey | Real (interpolated) |

Index scores rise from 0.0 (2015 baseline) to 96.0 (2025). The 2020 dip and
2022 spike are preserved as real data artefacts — not smoothed — because they
reflect genuine policy events.

---

## Data Sources

| Source | Coverage |
|---|---|
| SBP Payment Systems Quarterly Review (Q3FY15–Q3FY25) | Mobile banking, payments |
| FBR Year Books & Annual Reports | Tax filers 2015–2024 |
| PBS Labour Force Survey (2015, 2018, 2021, 2024-25) | Informal employment |
| SBP Development Finance Review | SME credit (estimated) |

**Key methodological notes:**

- Mobile banking users = bank-app users only. Branchless banking (JazzCash/Easypaisa) tracked separately from FY24.
- Digital payment volume = annualised Q3 figure × 4. Assumes stable seasonality.
- FBR filers = Total Return Filers (includes nil filers). Active taxpayer counts substantially lower.
- Informal employment = non-agricultural share (ILO/ICLS). 2018-19 PBS figure of 86.8% excluded — incompatible methodology.
- FY17 mobile banking break: apparent dip 2.36M→1.67M is SBP reclassification, not a real decline.

---

## Quickstart

```bash
# View immediately — no install
open dashboard.html

# Streamlit app
pip install -r requirements.txt
streamlit run dashboard.py

# Jupyter notebook
jupyter notebook notebooks/analysis.ipynb

# Regenerate HTML dashboard after updating data
python3 build_dashboard.py
```

---

## Project Structure

```
pakistan-formalization-tracker/
├── README.md
├── dashboard.html          ← main deliverable (single file, open in browser)
├── build_dashboard.py      ← regenerates dashboard.html
├── dashboard.py            ← Streamlit version
├── requirements.txt
├── data/
│   ├── formalization_data.csv
│   ├── policy_events.csv
│   └── regional_data.csv
└── notebooks/
    └── analysis.ipynb
```

---

## Tech Stack

Python · pandas · numpy · scikit-learn · scipy · matplotlib · Plotly.js · Streamlit · Jupyter

---

*The core analytical contribution is distinguishing between digital access — where
Pakistan has made extraordinary gains — and fiscal and labour market formalization,
where progress remains structurally limited despite a decade of fintech growth.*
