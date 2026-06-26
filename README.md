# 📦 JDE F42119 — Sales Demand Forecasting
### Prophet vs LSTM vs ARIMA | Real January 2026 Validation

> **Predicting monthly sales quantity and revenue from JD Edwards (JDE) F42119 Sales Order History data using three forecasting models — validated against real January 2026 actuals.**

---

## 🧭 Project Overview

Enterprise ERP systems like JD Edwards accumulate rich transactional history in tables such as **F42119 (Sales Order Detail History)**, accessed via the **P4210 (Sales Order Entry)** application. This project leverages real F42119 data from a Laos-based retail/distribution client to forecast:

- **Monthly Sales Quantity (units)** for January 2026
- **Monthly Revenue (LAK — Laos Kip)** for January 2026

Three models are built and compared:
- **Facebook Prophet** — classical decomposition model
- **Stacked LSTM** — deep learning sequence model
- **Auto ARIMA/SARIMA** — statistical time series model

What makes this project unique: **all three forecasts are validated against real January 2026 F42119 data** — not just held-out historical test data.

---


## 🗂️ Table of Contents

- [Business Problem](#-business-problem)
- [JDE Data Context](#-jde-data-context)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Methodology](#-methodology)
- [Understanding the Models](#-understanding-the-models)
- [Model Results](#-model-results)
- [Real World Validation — January 2026](#-real-world-validation--january-2026)
- [Key Learnings](#-key-learnings)
- [How to Run](#-how-to-run)
- [Future Scope](#-future-scope)

---

## 💼 Business Problem

Manufacturing and retail companies using JDE ERP face recurring challenges:

- **Overstocking** leads to increased warehouse and holding costs
- **Understocking** causes missed revenue and poor customer satisfaction
- Planning teams manually review historical sales reports with no predictive capability

**Goal:** Build an automated forecasting system that uses historical F42119 sales data to predict next-month quantity and revenue — enabling proactive inventory and procurement planning.

---

## 🗄️ JDE Data Context

This project is built around the **F42119 — Sales Order Detail History File**, a core JDE EnterpriseOne table populated when sales orders are processed through **P4210 (Sales Order Entry)**.

| JDE Column | Field Name | Used As |
|---|---|---|
| `SDDGL` | G/L Date | Time axis for forecasting |
| `SDLNTY` | Line Type | Filter: Stock lines only (S) |
| `SDQTYS` | Quantity Shipped | Target variable 1 — units sold |
| `SDAEXP` | Extended Price | Target variable 2 — revenue |
| `SDITM` | Short Item No | Product-level granularity |
| `SDMCU` | Business Unit | Branch/plant segmentation |
| `SDDOCO` | Order Number | Unique order reference |

**JDE Filters applied (same as a JDE Sales Analysis report):**
- `SDLNTY = 'S'` — Stock sales lines only (excludes freight, text, non-stock lines)
- `SDQTYS > 0` — Shipped quantity only (excludes cancellations and reversals)
- `SDAEXP > 0` — Positive revenue lines only

---

## 📁 Dataset

This project uses real F42119 (Sales Order Detail History) data from
JD Edwards EnterpriseOne ERP. Due to client confidentiality, the actual
data files are not included in this repository.

### To run this project, you need:
- `raw_data.csv` — F42119 extract with the following columns:
  - G/L Date, Quantity Shipped, Extended Price, Ln Ty,
    Short Item No, Description, Business Unit, Order Number
- `jan_raw_data.csv` — Same format, for validation month

### Data filters applied:
- Line Type (Ln Ty) = S (Stock lines only)
- Quantity Shipped > 0
- Extended Price > 0

---

## 📂 Project Structure

```
jde-demand-forecasting/
│
├── data/
│   ├── raw_data.csv                   # Raw F42119 extract (Jan 2023–Dec 2025)
│   ├── jan_raw_data.csv               # Real January 2026 actuals for validation
│   └── processed/
│       ├── cleaned_transactions.csv   # Filtered, cleaned transaction data
│       └── monthly_aggregated.csv     # Monthly aggregated time series
│
├── notebooks/
│   ├── 01_data_inspection.ipynb       # Load, clean, filter, aggregate
│   ├── 02_EDA.ipynb                   # Exploratory data analysis
│   ├── 03_prophet_model.ipynb         # Facebook Prophet model
│   ├── 04_lstm_model.ipynb            # Stacked LSTM model
│   ├── 04b_arima_model.ipynb          # Auto ARIMA/SARIMA model
│   └── 05_model_comparison.ipynb      # Final comparison + Jan 2026 validation
│
├── models/saved/
│   ├── prophet_qty_model.pkl
│   ├── prophet_rev_model.pkl
│   ├── lstm_qty_model.keras
│   ├── lstm_rev_model.keras
│   ├── scaler_qty.pkl
│   ├── scaler_rev.pkl
│   ├── arima_qty_model.pkl
│   └── arima_rev_model.pkl
│
├── reports/                           # All generated charts and plots
├── requirements.txt
└── README.md
└── app.py

```

## 🖥️ Production App — app.py

In addition to the Jupyter notebooks, the project includes a fully production-ready **Streamlit web application** (`app.py`) that runs the entire forecasting pipeline end-to-end in a single file — no notebooks required.

### What the app does

- Upload your raw F42119 CSV extract directly in the browser
- Select which models to run (Prophet, LSTM, ARIMA — individually or all together)
- App cleans the data, trains all selected models, and returns forecasts automatically
- Highlights the **best performing model** based on lowest average MAPE
- Optionally upload January 2026 actual data for real-world validation
- Download forecast summary as CSV

### App Structure

| Section | Description |
|---|---|
| Data Overview | Row count, date range, unique items and orders after cleaning |
| Best Model Card | Winner highlighted with Qty MAPE, Rev MAPE, Avg MAPE |
| Forecast Summary | Jan 2026 quantity and revenue forecast from best model with confidence range |
| All Models Comparison | Side-by-side cards for all models with color-coded MAPE pills |
| Forecast Charts | Interactive tabs — Quantity and Revenue trend + forecast |
| Real Validation | Appears only when Jan 2026 actuals uploaded — shows forecast vs actual error |
| Download | Export all model forecasts as CSV |

### How to run

```bash
pip install streamlit prophet tensorflow pmdarima
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Key design decisions

- **All cleaning logic is identical to Notebook 01** — same JDE filters (Ln Ty=S, Qty>0, Price>0), same column selection, same date parsing — ensuring notebook and app results are consistent
- **LSTM lookback=6** — same optimized setting discovered during notebook experimentation
- **ARIMA brute-force search** (`stepwise=False`) — same setting that improved AIC from 831 to 530
- **Ensemble** is automatically computed as the average of all three model forecasts in the validation section

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Classical Forecasting | Facebook Prophet |
| Deep Learning | TensorFlow / Keras (Stacked LSTM) |
| Statistical Forecasting | pmdarima (Auto ARIMA/SARIMA) |
| Evaluation | scikit-learn (MAE, RMSE, MAPE) |
| Environment | Jupyter Notebook |
| ERP Domain | JD Edwards F42119 / P4210 |

---

## 🔬 Methodology

### Stage 1 — Data Inspection & Cleaning (Notebook 01)
- Loaded raw F42119 CSV extract (268 columns → selected 8 relevant columns)
- Applied JDE-specific filters: Line Type = S, Qty > 0, Price > 0
- Converted JDE date format, handled null extended prices (58 rows removed)
- Aggregated daily transactions to monthly time series (36 data points)
- Checked for missing months and validated date range

### Stage 2 — Exploratory Data Analysis (Notebook 02)
- Full 36-month trend visualization for quantity and revenue
- Year-over-Year comparison: 2023 vs 2024 vs 2025
- Month-over-Month and Year-over-Year growth rate analysis
- Quarterly breakdown to identify peak selling quarters
- Item-level revenue analysis across all 20 SKUs
- Correlation analysis: Quantity vs Revenue vs Order Count (r = 0.99+)
- Time series decomposition: Trend + Seasonality + Residual
- ADF stationarity test: both series non-stationary (p > 0.05) — strong upward trend confirmed
- Structural break analysis: Jan–May 2023 ramp-up period identified (~17x lower than normal)

### Stage 3 — Prophet Model (Notebook 03)
- Additive mode selected (seasonal amplitude stays constant — confirmed from decomposition)
- Manual changepoint added at June 2023 to capture structural break
- `changepoint_prior_scale = 0.3` for flexible trend detection
- Yearly seasonality enabled
- Train: Jan 2023 – Sep 2025 | Test: Oct – Dec 2025

### Stage 4 — LSTM Model (Notebook 04)
- MinMaxScaler normalization to [0, 1] range
- Lookback window = 6 months (optimized from initial 12 — dramatically improved MAPE)
- Architecture: `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)`
- Callbacks: EarlyStopping (patience=20) + ReduceLROnPlateau
- Train: Jan 2023 – Sep 2025 | Test: Oct – Dec 2025

### Stage 5 — Auto ARIMA Model (Notebook 04b)
- Used `pmdarima.auto_arima` with brute-force search (`stepwise=False`)
- Forced `d=1` (regular differencing) and `D=1` (seasonal differencing)
- Best model selected: **SARIMA(0,1,0)(1,1,0,12)** for both quantity and revenue
- AIC score: 530.03 (significantly better than stepwise result of 831.25)
- Train: Jan 2023 – Sep 2025 | Test: Oct – Dec 2025

### Stage 6 — Model Comparison & Real Validation (Notebook 05)
- Loaded real January 2026 F42119 data — applied identical cleaning pipeline
- Compared all 3 forecasts + ensemble against actual January 2026 values
- Computed final MAPE on real future data — the true test of model quality

---

## 🧠 Understanding the Models

### Facebook Prophet — The Business Analyst

Prophet was built by Facebook (Meta) in 2017 specifically for business time series. It breaks your data into three components and models each separately:

**Trend:** Captures the overall growth direction. Detects changepoints — sudden shifts in growth rate. In our data, the massive jump from May 2023 (~43K units) to June 2023 (~272K units) is a changepoint that Prophet explicitly handles.

**Seasonality:** Captures repeating patterns. We use additive mode because the seasonal swings in our data stay roughly the same size regardless of the overall level — confirmed from decomposition analysis.

**Changepoint Detection:** `changepoint_prior_scale=0.3` tells Prophet to be moderately flexible in detecting trend shifts — not too rigid, not too sensitive.

**Why Prophet won on January 2026:** It captured the strong long-term upward trend and extrapolated it correctly. January 2026 actual (910K units) was higher than any model expected — Prophet was closest because it didn't anchor too heavily on the recent plateau.

---

### Stacked LSTM — The Pattern Memorizer

LSTM (Long Short-Term Memory) is a deep learning model with memory — unlike regular neural networks, it remembers patterns across time steps.

**Memory Gates:** LSTM uses three gates to decide what to remember, what to forget, and what to output. This makes it powerful for sequential data where recent history matters.

**Why Stacked (2 layers):**
- First LSTM layer (64 units): learns basic patterns — "sales tend to follow a monthly rhythm"
- Second LSTM layer (32 units): learns patterns of patterns — "the growth rate is changing over time"

**Critical finding — Lookback Window:** Initial training with lookback=12 gave MAPE of 23.28%. Reducing to lookback=6 dropped MAPE to 5.86% — a 75% improvement. With only 36 months of data, lookback=12 left too few training sequences. Lookback=6 gave the model more examples to learn from.

**Why LSTM underperformed on January 2026:** With lookback=6, LSTM heavily weighted the recent Jun–Dec 2025 plateau (~530K units/month avg). It missed that January is historically a strong demand month — resulting in a large under-prediction.

---

### Auto ARIMA/SARIMA — The Statistician

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical model with three components:

**AR (p) — AutoRegressive:** Uses past values to predict the future. Like saying "this month's sales depend on what I sold in the last p months."

**I (d) — Integrated:** Differencing to remove the trend and make the series stationary. With d=1, instead of modelling raw sales (500K, 520K, 550K...), it models the change between months (20K, 30K...).

**MA (q) — Moving Average:** Uses past forecast errors to correct predictions. If the model over-predicted last month, it adjusts downward this month.

**Seasonal Component (P,D,Q,m):** Our best model SARIMA(0,1,0)(1,1,0,12) adds a seasonal AR term that looks at the same month one year ago — appropriate for yearly patterns in monthly data.

**Why auto_arima with stepwise=False:** The stepwise search initially returned a very simple ARIMA(0,1,0) with no seasonal component (AIC: 831). Brute-force search found SARIMA(0,1,0)(1,1,0,12) with AIC: 530 — a dramatically better fit.

---

## 📊 Model Results

### Test Set Performance (Oct–Dec 2025)

| Model | Qty MAE | Qty MAPE | Rev MAPE |
|---|---|---|---|
| Prophet | 103,642 | 19.08% | 11.92% |
| **LSTM** | **32,665** | **5.86%** | **9.22%** |
| ARIMA | — | 15.19% | 11.93% |

**LSTM wins on the test set** — beating Prophet by 13.22 percentage points on quantity.


**Real Data Hidden for Security purposes.**

### 🏆 Winner: Prophet (8.96% average MAPE on real January 2026 data)

---

## 🧠 Key Learnings

**1. Test set performance ≠ future performance**
LSTM had the best test set MAPE (5.86%) but the worst January 2026 MAPE (46.66%). A model that memorizes recent patterns can fail when the future doesn't follow recent trends. Real-world validation with actual future data is critical.

**2. Lookback window matters more than model complexity**
Reducing LSTM lookback from 12 to 6 months improved quantity MAPE from 23.28% to 5.86% — a 75% improvement. With limited data (36 months), fewer training sequences hurt LSTM more than a shorter memory window.

**3. Brute-force model selection pays off**
Auto ARIMA with stepwise search returned ARIMA(0,1,0) — essentially a random walk. Switching to brute-force (`stepwise=False`) found SARIMA(0,1,0)(1,1,0,12) with AIC improving from 831 to 530. Always verify that your auto-selection method is exploring enough candidates.

**4. Domain knowledge accelerates data science**
Understanding JDE F42119 schema meant no guesswork about which columns to use, which filters to apply (SDLNTY=S, SDQTYS>0), or what the June 2023 structural break meant (system go-live). ERP domain expertise is a genuine data science accelerator in enterprise contexts.

**5. The structural break was the biggest modelling challenge**
Jan–May 2023 ramp-up volumes were ~17x lower than normal operations. This distorted seasonal patterns and made ARIMA's seasonal detection difficult. Prophet's explicit changepoint handling was the key advantage that helped it win on real future data.

**6. Prophet's strength: long-term trend extrapolation**
January 2026 actual (910K units) was significantly higher than all model forecasts — the business continued growing strongly. Prophet's trend component extended the long-term growth trajectory most accurately, while LSTM anchored to recent plateau months.

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add data files
Place your F42119 extracts in the root directory:
- `raw_data.csv` — Jan 2023 to Dec 2025 extract
- `jan_raw_data.csv` — January 2026 extract (for validation)

### 4. Run notebooks in order
```
01_data_inspection.ipynb
02_EDA.ipynb
03_prophet_model.ipynb
04_lstm_model.ipynb
04b_arima_model.ipynb
05_model_comparison.ipynb
```

---

## 📋 Requirements

```
pandas>=2.1.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
prophet>=1.1.5
tensorflow>=2.14.0
scikit-learn>=1.3.2
pmdarima>=2.0.4
statsmodels>=0.14.0
jupyter>=1.0.0
openpyxl>=3.1.0
```

---

## 🔭 Future Scope

- [ ] Item-level (SDITM) forecasting — predict demand per SKU
- [ ] Extend to February and March 2026 rolling forecasts
- [ ] Add XGBoost with lag features as a fourth benchmark model
- [ ] Deploy forecast API using FastAPI + Docker
- [X] Build interactive Streamlit dashboard for business users
- [ ] Automate monthly retraining pipeline using MLflow

---

## 👤 Author

*BikashBIOS*

[![GitHub](https://img.shields.io/badge/GitHub-BikashBIOS-black)](https://github.com/BikashBIOS)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-bikash-ranjan-ojha-5b4953178-blue)](https://linkedin.com/in/bikash-ranjan-ojha-5b4953178)

---

## ⭐ If you found this useful, please star the repo!