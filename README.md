# 🌍 Global Warming Analysis — DSL251

> Temperature trend analysis and forecasting using daily weather data for 15 cities across 6 continents (2015–2024).

---

## 📁 Project Structure

```
├── collect_weather_data.py   # API data collection pipeline
├── Preprocessing.ipynb       # Data cleaning & feature engineering
├── EDA.ipynb                 # Exploratory data analysis (11 views)
├── Model_building.ipynb      # Linear & SARIMA forecasting models
├── dashboard.py              # Interactive Streamlit dashboard
├── code.ipynb                # Supporting analysis notebook
└── data/processed/weather/
    ├── master_long.csv       # Daily records (all cities)
    ├── annual_city.csv       # Annual aggregates per city
    └── global_annual.csv     # Global annual summaries
```

---

## 🚀 Quick Start

**1. Collect data**
```bash
python collect_weather_data.py --start-date 2015-01-01 --end-date 2024-12-31
```

**2. Run the dashboard**
```bash
streamlit run dashboard.py
```

---

## 🔍 What's Inside

| Phase | Description |
|---|---|
| **Data Collection** | Daily min/max/mean temperatures via Open-Meteo API |
| **Preprocessing** | Null imputation, outlier handling, feature engineering |
| **EDA** | Trend plots, anomaly heatmaps, decomposition, geo risk map |
| **Modelling** | Linear regression (trend) + SARIMA(1,1,1)(1,1,1)[12] |
| **Dashboard** | Interactive forecasting studio with continent/city scope |

---

## 🛠 Tech Stack

`Python` · `pandas` · `statsmodels` · `scikit-learn` · `plotly` · `streamlit`

---

## 📊 Key Finding

The dataset shows a **positive warming trend** across all 6 continents over the 2015–2024 period, with Europe recording the steepest city-level warming slope in the sample.
