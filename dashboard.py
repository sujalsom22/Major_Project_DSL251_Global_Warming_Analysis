from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Global Warming Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data/processed/weather")
MASTER_PATH = DATA_DIR / "master_long.csv"
ANNUAL_PATH = DATA_DIR / "annual_city.csv"
GLOBAL_PATH = DATA_DIR / "global_annual.csv"

PALETTE = {
    "Africa": "#d55e00",
    "Asia": "#0072b2",
    "Australia": "#e69f00",
    "Europe": "#009e73",
    "North America": "#cc79a7",
    "South America": "#56b4e9",
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f4f8fb 0%, #eef4f7 45%, #f7fafc 100%);
    }
    .hero {
        background: linear-gradient(135deg, #0b3c5d 0%, #146c94 48%, #7ec8e3 100%);
        color: white;
        padding: 2rem 2rem 1.6rem 2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 24px rgba(11, 60, 93, 0.16);
    }
    .hero h1 {
        margin: 0 0 0.4rem 0;
        font-size: 2.1rem;
    }
    .hero p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.95;
    }
    .note-box {
        background: rgba(255, 255, 255, 0.82);
        border-left: 4px solid #146c94;
        padding: 0.9rem 1rem;
        border-radius: 6px;
        margin: 0.6rem 0 1.2rem 0;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(20, 108, 148, 0.08);
        border-radius: 8px;
        padding: 1rem 1rem 0.4rem 1rem;
        box-shadow: 0 8px 18px rgba(20, 108, 148, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master = pd.read_csv(MASTER_PATH, parse_dates=["date"])
    annual = pd.read_csv(ANNUAL_PATH)
    global_annual = pd.read_csv(GLOBAL_PATH)
    return master, annual, global_annual


@st.cache_data
def monthly_global(master: pd.DataFrame) -> pd.DataFrame:
    monthly = master.groupby(master["date"].dt.to_period("M"))["temp_mean"].mean().reset_index()
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly["rolling_12m"] = monthly["temp_mean"].rolling(12, center=True).mean()
    return monthly


@st.cache_data
def monthly_series_for(master: pd.DataFrame, level: str, region: str | None = None) -> pd.Series:
    if level == "Global":
        filtered = master.copy()
    elif level == "Continent":
        filtered = master[master["continent"] == region].copy()
    else:
        filtered = master[master["city"] == region].copy()

    series = filtered.groupby(filtered["date"].dt.to_period("M"))["temp_mean"].mean()
    series.index = series.index.to_timestamp()
    series = series.resample("MS").mean().interpolate()
    return series


@st.cache_data
def compute_key_findings(master: pd.DataFrame, annual: pd.DataFrame, global_annual: pd.DataFrame) -> dict[str, object]:
    x = global_annual["year"]
    y = global_annual["global_temp_mean"]
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    continent_annual = annual.groupby(["continent", "year"], as_index=False)["temp_annual_mean"].mean()
    continent_rows = []
    for continent, data in continent_annual.groupby("continent"):
        m, b = np.polyfit(data["year"], data["temp_annual_mean"], 1)
        continent_rows.append({"continent": continent, "trend_per_decade": m * 10})
    continent_slopes = pd.DataFrame(continent_rows).sort_values("trend_per_decade", ascending=False)

    city_rows = []
    for city, data in annual.groupby("city"):
        m, b = np.polyfit(data["year"], data["temp_annual_mean"], 1)
        city_rows.append(
            {
                "city": city,
                "continent": data["continent"].iloc[0],
                "country": data["country"].iloc[0],
                "trend_per_decade": m * 10,
                "latitude": data["latitude"].iloc[0],
                "longitude": data["longitude"].iloc[0],
            }
        )
    city_slopes = pd.DataFrame(city_rows).sort_values("trend_per_decade", ascending=False)

    return {
        "global_slope_decade": slope * 10,
        "global_r2": r2,
        "temp_change": global_annual["global_temp_mean"].iloc[-1] - global_annual["global_temp_mean"].iloc[0],
        "peak_daily_anomaly": master["temp_anomaly"].max(),
        "warmest_city": annual.groupby("city")["temp_annual_mean"].mean().idxmax(),
        "top_continent": continent_slopes.iloc[0]["continent"],
        "top_continent_trend": continent_slopes.iloc[0]["trend_per_decade"],
        "top_city": city_slopes.iloc[0]["city"],
        "top_city_trend": city_slopes.iloc[0]["trend_per_decade"],
        "continent_slopes": continent_slopes,
        "city_slopes": city_slopes,
    }


def make_risk_map(city_slopes: pd.DataFrame) -> go.Figure:
    risk_df = city_slopes.copy()
    risk_df["slope_per_year"] = risk_df["trend_per_decade"] / 10
    risk_df["slope_per_decade"] = risk_df["trend_per_decade"]
    risk_df["risk_level"] = np.where(risk_df["slope_per_year"] > 0, "High Risk", "Low Risk")

    fig = px.scatter_geo(
        risk_df,
        lat="latitude",
        lon="longitude",
        hover_name="city",
        hover_data={
            "country": True,
            "continent": True,
            "slope_per_year": ":.4f",
            "slope_per_decade": ":.3f",
            "latitude": False,
            "longitude": False,
        },
        color="risk_level",
        color_discrete_map={
            "High Risk": "red",
            "Low Risk": "blue",
        },
        size=abs(risk_df["slope_per_decade"]) + 0.05,
        projection="natural earth",
        title="Global Warming Risk Map Based on City Temperature Trends",
    )

    fig.update_geos(
        showland=True,
        landcolor="rgb(235, 235, 235)",
        showocean=True,
        oceancolor="rgb(220, 235, 250)",
        showcountries=True,
        countrycolor="white",
    )

    fig.update_layout(
        height=600,
        legend_title_text="Risk Level",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def linear_monthly_forecast(series: pd.Series, target_year: int, target_month: int) -> tuple[float, pd.DataFrame, dict[str, float]]:
    annual = series.groupby(series.index.year).mean().reset_index()
    annual.columns = ["year", "annual_temp"]
    model = LinearRegression().fit(annual[["year"]], annual["annual_temp"])

    annual_pred = model.predict(annual[["year"]])
    r2 = r2_score(annual["annual_temp"], annual_pred)
    mae = mean_absolute_error(annual["annual_temp"], annual_pred)
    rmse = np.sqrt(mean_squared_error(annual["annual_temp"], annual_pred))

    month_effect = series.groupby(series.index.month).mean() - series.mean()
    target_value = float(model.predict(pd.DataFrame({"year": [target_year]}))[0] + month_effect.loc[target_month])

    future_dates = pd.date_range(series.index.max() + pd.DateOffset(months=1), datetime(target_year, target_month, 1), freq="MS")
    future_values = [
        float(model.predict(pd.DataFrame({"year": [d.year]}))[0] + month_effect.loc[d.month])
        for d in future_dates
    ]

    chart_df = pd.DataFrame({"date": future_dates, "forecast_temp": future_values})
    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "trend_per_decade": model.coef_[0] * 10,
    }
    return target_value, chart_df, metrics


@st.cache_data(show_spinner=False)
def sarima_monthly_forecast(series: pd.Series, target_year: int, target_month: int) -> tuple[float, pd.DataFrame, dict[str, float]]:
    split_date = pd.Timestamp("2023-01-01")
    train = series[: split_date - pd.DateOffset(months=1)]
    test = series[split_date:]

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    test_pred = fit.get_forecast(steps=len(test)).predicted_mean
    mae = mean_absolute_error(test, test_pred)
    rmse = np.sqrt(mean_squared_error(test, test_pred))

    final_model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    final_fit = final_model.fit(disp=False)

    target_date = pd.Timestamp(datetime(target_year, target_month, 1))
    future_steps = len(pd.date_range(series.index.max() + pd.DateOffset(months=1), target_date, freq="MS"))
    forecast_obj = final_fit.get_forecast(steps=future_steps)
    forecast_mean = forecast_obj.predicted_mean
    conf = forecast_obj.conf_int()

    chart_df = pd.DataFrame(
        {
            "date": forecast_mean.index,
            "forecast_temp": forecast_mean.values,
            "lower_ci": conf.iloc[:, 0].values,
            "upper_ci": conf.iloc[:, 1].values,
        }
    )
    target_value = float(chart_df.iloc[-1]["forecast_temp"])
    metrics = {"mae": mae, "rmse": rmse, "aic": final_fit.aic}
    return target_value, chart_df, metrics


def plot_history_and_forecast(series: pd.Series, forecast_df: pd.DataFrame, title: str, show_band: bool = False) -> go.Figure:
    recent_history = series.tail(24).reset_index()
    recent_history.columns = ["date", "temp_mean"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent_history["date"],
            y=recent_history["temp_mean"],
            mode="lines+markers",
            name="Recent history",
            line=dict(color="#0b3c5d", width=2),
        )
    )
    if show_band and {"lower_ci", "upper_ci"}.issubset(forecast_df.columns):
        fig.add_trace(
            go.Scatter(
                x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
                y=list(forecast_df["upper_ci"]) + list(forecast_df["lower_ci"])[::-1],
                fill="toself",
                fillcolor="rgba(215,48,39,0.14)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Confidence interval",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast_temp"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#d73027", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(showgrid=True, gridcolor="#edf1f4"),
        yaxis=dict(showgrid=True, gridcolor="#edf1f4", title="Temperature (C)"),
        legend=dict(orientation="h", y=1.05),
    )
    return fig


master, annual_city, global_annual = load_data()
findings = compute_key_findings(master, annual_city, global_annual)
monthly_global_df = monthly_global(master)

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Go to", ["Home", "EDA", "Forecasting"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Global Warming Analysis Dashboard")
    st.caption("15 cities | 6 continents | 2015-2024")

if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <h1>Global Warming Analysis and Temperature Trend Forecasting</h1>
            <p>An interactive climate dashboard built from cleaned city-level temperature records across 6 continents, combining exploratory analysis with trend and forecasting models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global Trend", f"{findings['global_slope_decade']:.2f} C/decade")
    c2.metric("Model Fit (R2)", f"{findings['global_r2']:.2f}")
    c3.metric("Temp Change 2015-2024", f"{findings['temp_change']:+.2f} C")
    c4.metric("Peak Daily Anomaly", f"{findings['peak_daily_anomaly']:+.1f} C")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_global_df["date"], y=monthly_global_df["temp_mean"], line=dict(color="#8cc7ea", width=1), name="Monthly mean", opacity=0.6))
        fig.add_trace(go.Scatter(x=monthly_global_df["date"], y=monthly_global_df["rolling_12m"], line=dict(color="#0b3c5d", width=2.6), name="12-month rolling mean"))
        fig.update_layout(title="Global Monthly Temperature Trend", height=380, plot_bgcolor="white", margin=dict(l=10, r=10, t=55, b=10), xaxis=dict(showgrid=True, gridcolor="#edf1f4"), yaxis=dict(showgrid=True, gridcolor="#edf1f4", title="Temperature (C)"), legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f"""
            <div class="note-box">
            <strong>Key takeaways</strong><br>
            The selected-city global average increased by {findings['temp_change']:.2f} C between 2015 and 2024.<br>
            The fitted annual warming trend is {findings['global_slope_decade']:.2f} C per decade.<br>
            Europe currently shows the strongest continent-level warming trend in this cleaned dataset.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(make_risk_map(findings["city_slopes"]), use_container_width=True)
        st.markdown(
            f"""
            <div class="note-box">
            <strong>Spatial snapshot</strong><br>
            Highest warming city in the current sample: {findings['top_city']} ({findings['top_city_trend']:.2f} C/decade).<br>
            Warmest city by average annual mean temperature: {findings['warmest_city']}.<br>
            Red markers indicate stronger observed warming slopes across the selected cities.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Project Snapshot")
    a, b, c = st.columns(3)
    a.markdown(
        """
        <div class="section-card">
        <h4>Data</h4>
        <p>Daily mean, max, and min temperatures collected for 15 cities across Asia, Europe, North America, South America, Africa, and Australia.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    b.markdown(
        """
        <div class="section-card">
        <h4>EDA</h4>
        <p>Trend plots, anomaly heatmaps, distribution views, decomposition, correlation analysis, and global risk mapping built from cleaned monthly and annual series.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c.markdown(
        """
        <div class="section-card">
        <h4>Models</h4>
        <p>Linear models for interpretable warming trends and SARIMA for seasonal monthly forecasting with user-selected global, continent, or city scope.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "EDA":
    st.markdown(
        """
        <div class="hero">
            <h1>Exploratory Data Analysis</h1>
            <p>Eleven visual views spanning trend, seasonality, anomalies, distribution, correlation, decomposition, and spatial warming risk.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Global Trend", "Annual Anomaly", "Heatmap", "City Trends", "Violin", "Season Box", "Slope Ranking", "Scatter", "Correlation", "Decomposition", "Geo Risk"])

    with tabs[0]:
        monthly_global = master.groupby(master["date"].dt.to_period("M"))["temp_mean"].mean().reset_index()
        monthly_global["date"] = monthly_global["date"].dt.to_timestamp()
        monthly_global["rolling_12m"] = monthly_global["temp_mean"].rolling(12, center=True).mean()

        x = np.arange(len(monthly_global))
        valid = monthly_global["rolling_12m"].notnull()
        slope, intercept = np.polyfit(x[valid], monthly_global.loc[valid, "rolling_12m"], 1)

        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(monthly_global["date"], monthly_global["temp_mean"], color="#65c7fc", alpha=0.55, label="Monthly mean")
        ax.plot(monthly_global["date"], monthly_global["rolling_12m"], color="#08519c", linewidth=2.3, label="12-month rolling mean")
        ax.plot(monthly_global["date"][valid], slope * x[valid] + intercept, color="#d73027", linestyle="--", label=f"Fitted line Trend: {slope * 12:.2f} C/year")
        ax.set_title("Global Monthly Temperature Trend")
        ax.set_ylabel("Temperature (C)")
        ax.set_xlabel("")
        ax.legend(frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. The 12-month rolling average smooths short-term seasonal variation and makes the long-term pattern easier to observe.
2. The fitted trend line shows a positive trend of about 0.01 C per year based on the collected city sample.
3. From the rolling average, we can see that the global temperature has rose up by almost 0.4 C in the last decade.
4. Monthly temperatures fluctuate strongly due to seasonality, so rolling averages are more useful than raw monthly values for trend interpretation.
""")

    with tabs[1]:
        colors = np.where(global_annual["global_anomaly_mean"] >= 0, "#d73027", "#4575b4")

        fig, ax = plt.subplots(figsize=(11, 5))
        bars = ax.bar(global_annual["year"], global_annual["global_anomaly_mean"], color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Global Annual Temperature Anomaly")
        ax.set_ylabel("Anomaly (C)")
        ax.set_xlabel("Year")

        for bar, value in zip(bars, global_annual["global_anomaly_mean"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. The annual anomaly increased from about 0.00 C in 2015 to about 0.46 C in 2024.
2. Later years show more positive anomaly values, which suggests warming relative to the baseline period(2015-19) used in this project.
3. Since anomalies are calculated relative to city-specific baseline temperatures, they are more reliable.
4. Year 2021-22 having negative anomaly may be due to Covid time that industry work have declined their rate which decreased pollution and greenhouse gas emmision. That is why pollution is positively correlated with global temperature.  
""")

    with tabs[2]:
        monthly_anomaly = master.groupby(["year", "month"])["temp_anomaly"].mean().reset_index()
        heatmap_data = monthly_anomaly.pivot(index="month", columns="year", values="temp_anomaly")
        heatmap_data.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            heatmap_data,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".1f",
            linewidths=0.4,
            cbar_kws={"label": "Temperature anomaly (C)"},
            ax=ax,
        )
        ax.set_title("Monthly Temperature Anomaly Heatmap")
        ax.set_xlabel("Year")
        ax.set_ylabel("Month")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. The heatmap highlights which months and years were warmer or cooler compared with the baseline.
2. Strong positive anomalies are visible in several mid-year months, especially July and August.
3. This chart is useful for detecting seasonal warming patterns, not just yearly averages.
""")

    with tabs[3]:
        fig, ax = plt.subplots(figsize=(13, 6))

        for city, city_data in annual_city.groupby("city"):
            city_data = city_data.sort_values("year")
            continent = city_data["continent"].iloc[0]
            ax.plot(
                city_data["year"],
                city_data["temp_annual_mean"],
                color=PALETTE.get(continent, "gray"),
                alpha=0.7,
                linewidth=1.5,
            )
            ax.text(
                city_data["year"].iloc[-1] + 0.05,
                city_data["temp_annual_mean"].iloc[-1],
                city,
                fontsize=7,
                color=PALETTE.get(continent, "gray"),
            )

        ax.set_title("Annual Mean Temperature by City")
        ax.set_ylabel("Temperature (C)")
        ax.set_xlabel("Year")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. Cities show different warming behavior, which indicates that temperature change is not uniform across locations.
2. Tokyo, Cairo, New York, and Toronto show some of the strongest positive annual temperature slopes in this dataset.
3. Some cities show weaker or negative slopes, which may be due to local climate variability, coastal effects, or the limited 10-year period.
""")

    with tabs[4]:
        fig, ax = plt.subplots(figsize=(12, 6))
        continent_order = master.groupby("continent")["temp_anomaly"].median().sort_values(ascending=False).index

        sns.violinplot(
            data=master,
            x="continent",
            y="temp_anomaly",
            order=continent_order,
            palette=PALETTE,
            inner="quartile",
            cut=0,
            ax=ax,
        )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Temperature Anomaly Distribution by Continent")
        ax.set_ylabel("Temperature anomaly (C)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. The violin plot shows both the spread and density of temperature anomalies across continents.
2. Europe and North America have higher average anomalies in this dataset compared with Australia and South America.
3. Wider distributions suggest stronger day-to-day or seasonal variability in anomaly values.
4. Asia and Africa have more density towards center which means average anomaly across the continent is almost 0. Means in other words they are stable since last decade.
""")

    with tabs[5]:
        fig, ax = plt.subplots(figsize=(10, 6))
        season_order = ["Winter", "Spring", "Summer", "Autumn"]

        sns.boxplot(
            data=master,
            x="season",
            y="temp_anomaly",
            order=season_order,
            hue="hemisphere",
            palette=["#0072B2", "#D55E00"],
            ax=ax,
        )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Season-wise Temperature Anomaly by Hemisphere")
        ax.set_ylabel("Temperature anomaly (C)")
        ax.set_xlabel("")
        ax.legend(title="Hemisphere")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. Seasonal anomaly patterns differ clearly between the Northern and Southern Hemispheres.
2. Northern Hemisphere summer months show high positive anomalies, while winter months show lower values.
3. The opposite seasonal behavior in the Southern Hemisphere confirms that hemisphere should be considered during seasonal analysis. This is due to the fact that southern hemisphere has summers in November to January and I have extracted season column based on northern hemisphere.
""")

    with tabs[6]:
        slopes = []

        for city, city_data in annual_city.groupby("city"):
            city_data = city_data.sort_values("year")
            slope, intercept = np.polyfit(city_data["year"], city_data["temp_annual_mean"], 1)
            slopes.append({
                "city": city,
                "continent": city_data["continent"].iloc[0],
                "slope_per_year": slope,
            })

        slope_df = pd.DataFrame(slopes).sort_values("slope_per_year", ascending=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=slope_df,
            x="slope_per_year",
            y="city",
            hue="continent",
            dodge=False,
            palette=PALETTE,
            ax=ax,
        )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("City-wise Warming/Cooling Slope")
        ax.set_xlabel("Temperature change per year (C/year)")
        ax.set_ylabel("")
        ax.legend(title="Continent", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. The slope chart ranks cities by their annual temperature change, making local warming patterns easier to compare.
2. Tokyo and Cairo show the strongest positive slopes among the selected cities.
3. Negative slopes in some cities should not be interpreted as global cooling, because local variation and short study duration can affect city-level trends. Main example of this can be seen in Delhi, may be the max temperature of Delhi is increasing year by year but Delhi have severe winters making its yearly mean stable.
""")

    with tabs[7]:
        sample_df = master.sample(min(30000, len(master)), random_state=42)

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.scatterplot(
            data=sample_df,
            x="temp_mean",
            y="temp_range_day_wise",
            hue="continent",
            palette=PALETTE,
            alpha=0.35,
            s=18,
            ax=ax,
        )

        ax.set_title("Daily Temperature Range vs Mean Temperature")
        ax.set_xlabel("Mean temperature (C)")
        ax.set_ylabel("Daily temperature range (C)")
        ax.legend(title="Continent", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. The scatter plot shows how daily temperature range changes with average temperature.
2. Cities and continents form different clusters, suggesting that geography influences both average temperature and daily variation.
3. Temperature range is only weakly correlated with mean temperature, so it provides additional information beyond average temperature alone.
4. From this we can see that Asia has the highest mean temperature whereas North America has lowest mean temperature.
""")

    with tabs[8]:
        corr_cols = [
            "temp_mean",
            "temp_max",
            "temp_min",
            "temp_range_day_wise",
            "temp_anomaly",
            "temp_7day_avg",
            "temp_30day_avg",
            "city_yearly_mean",
            "continent_yearly_mean",
            "global_daily_mean",
        ]

        corr_data = master[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr_data, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.4, ax=ax)
        ax.set_title("Correlation Heatmap of Temperature Features")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. Mean, maximum, and minimum temperatures are strongly correlated, which is expected because they measure related daily temperature behavior.
2. Rolling averages are highly correlated with mean temperature because they are smoothed versions of the same signal.
3. Temperature range has a much weaker relationship with mean temperature, so it can be useful as a separate feature.
""")

    with tabs[9]:
        monthly_series = master.groupby(master["date"].dt.to_period("M"))["temp_mean"].mean()
        monthly_series.index = monthly_series.index.to_timestamp()
        monthly_series = monthly_series.resample("MS").mean().interpolate()

        decomp = seasonal_decompose(monthly_series, model="additive", period=12)

        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

        axes[0].plot(decomp.observed, color="#2c3e50")
        axes[0].set_ylabel("Observed")

        axes[1].plot(decomp.trend, color="#d73027")
        axes[1].set_ylabel("Trend")

        axes[2].plot(decomp.seasonal, color="#0072B2")
        axes[2].set_ylabel("Seasonal")

        axes[3].plot(decomp.resid, color="#666666")
        axes[3].set_ylabel("Residual")

        fig.suptitle("Time Series Decomposition of Global Monthly Temperature", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""### Insights
1. The decomposition separates the global monthly temperature series into trend, seasonality, and residual variation.
2. The seasonal component confirms that temperature follows a strong yearly cycle.
3. The trend component is the most useful part for studying long-term warming behavior which is increasing in recent years.
""")

    with tabs[10]:
        st.plotly_chart(make_risk_map(findings["city_slopes"]), use_container_width=True)
        st.markdown("""### Insights
1. Europe, Japan and New York region is in very high risk of global warming which is a concerning matter. As these countries are away from equator and cooler, due to global warming temperatures are rising and it increases the global temperature on a huge scale.
2. This doesn't mean places India, temperature is not rising, the point is that global warming has affected cooler regions more by drastically changing their temperatures, temperature near equator also rose but it was already warm so the change is not much.
""")

else:
    st.markdown(
        """
        <div class="hero">
            <h1>Forecasting Studio</h1>
            <p>Choose a geography level, a location, a forecasting model, and a target month to generate a temperature prediction with supporting context.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1.1])
    with col1:
        level = st.selectbox("Forecast Level", ["Global", "Continent", "City"])
    with col2:
        if level == "Continent":
            region = st.selectbox("Choose Continent", sorted(master["continent"].unique()))
        elif level == "City":
            region = st.selectbox("Choose City", sorted(master["city"].unique()))
        else:
            region = None
            st.selectbox("Region", ["Global"], disabled=True)
    with col3:
        model_choice = st.selectbox("Model", ["Linear", "SARIMA"])
    with col4:
        st.markdown(
            """
            <div class="note-box">
            Linear gives an interpretable trend projection.<br>
            SARIMA handles monthly seasonality and usually gives the stronger forecast.
            </div>
            """,
            unsafe_allow_html=True,
        )

    target_col1, target_col2 = st.columns(2)
    with target_col1:
        target_year = st.slider("Forecast Year", 2025, 2050, 2040, 1)
    with target_col2:
        target_month = st.slider("Forecast Month", 1, 12, 6, 1, format="%d")
        st.caption(f"Selected month: {MONTH_NAMES[target_month]}")

    if st.button("Predict Temperature", type="primary", use_container_width=True):
        series = monthly_series_for(master, level, region)
        target_date = pd.Timestamp(datetime(target_year, target_month, 1))

        if target_date <= series.index.max():
            st.error("Please choose a month after the observed data period.")
        else:
            with st.spinner("Running forecast..."):
                if model_choice == "Linear":
                    predicted_temp, forecast_df, metrics = linear_monthly_forecast(series, target_year, target_month)
                    chart = plot_history_and_forecast(series, forecast_df, f"{level} Linear Forecast up to {MONTH_NAMES[target_month]} {target_year}", show_band=False)
                else:
                    predicted_temp, forecast_df, metrics = sarima_monthly_forecast(series, target_year, target_month)
                    chart = plot_history_and_forecast(series, forecast_df, f"{level} SARIMA Forecast up to {MONTH_NAMES[target_month]} {target_year}", show_band=True)

            seasonal_norm = series[series.index.month == target_month].mean()
            delta = predicted_temp - seasonal_norm
            label = "Global" if level == "Global" else region

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Temperature", f"{predicted_temp:.2f} C")
            m2.metric("Vs historical same-month mean", f"{delta:+.2f} C")
            if model_choice == "Linear":
                m3.metric("Trend", f"{metrics['trend_per_decade']:.2f} C/decade")
            else:
                m3.metric("RMSE", f"{metrics['rmse']:.3f} C")

            st.plotly_chart(chart, use_container_width=True)

            tail_fluctuation = series.tail(12).reset_index()
            tail_fluctuation.columns = ["date", "temp_mean"]
            fluct_fig = px.bar(tail_fluctuation, x="date", y="temp_mean", title=f"Nearby Monthly Fluctuations for {label}", labels={"temp_mean": "Temperature (C)", "date": "Month"})
            fluct_fig.update_layout(height=300, plot_bgcolor="white", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fluct_fig, use_container_width=True)

            if model_choice == "Linear":
                st.info(f"The linear model gives a smooth long-run projection for {label}. It is best read as a trend estimate rather than a full climate simulation. For this run, the fitted annual trend is {metrics['trend_per_decade']:.2f} C per decade.")
            else:
                st.info(f"The SARIMA model uses monthly seasonality, so it is usually the stronger forecasting choice for temperature. This run achieved MAE={metrics['mae']:.3f} C and RMSE={metrics['rmse']:.3f} C on the held-out monthly test window. That makes it the better model for the dashboard's final forecast view.")


