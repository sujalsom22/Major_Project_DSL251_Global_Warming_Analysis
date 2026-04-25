from __future__ import annotations

from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
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
    risk_df["risk_level"] = pd.qcut(
        risk_df["trend_per_decade"],
        q=3,
        labels=["Low Risk", "Moderate Risk", "High Risk"],
    )

    fig = px.scatter_geo(
        risk_df,
        lat="latitude",
        lon="longitude",
        color="risk_level",
        size=risk_df["trend_per_decade"].abs() + 0.1,
        hover_name="city",
        hover_data={
            "country": True,
            "continent": True,
            "trend_per_decade": ":.3f",
            "latitude": False,
            "longitude": False,
        },
        color_discrete_map={
            "High Risk": "#d73027",
            "Moderate Risk": "#fdae61",
            "Low Risk": "#4575b4",
        },
        projection="natural earth",
        title="Warming Risk Across Selected Global Cities",
    )
    fig.update_geos(
        showland=True,
        landcolor="#eef3f6",
        showocean=True,
        oceancolor="#dbeaf4",
        showcountries=True,
        countrycolor="white",
    )
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=55, b=0))
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
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_global_df["date"], y=monthly_global_df["temp_mean"], line=dict(color="#8cc7ea", width=1), name="Monthly mean", opacity=0.55))
        fig.add_trace(go.Scatter(x=monthly_global_df["date"], y=monthly_global_df["rolling_12m"], line=dict(color="#0b3c5d", width=2.5), name="12-month rolling mean"))
        fig.update_layout(title="Global Monthly Temperature Trend", height=430, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"The 12-month rolling line makes the warming signal easier to see than raw monthly values. Across the cleaned dataset, global annual temperature rises by about {findings['global_slope_decade']:.2f} C per decade, with a modest fit of R2={findings['global_r2']:.2f}. This tells us the long-term direction is upward, even though year-to-year variation is still visible.")

    with tabs[1]:
        fig = px.bar(global_annual, x="year", y="global_anomaly_mean", color="global_anomaly_mean", color_continuous_scale="RdBu_r", color_continuous_midpoint=0, title="Global Annual Temperature Anomaly")
        fig.update_layout(height=430, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        max_row = global_annual.loc[global_annual["global_anomaly_mean"].idxmax()]
        st.info(f"Positive anomaly bars become stronger in the later part of the series, which supports the warming pattern. The highest annual anomaly in this cleaned run appears in {int(max_row['year'])} at {max_row['global_anomaly_mean']:.2f} C. Because the anomaly is measured against a city baseline, it is more comparable across locations than raw temperature alone.")

    with tabs[2]:
        monthly_anomaly = master.groupby(["year", "month"], as_index=False)["temp_anomaly"].mean()
        heatmap_data = monthly_anomaly.pivot(index="month", columns="year", values="temp_anomaly")
        heatmap_data.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        heat_range = float(heatmap_data.abs().max().max())
        fig = px.imshow(heatmap_data, aspect="auto", color_continuous_scale="RdBu_r", zmin=-heat_range, zmax=heat_range, title="Monthly Temperature Anomaly Heatmap")
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)
        hottest = monthly_anomaly.sort_values("temp_anomaly", ascending=False).iloc[0]
        st.info(f"The heatmap highlights when warming is concentrated within the year, not just across years. The strongest month-year anomaly in the current data is {MONTH_NAMES[int(hottest['month'])]} {int(hottest['year'])}, at {hottest['temp_anomaly']:.2f} C. Several mid-year months stand out, showing that the warming signal is also seasonal.")

    with tabs[3]:
        fig = go.Figure()
        for city, data in annual_city.groupby("city"):
            data = data.sort_values("year")
            color = PALETTE.get(data["continent"].iloc[0], "gray")
            fig.add_trace(go.Scatter(x=data["year"], y=data["temp_annual_mean"], mode="lines", name=city, line=dict(color=color, width=1.8), opacity=0.65))
        fig.update_layout(title="Annual Mean Temperature by City", height=470, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        top_three = findings["city_slopes"].head(3)
        st.info(f"City trends are not uniform, which is exactly why a city-wise view matters. The fastest warming cities in this cleaned dataset are {top_three.iloc[0]['city']}, {top_three.iloc[1]['city']}, and {top_three.iloc[2]['city']}. That spread shows local climate behavior can differ a lot even when the global direction is upward.")

    with tabs[4]:
        fig = px.violin(master, x="continent", y="temp_anomaly", color="continent", color_discrete_map=PALETTE, box=True, points=False, title="Temperature Anomaly Distribution by Continent")
        fig.update_layout(height=430, showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        continent_anom = master.groupby("continent")["temp_anomaly"].mean().sort_values(ascending=False)
        st.info(f"The violin plot shows both spread and density, so it tells us more than a simple average. On average, {continent_anom.index[0]} has the highest anomaly in the cleaned dataset, while {continent_anom.index[-1]} has the lowest. Wider violin shapes point to stronger seasonal or day-to-day variation in anomaly values.")

    with tabs[5]:
        fig = px.box(master, x="season", y="temp_anomaly", color="hemisphere", category_orders={"season": ["Winter", "Spring", "Summer", "Autumn"]}, title="Season-wise Temperature Anomaly by Hemisphere")
        fig.update_layout(height=430, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        season_stats = master.groupby(["hemisphere", "season"])["temp_anomaly"].mean().reset_index()
        north_top = season_stats[season_stats["hemisphere"] == "Northern"].sort_values("temp_anomaly", ascending=False).iloc[0]
        south_top = season_stats[season_stats["hemisphere"] == "Southern"].sort_values("temp_anomaly", ascending=False).iloc[0]
        st.info(f"Seasonality behaves differently across hemispheres, which is exactly what we expect in climate data. In the Northern Hemisphere, the strongest positive anomaly appears in {north_top['season']}; in the Southern Hemisphere, it appears in {south_top['season']}. This is why season and hemisphere both matter when we interpret warming patterns.")

    with tabs[6]:
        slope_df = findings["city_slopes"].sort_values("trend_per_decade", ascending=True)
        fig = px.bar(slope_df, x="trend_per_decade", y="city", color="continent", orientation="h", color_discrete_map=PALETTE, title="City-wise Warming Trend Ranking")
        fig.add_vline(x=0, line_width=1, line_dash="dash")
        fig.update_layout(height=470, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"This chart ranks the local warming slopes directly in C per decade, which makes interpretation straightforward. {findings['top_city']} has the highest warming slope at about {findings['top_city_trend']:.2f} C per decade. Some cities still show weak or negative slopes, which is a reminder that a 10-year climate sample can contain local variability.")

    with tabs[7]:
        sample_df = master.sample(min(30000, len(master)), random_state=42)
        fig = px.scatter(sample_df, x="temp_mean", y="temp_range_day_wise", color="continent", color_discrete_map=PALETTE, opacity=0.35, title="Daily Temperature Range vs Mean Temperature")
        fig.update_layout(height=430, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        corr_val = master[["temp_mean", "temp_range_day_wise"]].corr().iloc[0, 1]
        st.info(f"The scatter plot shows how temperature range behaves relative to average temperature. The correlation here is only {corr_val:.2f}, so daily temperature range adds information that mean temperature alone does not capture. The continent clusters also suggest geography plays a role in both central tendency and variability.")

    with tabs[8]:
        corr_cols = ["temp_mean", "temp_max", "temp_min", "temp_range_day_wise", "temp_anomaly", "temp_7day_avg", "temp_30day_avg", "city_yearly_mean", "continent_yearly_mean", "global_daily_mean"]
        corr_data = master[corr_cols].corr()
        fig = px.imshow(corr_data, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap of Temperature Features")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        strongest = corr_data["temp_mean"].drop("temp_mean").abs().sort_values(ascending=False).index[0]
        st.info(f"Mean, max, and min temperatures move closely together, which is exactly what we would expect. The strongest companion variable for temp_mean here is {strongest}. Temperature range is much weaker, so it behaves like a useful complementary feature rather than a duplicate one.")

    with tabs[9]:
        monthly_series = master.groupby(master["date"].dt.to_period("M"))["temp_mean"].mean()
        monthly_series.index = monthly_series.index.to_timestamp()
        monthly_series = monthly_series.resample("MS").mean().interpolate()
        decomp = seasonal_decompose(monthly_series, model="additive", period=12, extrapolate_trend="freq")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
        fig.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed, line=dict(color="#0b3c5d")), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, line=dict(color="#d73027")), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, line=dict(color="#0072b2")), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, line=dict(color="#6b7280")), row=4, col=1)
        fig.update_layout(height=720, title="Time-Series Decomposition of Global Monthly Temperature", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        seasonal_amp = float(decomp.seasonal.max() - decomp.seasonal.min())
        st.info(f"The decomposition separates long-term movement from repeating seasonal structure. The seasonal component spans about {seasonal_amp:.2f} C from low to high, which confirms a strong yearly climate cycle. The trend panel is the part we care about most for long-run warming interpretation.")

    with tabs[10]:
        st.plotly_chart(make_risk_map(findings["city_slopes"]), use_container_width=True)
        high_risk_count = (pd.qcut(findings["city_slopes"]["trend_per_decade"], q=3, labels=["Low", "Moderate", "High"]) == "High").sum()
        st.info(f"This map translates city slopes into a spatial warming-risk view. {high_risk_count} of the selected cities fall into the highest warming-risk group under this ranking. It is a sampled-city risk map, not a complete global climate hazard map, so the interpretation should stay tied to the cities studied here.")

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


