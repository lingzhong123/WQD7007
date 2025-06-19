import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# ----------------------------
# Load dataset (parquet file)
# ----------------------------
@st.cache_data

def load_data():
    df = pd.read_parquet("data.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

df = load_data()

st.title("🌍 Global COVID-19 Data Analysis Dashboard")

# ----------------------------
# Tab 1: Global Summary
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🌍 Global Summary", "🚨 Anomaly Detection", "📈 Daily Trends"])

with tab1:
    st.header("🌍 Cumulative Confirmed and Deaths Overview")

    # Fill missing country names or remove rows without country
    df_filtered = df[df["country"].notnull() & (df["confirmed"] > 0)]

    # 1. World Map
    latest_map_date = df_filtered["date"].max()
    latest_df = df_filtered[df_filtered["date"] == latest_map_date]

    map_fig = px.choropleth(latest_df,
        locations="country",
        locationmode="country names",
        color="confirmed",
        color_continuous_scale="Reds",
        title="Confirmed Cases Worldwide",
        height=500)
    map_fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(map_fig, use_container_width=True)

    # 2. Global Summary Cards
    agg_df = df_filtered.groupby("country", as_index=False).agg({"confirmed": "sum", "deaths": "sum"})
    total_confirmed = int(agg_df["confirmed"].sum())
    total_deaths = int(agg_df["deaths"].sum())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Confirmed Cases", f"{total_confirmed:,}")
    with col2:
        st.metric("Total Deaths", f"{total_deaths:,}")

    # 3. Yearly Trend Lines
    df_filtered["year"] = df_filtered["date"].dt.year
    yearly_df = df_filtered.groupby("year")[["confirmed", "deaths"]].sum().reset_index()

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=yearly_df["year"], y=yearly_df["confirmed"], name="Confirmed", mode="lines+markers"))
    line_fig.add_trace(go.Scatter(x=yearly_df["year"], y=yearly_df["deaths"], name="Deaths", mode="lines+markers"))
    line_fig.update_layout(title="Yearly Cumulative Confirmed vs Deaths", xaxis_title="Year", yaxis_title="Count")
    st.plotly_chart(line_fig, use_container_width=True)

# ----------------------------
# Tab 2: Anomaly Detection
# ----------------------------
with tab2:
    st.header("🚨 Country-Level Anomaly Detection")
    countries_with_data = df[df["daily_new_cases"] > 0]["country"].unique()
    selected_country = st.selectbox("Select Country", sorted(countries_with_data))

    df_country = df[(df["country"] == selected_country) & (df["daily_new_cases"] > 0)]

    fig_anomaly = px.box(df_country, x="country", y="daily_new_cases",
                         points="all", title="Daily New Cases Distribution (Outliers Highlighted)")
    st.plotly_chart(fig_anomaly, use_container_width=True)

# ----------------------------
# Tab 3: Daily Trends & Growth Rate
# ----------------------------
with tab3:
    st.header("📈 Daily Trends and Growth Rate")
    countries_with_data = df[df["daily_new_cases"] > 0]["country"].unique()
    selected_country = st.selectbox("Select Country for Trend Analysis", sorted(countries_with_data), key="country_trend")

    df_country = df[(df["country"] == selected_country)].copy()
    df_country = df_country.sort_values("date")

    df_country["growth_rate"] = df_country["confirmed"].pct_change().fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_country["date"], y=df_country["daily_new_cases"],
                             mode='lines', name='Daily New Cases'))
    fig.add_trace(go.Scatter(x=df_country["date"], y=df_country["growth_rate"] * 100,
                             mode='lines', name='Growth Rate (%)', yaxis="y2"))

    fig.update_layout(
        title=f"Daily New Cases and Growth Rate in {selected_country}",
        xaxis_title="Date",
        yaxis=dict(title="New Cases"),
        yaxis2=dict(title="Growth Rate (%)", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)