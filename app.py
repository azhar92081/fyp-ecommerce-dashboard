import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, silhouette_score
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="E-Commerce Intelligence Pro", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff !important; padding: 20px !important; border-radius: 10px !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important; }
    [data-testid="stMetricLabel"] * { color: #4b5563 !important; font-weight: bold !important; }
    [data-testid="stMetricValue"] * { color: #111827 !important; }
    h1 { color: #1E3A8A !important; font-family: 'Helvetica Neue', sans-serif !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 E-Commerce Sales & Customer Intelligence")

# --- 1. DATA GENERATION ---
@st.cache_data
def load_advanced_data():
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', end='2026-04-01', freq='D')
    data = {
        'OrderDate': np.random.choice(dates, 500),
        'CustomerID': np.random.randint(1000, 1050, 500),
        'TotalAmount': np.random.uniform(10.0, 500.0, 500),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 500),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 500)
    }
    df = pd.DataFrame(data)
    return df.sort_values('OrderDate').reset_index(drop=True)

# --- 2. SIDEBAR & CRASH PREVENTION ---
st.sidebar.header("📂 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Crash Prevention Check
    required_cols = ['OrderDate', 'CustomerID', 'TotalAmount', 'Category', 'Region']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.sidebar.error(f"⚠️ Uploaded CSV is missing columns: {', '.join(missing_cols)}.")
        st.stop()
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
else:
    df = load_advanced_data()

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filters")

# Date Filter
min_date = df['OrderDate'].min().date()
max_date = df['OrderDate'].max().date()
date_range = st.sidebar.date_input("Select Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['OrderDate'].dt.date >= start_date) & (df['OrderDate'].dt.date <= end_date)]

selected_categories = st.sidebar.multiselect("Categories", df['Category'].unique(), default=df['Category'].unique())
selected_regions = st.sidebar.multiselect("Regions", df['Region'].unique(), default=df['Region'].unique())

df = df[(df['Category'].isin(selected_categories)) & (df['Region'].isin(selected_regions))]

if df.empty:
    st.warning("No data matches your filters! Please adjust your selections.")
    st.stop()

# --- 3. TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Predictive AI", "🤖 Customer Clusters"])

# -- TAB 1: EXECUTIVE OVERVIEW --
with tab1:
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sales", f"${df['TotalAmount'].sum():,.2f}")
    m2.metric("Total Orders", f"{len(df):,}")
    m3.metric("Avg Order Value", f"${df['TotalAmount'].mean():,.2f}")
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        fig_cat = px.bar(df.groupby('Category')['TotalAmount'].sum().reset_index(), 
                         x='Category', y='TotalAmount', color='Category',
                         title="Revenue by Category", template="plotly_white")
        st.plotly_chart(fig_cat, use_container_width=True)
    with c2:
        fig_reg = px.pie(df, values='TotalAmount', names='Region', hole=0.4,
                         title="Regional Market Share", color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_reg, use_container_width=True)

# -- TAB 2: SALES FORECASTING (Linear Regression) --
with tab2:
    st.header("🔮 14-Day Sales Projection")
    daily_sales = df.groupby('OrderDate')['TotalAmount'].sum().reset_index()
    daily_sales['Days'] = (daily_sales['OrderDate'] - daily_sales['OrderDate'].min()).dt.days
    
    if len(daily_sales) < 5:
        st.warning("Not enough data to run a forecast. Please expand your date range.")
    else:
        X = daily_sales[['Days']]
        y = daily_sales['TotalAmount']
        
        model = LinearRegression().fit(X, y)
        train_preds = model.predict(X)
        
        st.markdown("### 🧮 Model Evaluation")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("R-Squared (Accuracy)", f"{r2_score(y, train_preds):.4f}")
        col_m2.metric("Mean Absolute Error", f"${mean_absolute_error(y, train_preds):.2f}")
        st.info("💡 **Academic Note:** An R² closer to 1.0 indicates a strong model fit. The MAE shows the average dollar variance in our predictions.")
        
        last_day = daily_sales['Days'].max()
        future_days = np.array([[last_day + i] for i in range(1, 15)])
        preds = model.predict(future_days)
        future_dates = [daily_sales['OrderDate'].max() + datetime.timedelta(days=i) for i in range(1, 15)]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': preds})
        
        fig_forecast = px.line(forecast_df, x='Date', y='Forecast', title="AI Predicted Sales Trend", markers=True)
        fig_forecast.update_traces(line_color='#10b981')
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.download_button("📥 Download Forecast Data (CSV)", data=forecast_df.to_csv(index=False), file_name="sales_forecast.csv", mime="text/csv")

# -- TAB 3: CUSTOMER SEGMENTATION (K-Means) --
with tab3:
    st.header("🧠 Machine Learning Customer Clusters")
    cust_df = df.groupby('CustomerID').agg({'TotalAmount':'sum', 'OrderDate':'count'}).rename(columns={'OrderDate':'Orders'})
    
    if len(cust_df) < 3:
        st.warning("Not enough customers to form 3 clusters.")
    else:
        features = cust_df[['TotalAmount', 'Orders']]
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(features)
        cust_df['Segment'] = kmeans.labels_.astype(str)
        
        st.markdown("### 🧮 Clustering Evaluation")
        st.metric("Silhouette Score", f"{silhouette_score(features, kmeans.labels_):.4f}")
        st.info("💡 **Academic Note:** The Silhouette Score (ranging from -1 to 1) measures cluster separation. A positive score confirms that our segments are distinct.")
        
        fig_cluster = px.scatter(cust_df, x='TotalAmount', y='Orders', color='Segment', title="Customer Groups (Value vs. Frequency)")
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.download_button("📥 Download Customer Segments (CSV)", data=cust_df.to_csv(), file_name="customer_segments.csv", mime="text/csv")
