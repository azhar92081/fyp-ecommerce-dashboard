import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="E-Commerce Intelligence Pro", layout="wide")

# Custom CSS to make it pop
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1 { color: #1E3A8A; font-family: 'Helvetica Neue', sans-serif; }
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

# --- 2. SIDEBAR ---
st.sidebar.header("📂 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'OrderDate' in df.columns: df['OrderDate'] = pd.to_datetime(df['OrderDate'])
else:
    df = load_advanced_data()

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filters")
selected_categories = st.sidebar.multiselect("Categories", df['Category'].unique(), default=df['Category'].unique())
selected_regions = st.sidebar.multiselect("Regions", df['Region'].unique(), default=df['Region'].unique())

# Apply Filters
df = df[(df['Category'].isin(selected_categories)) & (df['Region'].isin(selected_regions))]

if df.empty:
    st.warning("No data matches your filters!")
    st.stop()

# --- 3. TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Predictive AI", "🤖 Customer Clusters"])

# -- TAB 1: EXECUTIVE OVERVIEW --
with tab1:
    # KPI Metrics
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
    
    model = LinearRegression().fit(daily_sales[['Days']], daily_sales['TotalAmount'])
    
    last_day = daily_sales['Days'].max()
    future_days = np.array([[last_day + i] for i in range(1, 15)])
    preds = model.predict(future_days)
    
    future_dates = [daily_sales['OrderDate'].max() + datetime.timedelta(days=i) for i in range(1, 15)]
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': preds})
    
    fig_forecast = px.line(forecast_df, x='Date', y='Forecast', title="AI Predicted Sales Trend",
                           markers=True, line_shape="spline", render_mode="svg")
    fig_forecast.update_traces(line_color='#10b981')
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.dataframe(forecast_df.style.format({'Forecast': '${:.2f}'}), use_container_width=True)

# -- TAB 3: CUSTOMER SEGMENTATION (K-Means) --
with tab3:
    st.header("🧠 Machine Learning Customer Clusters")
    
    # Feature Engineering
    cust_df = df.groupby('CustomerID').agg({'TotalAmount':'sum', 'OrderDate':'count'}).rename(columns={'OrderDate':'Orders'})
    
    # ML Model
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(cust_df)
    cust_df['Segment'] = kmeans.labels_.astype(str)
    
    # Visualization
    fig_cluster = px.scatter(cust_df, x='TotalAmount', y='Orders', color='Segment',
                             title="Customer Groups (Value vs. Frequency)",
                             labels={'TotalAmount':'Total Spend ($)', 'Orders':'Number of Visits'},
                             color_discrete_map={'0':'#636EFA', '1':'#EF553B', '2':'#00CC96'})
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.write("### Detailed Customer Segments")
    st.dataframe(cust_df, use_container_width=True)
