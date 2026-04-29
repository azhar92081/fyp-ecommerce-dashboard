import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="E-Commerce Intelligence", layout="wide")
st.title("E-Commerce Sales & Customer Intelligence Dashboard")

# --- 1. DATA GENERATION (DUMMY DATA) ---
@st.cache_data
def load_advanced_data():
    np.random.seed(42)
    dates = pd.date_range(start='2025-01-01', end='2026-04-01', freq='D')
    data = {
        'OrderDate': np.random.choice(dates, 500),
        'CustomerID': np.random.randint(1000, 1050, 500),
        'TotalAmount': np.random.uniform(10.0, 500.0, 500),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 500),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 500) # Added Region!
    }
    df = pd.DataFrame(data)
    df = df.sort_values('OrderDate').reset_index(drop=True)
    return df

# --- 2. SIDEBAR UPLOAD & FILTERS ---
st.sidebar.header("1. Upload Custom Data")
uploaded_file = st.sidebar.file_uploader("Upload your E-commerce CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
else:
    df = load_advanced_data()

st.sidebar.markdown("---")
st.sidebar.header("2. Filter Data")

# Category Filter
if 'Category' in df.columns:
    available_categories = df['Category'].unique().tolist()
    selected_categories = st.sidebar.multiselect(
        "Select Categories:",
        options=available_categories,
        default=available_categories
    )
    if selected_categories:
        df = df[df['Category'].isin(selected_categories)]
    else:
        st.warning("Please select at least one category to view data.")
        st.stop()

# Region Filter
if 'Region' in df.columns:
    available_regions = df['Region'].unique().tolist()
    selected_regions = st.sidebar.multiselect(
        "Select Regions:",
        options=available_regions,
        default=available_regions
    )
    if selected_regions:
        df = df[df['Region'].isin(selected_regions)]
    else:
        st.warning("Please select at least one region to view data.")
        st.stop()

# --- 3. THE TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Executive Overview", "📈 Sales Forecasting", "👥 Customer Segmentation"])

# -- TAB 1: Overview --
with tab1:
    st.header("Business Performance Overview")
    st.write("Recent Transactions")
    st.dataframe(df.head(15), use_container_width=True)
    
    # Show charts side-by-side using columns
    col1, col2 = st.columns(2)
    with col1:
        if 'Category' in df.columns:
            st.write("Sales by Category")
            category_sales = df.groupby('Category')['TotalAmount'].sum()
            st.bar_chart(category_sales)
            
    with col2:
        if 'Region' in df.columns:
            st.write("Sales by Region")
            region_sales = df.groupby('Region')['TotalAmount'].sum()
            st.bar_chart(region_sales)

# -- TAB 2: Linear Regression --
with tab2:
    st.header("14-Day Sales Forecast")
    st.markdown("Predictive analytics based on historical data using Scikit-Learn.")
    
    # Prepare data for ML
    daily_sales = df.groupby('OrderDate')['TotalAmount'].sum().reset_index()
    daily_sales['DaysSinceStart'] = (daily_sales['OrderDate'] - daily_sales['OrderDate'].min()).dt.days
    
    X = daily_sales[['DaysSinceStart']]
    y = daily_sales['TotalAmount']
    
    # Train Model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 14 days
    last_day = daily_sales['DaysSinceStart'].max()
    future_days = np.array([[last_day + i] for i in range(1, 15)])
    predictions = model.predict(future_days)
    
    # Create forecast dataframe
    future_dates = [daily_sales['OrderDate'].max() + datetime.timedelta(days=i) for i in range(1, 15)]
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Sales': predictions})
    
    st.line_chart(forecast_df.set_index('Date'))
    st.dataframe(forecast_df)

# -- TAB 3: K-Means Clustering --
with tab3:
    st.header("Customer Intelligence")
    st.markdown("Automated customer grouping using K-Means Clustering.")
    
    # Group by customer
    customer_data = df.groupby('CustomerID').agg({
        'TotalAmount': 'sum',
        'OrderDate': 'count'
    }).rename(columns={'OrderDate': 'TotalOrders'})
    
    # Train K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_data['Customer Segment'] = kmeans.fit_predict(customer_data)
    
    # Rename clusters for business value
    segment_map = {0: "Standard", 1: "High Value", 2: "Occasional"}
    customer_data['Customer Segment'] = customer_data['Customer Segment'].map(segment_map)
    
    st.write("Customer Groupings based on Purchase History")
    st.dataframe(customer_data.sort_values('TotalAmount', ascending=False))
