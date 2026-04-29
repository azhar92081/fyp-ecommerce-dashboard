import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# 1. Page Configuration
st.set_page_config(page_title="E-Commerce Intelligence", layout="wide")

st.title("E-Commerce Sales & Customer Intelligence Dashboard")
st.markdown("---")

# 2. Advanced Data Generation
@st.cache_data
def load_advanced_data():
    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=90)
    order_dates = np.random.choice(dates, 500)
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Toys']
    regions = ['North', 'South', 'East', 'West']
    
    data = pd.DataFrame({
        'OrderID': range(1001, 1501),
        'OrderDate': order_dates,
        'CustomerID': np.random.randint(100, 300, size=500),
        'Category': np.random.choice(categories, 500, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        'Region': np.random.choice(regions, 500),
        'Quantity': np.random.randint(1, 6, size=500),
        'UnitPrice': np.random.uniform(10.0, 500.0, size=500).round(2),
        'CustomerAge': np.random.randint(18, 65, size=500)
    })
    
    data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
    data = data.sort_values('OrderDate').reset_index(drop=True)
    return data

# --- NEW: File Uploader ---
st.sidebar.header("Upload Custom Data")
uploaded_file = st.sidebar.file_uploader("Upload your E-commerce CSV", type=["csv"])

if uploaded_file is not None:
    # If a user drops a file, read it!
    df = pd.read_csv(uploaded_file)
    
    # Ensure the date column is formatted correctly for the ML models
    if 'OrderDate' in df.columns:
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
else:
    # Fallback: If no file is uploaded, show the dummy data so the app isn't blank
    df = load_advanced_data()
# --------------------------

# 3. Sidebar Filtering
st.sidebar.header("Dashboard Filters")

selected_category = st.sidebar.multiselect(
    "Select Categories:",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

selected_region = st.sidebar.multiselect(
    "Select Regions:",
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

filtered_df = df[(df['Category'].isin(selected_category)) & (df['Region'].isin(selected_region))]

# 4. Top Level KPIs
st.subheader("Executive Summary")
col1, col2, col3, col4 = st.columns(4)

total_revenue = filtered_df['TotalAmount'].sum()
total_orders = len(filtered_df)
avg_order_value = filtered_df['TotalAmount'].mean() if total_orders > 0 else 0
unique_customers = filtered_df['CustomerID'].nunique()

col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Total Orders", f"{total_orders}")
col3.metric("Avg Order Value", f"${avg_order_value:,.2f}")
col4.metric("Unique Customers", f"{unique_customers}")

st.markdown("---")

# 5. Visualizations 
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Revenue by Category")
    if not filtered_df.empty:
        category_sales = filtered_df.groupby('Category')['TotalAmount'].sum()
        st.bar_chart(category_sales)

with col_chart2:
    st.subheader("Daily Sales Trend")
    if not filtered_df.empty:
        daily_sales = filtered_df.groupby('OrderDate')['TotalAmount'].sum()
        st.line_chart(daily_sales)

st.markdown("---")

# 6. Predictive Analytics (Linear Regression)
st.subheader("Sales Forecasting (Next 14 Days)")

if not filtered_df.empty:
    daily_totals = filtered_df.groupby('OrderDate')['TotalAmount'].sum().reset_index()
    daily_totals['DateOrdinal'] = daily_totals['OrderDate'].map(datetime.datetime.toordinal)
    
    X = daily_totals[['DateOrdinal']]
    y = daily_totals['TotalAmount']
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = daily_totals['OrderDate'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14)
    future_ordinals = pd.DataFrame({'DateOrdinal': [d.toordinal() for d in future_dates]})
    
    predictions = model.predict(future_ordinals)
    predictions = np.maximum(predictions, 0) 
    
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Sales': predictions})
    
    col_forecast_chart, col_forecast_data = st.columns([2, 1])
    
    with col_forecast_chart:
        historical_plot = daily_totals[['OrderDate', 'TotalAmount']].rename(columns={'OrderDate': 'Date', 'TotalAmount': 'Historical Sales'}).set_index('Date')
        predicted_plot = forecast_df.set_index('Date')
        combined_plot = pd.concat([historical_plot, predicted_plot])
        st.line_chart(combined_plot)
        
    with col_forecast_data:
        st.dataframe(forecast_df, use_container_width=True)
        # Export Forecast Button
        csv_forecast = forecast_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Forecast CSV", data=csv_forecast, file_name='sales_forecast.csv', mime='text/csv')

st.markdown("---")

# 7. NEW: Customer Segmentation (K-Means Clustering)
st.subheader("Customer Segmentation Analysis")
st.markdown("Using Scikit-Learn `KMeans` to group customers into behavior tiers based on their spending and order frequency.")

if not filtered_df.empty and len(filtered_df['CustomerID'].unique()) >= 3:
    # Prepare data for clustering: Group by Customer
    customer_data = filtered_df.groupby('CustomerID').agg({
        'TotalAmount': 'sum',
        'OrderID': 'count'
    }).reset_index()
    customer_data.rename(columns={'TotalAmount': 'Total Spend', 'OrderID': 'Total Orders'}, inplace=True)
    
    # Apply K-Means Clustering (3 Clusters)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer_data['Cluster'] = kmeans.fit_predict(customer_data[['Total Spend', 'Total Orders']])
    
    # Map cluster numbers to business logic tiers
    cluster_mapping = {0: "Standard", 1: "High Value", 2: "Occasional"}
    customer_data['Customer Tier'] = customer_data['Cluster'].map(cluster_mapping)
    
    col_cluster_chart, col_cluster_data = st.columns([2, 1])
    
    with col_cluster_chart:
        st.scatter_chart(data=customer_data, x='Total Orders', y='Total Spend', color='Customer Tier')
        
    with col_cluster_data:
        st.dataframe(customer_data[['CustomerID', 'Total Spend', 'Total Orders', 'Customer Tier']], use_container_width=True)
        # Export Segmentation Button
        csv_customers = customer_data.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Customer Tiers CSV", data=csv_customers, file_name='customer_segmentation.csv', mime='text/csv')
else:
    st.warning("Not enough customer data to perform segmentation. Please expand your filters.")

st.markdown("---")

# 8. Raw Data & Export
with st.expander("View Raw E-Commerce Data"):
    st.dataframe(filtered_df, use_container_width=True)
    csv_raw = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Raw Data CSV", data=csv_raw, file_name='raw_ecommerce_data.csv', mime='text/csv')
