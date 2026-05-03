import streamlit as st
import pandas as pd
import datetime as dt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide", page_icon="🛍️")
st.title("🛍️ E-commerce Sales & Customer Intelligence Dashboard")
st.markdown("Welcome to the offline analytics engine. Upload your transactional data to generate business insights and segment customers using Machine Learning.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your retail CSV file", type=['csv'])

st.sidebar.header("2. Machine Learning Settings")
k_value = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=3)

# --- MAIN APPLICATION LOGIC ---
if uploaded_file is not None:
    # Read the data into Pandas
    df = pd.read_csv(uploaded_file)
    
    # --- UPGRADE 1: THE SAFETY NET ---
    # Added 'Country' to required columns to protect the new donut chart
    required_columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'Country']
    if not all(col in df.columns for col in required_columns):
        st.error(f"🚨 Error: Your CSV is missing required columns. Please make sure it has exactly: {required_columns}")
        st.stop() # Halts the script to prevent tracebacks
        
    st.success("✅ Dataset loaded and validated successfully!")
    
    # --- DATA CLEANING & ETL ---
    # Drop rows without Customer IDs, filter out negative quantities, create TotalSales
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # --- UI TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Business KPIs", "🤖 Customer Segments (ML)"])
    
    # TAB 1: DATA OVERVIEW
    with tab1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        st.write(f"**Total Valid Transactions:** {df.shape[0]:,}")
        
    # TAB 2: BUSINESS KPIs
    with tab2:
        st.subheader("High-Level Business Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Gross Revenue", f"${df['TotalSales'].sum():,.2f}")
        col2.metric("Total Unique Orders", f"{df['InvoiceNo'].nunique():,}")
        col3.metric("Total Unique Customers", f"{df['CustomerID'].nunique():,}")
        
        st.divider()
        st.subheader("Daily Revenue Trend")
        # Group by date for the line chart
        daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
        fig_line = px.line(daily_sales, x='InvoiceDate', y='TotalSales', title="Revenue Generation Over Time")
        st.plotly_chart(fig_line, use_container_width=True)

        # --- THE CHERRY ON TOP: REGIONAL DONUT CHART ---
        st.divider()
        st.subheader("Regional Revenue Distribution")
        
        # Group the data by Country
        country_sales = df.groupby('Country')['TotalSales'].sum().reset_index()
        
        # Create a beautiful donut chart
        fig_pie = px.pie(
            country_sales, 
            values='TotalSales', 
            names='Country', 
            title="Total Sales by Country", 
            hole=0.4, 
            color_discrete_sequence=px.colors.sequential.Tealgrn
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    # TAB 3: MACHINE LEARNING & CLUSTERING
    with tab3:
        st.subheader("Unsupervised Customer Segmentation")
        st.write("Applying K-Means clustering to Recency, Frequency, and Monetary (RFM) metrics.")
        
        # RFM Calculation Engine
        snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
        rfm_df = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalSales': 'sum'
        }).reset_index()
        rfm_df.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSales': 'Monetary'}, inplace=True)
        
        # --- UPGRADE 2: THE HOLLYWOOD SPINNER ---
        with st.spinner(f"🤖 Executing K-Means Clustering Algorithm for {k_value} segments... Please wait."):
            # Scale the data to normalize variance
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
            
            # Run the K-Means Model
            kmeans = KMeans(n_clusters=k_value, random_state=42)
            rfm_df['Cluster'] = kmeans.fit_predict(scaled_features)
            
        # 3D Scatter Plot using Plotly
        fig_3d = px.scatter_3d(
            rfm_df, x='Recency', y='Frequency', z='Monetary', 
            color=rfm_df['Cluster'].astype(str), 
            title="3D Visualization of Customer Cohorts"
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # --- UPGRADE 3: CLUSTER AVERAGES TABLE ---
        st.divider()
        st.subheader("📊 Cluster Averages (Segment Breakdown)")
        st.write("This table interprets the math, showing the average spending and behavior for each distinct segment.")
        cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2).reset_index()
        st.dataframe(cluster_summary, use_container_width=True)
        
        # --- UPGRADE 4: THE EXPORT BUTTON ---
        st.write("Ready to take action? Download this segmented list to send targeted marketing campaigns.")
        csv_export = rfm_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Segmented Customers (.csv)",
            data=csv_export,
            file_name='customer_segments_final.csv',
            mime='text/csv',
        )

else:
    # What shows up before a file is uploaded
    st.info("👈 Please upload your E-commerce CSV file in the left sidebar to initialize the dashboard.")
