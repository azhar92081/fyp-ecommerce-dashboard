import streamlit as st
import pandas as pd
import datetime as dt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# --- PAGE CONFIGURATION & "WHITE-LABEL" HACK ---
# FIX: 'initial_sidebar_state="expanded"' locks the sidebar open automatically
st.set_page_config(page_title="E-Commerce Intelligence", layout="wide", page_icon="🛍️", initial_sidebar_state="expanded")

# FIX: Removed the 'header' hide command so you are never locked out of the sidebar toggle
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🛍️ E-commerce Sales & Customer Intelligence Dashboard")
st.markdown("Welcome to the offline analytics engine. Generate business insights and segment customers using Machine Learning.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your retail CSV file", type=['csv'])

# --- THE "BULLETPROOF" PRESENTATION MODE ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists("FYP_Perfect_Retail_Data.csv"):
    df = pd.read_csv("FYP_Perfect_Retail_Data.csv")
    st.sidebar.success("✅ Connected to Live Enterprise Database.")
else:
    st.info("👈 Please upload your E-commerce CSV file in the left sidebar to initialize the dashboard.")
    st.stop()
    
# --- UPGRADE 1: THE SAFETY NET ---
required_columns = ['CustomerID', 'InvoiceDate', 'Quantity', 'UnitPrice', 'Country', 'Description']
if not all(col in df.columns for col in required_columns):
    st.error(f"🚨 Error: Your CSV is missing required columns. Please make sure it has exactly: {required_columns}")
    st.stop()
    
# --- DATA CLEANING & ETL ---
df.dropna(subset=['CustomerID', 'Description'], inplace=True)
df = df[df['Quantity'] > 0]
df['TotalSales'] = df['Quantity'] * df['UnitPrice']
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# --- SIDEBAR: DATE FILTER & ML SETTINGS ---
st.sidebar.header("2. Filter Data")
min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['InvoiceDate'].dt.date >= start_date) & (df['InvoiceDate'].dt.date <= end_date)]
else:
    st.sidebar.warning("⚠️ Please select an end date to continue.")
    st.stop()

st.sidebar.header("3. Machine Learning Settings")
k_value = st.sidebar.slider("Select Number of Clusters (K)", min_value=2, max_value=6, value=4)

# --- UI TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Business KPIs", "🤖 Customer Segments (ML)"])

# TAB 1: DATA OVERVIEW
with tab1:
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.write(f"**Total Valid Transactions in Selected Range:** {df.shape[0]:,}")
    
# TAB 2: BUSINESS KPIs
with tab2:
    st.subheader("High-Level Business Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Gross Revenue", f"${df['TotalSales'].sum():,.2f}")
    col2.metric("Total Unique Orders", f"{df['InvoiceNo'].nunique():,}")
    col3.metric("Total Unique Customers", f"{df['CustomerID'].nunique():,}")
    
    st.divider()
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Daily Revenue Trend")
        daily_sales = df.groupby(df['InvoiceDate'].dt.date)['TotalSales'].sum().reset_index()
        fig_line = px.line(daily_sales, x='InvoiceDate', y='TotalSales', title="Revenue Generation Over Time", color_discrete_sequence=['#00E5FF'])
        fig_line.update_layout(font=dict(color="white", size=14), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_line, use_container_width=True, theme=None)

    with chart_col2:
        st.subheader("Top 5 Best-Selling Products")
        top_products = df.groupby('Description')['TotalSales'].sum().sort_values(ascending=True).tail(5).reset_index()
        fig_bar = px.bar(
            top_products, x='TotalSales', y='Description', orientation='h', 
            title="Revenue by Product", color_discrete_sequence=['#FF4B4B']
        )
        fig_bar.update_layout(
            font=dict(color="white", size=14), 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="", 
            margin=dict(l=150) 
        )
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)

    st.divider()
    st.subheader("Regional Revenue Distribution")
    country_sales = df.groupby('Country')['TotalSales'].sum().reset_index()
    fig_pie = px.pie(
        country_sales, values='TotalSales', names='Country', title="Total Sales by Country", 
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Vivid
    )
    fig_pie.update_traces(textfont=dict(color="white", size=16))
    fig_pie.update_layout(font=dict(color="white", size=14), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pie, use_container_width=True, theme=None)
    
# TAB 3: MACHINE LEARNING & CLUSTERING
with tab3:
    st.subheader("Unsupervised Customer Segmentation")
    st.write("Applying K-Means clustering to Recency, Frequency, and Monetary (RFM) metrics.")
    
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSales': 'sum'
    }).reset_index()
    rfm_df.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSales': 'Monetary'}, inplace=True)
    
    with st.spinner(f"🤖 Executing K-Means Clustering Algorithm for {k_value} segments..."):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
        kmeans = KMeans(n_clusters=k_value, random_state=42)
        rfm_df['Cluster'] = kmeans.fit_predict(scaled_features)
        
    fig_3d = px.scatter_3d(
        rfm_df, x='Recency', y='Frequency', z='Monetary', 
        color=rfm_df['Cluster'].astype(str), title="3D Visualization of Customer Cohorts",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_3d.update_layout(font=dict(color="white", size=12), paper_bgcolor="rgba(0,0,0,0)", scene=dict(
        xaxis=dict(color="white"), yaxis=dict(color="white"), zaxis=dict(color="white")
    ))
    st.plotly_chart(fig_3d, use_container_width=True, theme=None)
    
    st.divider()
    st.subheader("📊 Cluster Averages (Segment Breakdown)")
    cluster_summary = rfm_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2).reset_index()
    st.dataframe(cluster_summary, use_container_width=True)
    
    st.write("Ready to take action? Download this segmented list to send targeted marketing campaigns.")
    csv_export = rfm_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Segmented Customers (.csv)", data=csv_export,
        file_name='customer_segments_final.csv', mime='text/csv',
    )
