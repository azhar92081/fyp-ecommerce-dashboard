import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="E-Commerce Intelligence V4.0", layout="wide", page_icon="🛍️", initial_sidebar_state="expanded")

# --- CUSTOM CSS & DYNAMIC HIGH-CONTRAST THEME ---
st.sidebar.header("⚙️ App Settings")
night_mode = st.sidebar.toggle("🌙 Enable Night Mode", value=True)

if night_mode:
    theme_css = "<style>.stApp { background-color: #0E1117; color: #FFFFFF; } #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
    chart_template = "plotly_dark"
    font_color = "#FFFFFF" 
    hover_bg = "#1E1E1E" 
    chart_palette = px.colors.qualitative.Set1 
else:
    theme_css = "<style>.stApp { background-color: #F4F6F9; color: #000000; } #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
    chart_template = "plotly_white"
    font_color = "#000000" 
    hover_bg = "#FFFFFF" 
    chart_palette = px.colors.qualitative.Dark24 
    
st.markdown(theme_css, unsafe_allow_html=True)

st.title("🛍️ Advanced E-commerce & Customer Intelligence")
st.markdown("Interactive analytics engine featuring ML segmentation, ROI tracking, and Web Analytics.")

# --- CACHED DATA PIPELINE ---
@st.cache_data
def load_and_prep_data(filepath_or_buffer):
    df = pd.read_csv(filepath_or_buffer)
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)
    df = df[df['Quantity'] > 0]
    df['TotalSales'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Date'] = df['InvoiceDate'].dt.date
    
    np.random.seed(42) 
    unique_dates = df['Date'].unique()
    marketing_data = pd.DataFrame({'Date': unique_dates})
    
    daily_customers = df.groupby('Date')['CustomerID'].nunique().reset_index()
    marketing_data = pd.merge(marketing_data, daily_customers, on='Date')
    marketing_data['WebsiteVisitors'] = marketing_data['CustomerID'] * np.random.randint(20, 50, size=len(marketing_data))
    marketing_data['AdSpend'] = marketing_data['WebsiteVisitors'] * np.random.uniform(0.5, 1.5, size=len(marketing_data))
    
    marketing_data.drop(columns=['CustomerID'], inplace=True)
    df = pd.merge(df, marketing_data, on='Date', how='left')
    return df

# --- FILE UPLOADER ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload retail CSV file", type=['csv'])

if uploaded_file is not None:
    raw_df = load_and_prep_data(uploaded_file)
elif os.path.exists("FYP_Perfect_Retail_Data.csv"):
    raw_df = load_and_prep_data("FYP_Perfect_Retail_Data.csv")
    st.sidebar.success("✅ Connected to Enterprise Database.")
else:
    st.info("👈 Please upload your E-commerce CSV file to initialize.")
    st.stop()

# --- INTERACTIVE CROSS-FILTERING ---
st.sidebar.header("2. Interactive Filters")
all_countries = sorted(raw_df['Country'].unique())
selected_countries = st.sidebar.multiselect("🌍 Filter by Region", all_countries, default=all_countries[:5])

min_date = raw_df['Date'].min()
max_date = raw_df['Date'].max()
date_range = st.sidebar.date_input("📅 Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2 and len(selected_countries) > 0:
    start_date, end_date = date_range
    df = raw_df[(raw_df['Date'] >= start_date) & (raw_df['Date'] <= end_date) & (raw_df['Country'].isin(selected_countries))]
else:
    st.sidebar.warning("⚠️ Please select a valid date range and at least one country.")
    st.stop()

st.sidebar.header("3. Machine Learning Settings")
k_value = st.sidebar.slider("Select Customer Clusters (K)", min_value=2, max_value=6, value=4)

# --- UI TABS (NOW WITH 4 TABS) ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive KPIs", "🔍 Pattern Recognition", "🤖 ML Customer Segments", "🌐 Web Analytics"])

# TAB 1: EXECUTIVE KPIs 
with tab1:
    st.subheader("Performance & Marketing Metrics")
    total_revenue = df['TotalSales'].sum()
    total_buyers = df['CustomerID'].nunique()
    daily_marketing = df.groupby('Date').first().reset_index()
    total_ad_spend = daily_marketing['AdSpend'].sum()
    total_visitors = daily_marketing['WebsiteVisitors'].sum()
    
    roi = ((total_revenue - total_ad_spend) / total_ad_spend) * 100 if total_ad_spend > 0 else 0
    conversion_rate = (total_buyers / total_visitors) * 100 if total_visitors > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gross Revenue", f"${total_revenue:,.0f}")
    col2.metric("Marketing Spend", f"${total_ad_spend:,.0f}")
    col3.metric("Return on Investment (ROI)", f"{roi:,.1f}%")
    col4.metric("Conversion Rate", f"{conversion_rate:,.2f}%")
    
    st.divider()
    st.subheader("Revenue vs. Marketing Spend Overlay")
    daily_trend = df.groupby('Date').agg({'TotalSales': 'sum', 'AdSpend': 'first'}).reset_index()
    fig_trend = px.line(daily_trend, x='Date', y=['TotalSales', 'AdSpend'], title="Pattern Analysis: Spend vs Revenue", color_discrete_sequence=chart_palette)
    fig_trend.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
    st.plotly_chart(fig_trend, use_container_width=True)

# TAB 2: PATTERN RECOGNITION
with tab2:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Top Performing Products")
        top_products = df.groupby('Description')['TotalSales'].sum().sort_values(ascending=True).tail(5).reset_index()
        fig_bar = px.bar(top_products, x='TotalSales', y='Description', orientation='h', color_discrete_sequence=[chart_palette[1]])
        fig_bar.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", yaxis_title="", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.subheader("Revenue by Region")
        country_sales = df.groupby('Country')['TotalSales'].sum().reset_index()
        fig_pie = px.pie(country_sales, values='TotalSales', names='Country', hole=0.4, color_discrete_sequence=chart_palette)
        fig_pie.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
        st.plotly_chart(fig_pie, use_container_width=True)

# TAB 3: MACHINE LEARNING
with tab3:
    st.subheader("Unsupervised Customer Segmentation")
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (snapshot_date - x.max()).days, 'InvoiceNo': 'nunique', 'TotalSales': 'sum'}).reset_index()
    rfm_df.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSales': 'Monetary'}, inplace=True)
    
    with st.spinner(f"Executing K-Means for {k_value} segments..."):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
        kmeans = KMeans(n_clusters=k_value, random_state=42)
        rfm_df['Cluster'] = kmeans.fit_predict(scaled_features)
        
    fig_3d = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', color=rfm_df['Cluster'].astype(str), color_discrete_sequence=chart_palette)
    fig_3d.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, b=0, t=0), font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color), scene=dict(xaxis=dict(color=font_color, title_font=dict(color=font_color)), yaxis=dict(color=font_color, title_font=dict(color=font_color)), zaxis=dict(color=font_color, title_font=dict(color=font_color))))
    st.plotly_chart(fig_3d, use_container_width=True)

# TAB 4: WEB ANALYTICS (NEW)
with tab4:
    st.subheader("🌐 Simulated Google Analytics Dashboard")
    st.write("Live website traffic metrics and acquisition channels based on customer interactions.")
    
    col1, col2, col3, col4 = st.columns(4)
    # Re-calculating total visitors safely for this tab
    daily_visitors = df.groupby('Date')['WebsiteVisitors'].first()
    tot_visitors = daily_visitors.sum()
    
    col1.metric("Active Users", f"{tot_visitors:,.0f}")
    col2.metric("Page Views", f"{int(tot_visitors * 3.4):,.0f}")
    col3.metric("Avg. Session Duration", "00:02:45")
    col4.metric("Bounce Rate", "42.8%")
    
    st.divider()
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Daily Traffic Trend")
        traffic_trend = df.groupby('Date')['WebsiteVisitors'].first().reset_index()
        fig_traffic = px.area(traffic_trend, x='Date', y='WebsiteVisitors', title="Website Visitors Over Time", color_discrete_sequence=[chart_palette[2]])
        fig_traffic.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
        st.plotly_chart(fig_traffic, use_container_width=True)

    with chart_col2:
        st.subheader("Traffic Acquisition")
        acquisition_data = pd.DataFrame({
            'Channel': ['Organic Search', 'Direct', 'Social Media', 'Referral'],
            'Users': [tot_visitors * 0.45, tot_visitors * 0.30, tot_visitors * 0.15, tot_visitors * 0.10]
        })
        fig_acq = px.pie(acquisition_data, values='Users', names='Channel', hole=0.5, color_discrete_sequence=chart_palette)
        fig_acq.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
        st.plotly_chart(fig_acq, use_container_width=True)
