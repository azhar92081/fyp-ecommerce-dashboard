import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sqlite3
import hashlib

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise Intelligence V8.0 (Full-Stack)", layout="wide", page_icon="🛍️", initial_sidebar_state="expanded")

# --- CUSTOM CSS & DYNAMIC THEME ---
st.sidebar.header("⚙️ System Settings")
night_mode = st.sidebar.toggle("🌙 Enable Night Mode", value=True)

if night_mode:
    theme_css = "<style>.stApp { background-color: #0E1117; color: #FFFFFF; } #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
    chart_template = "plotly_dark"
    font_color = "#FFFFFF" 
    hover_bg = "#1E1E1E" 
    bg_color = "#0E1117"
    chart_palette = ["#00E5FF", "#FF007F", "#FFD60A", "#8A2BE2", "#00F5D4", "#FF4D00"] 
else:
    theme_css = "<style>.stApp { background-color: #F4F6F9; color: #000000; } #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
    chart_template = "plotly_white"
    font_color = "#000000" 
    hover_bg = "#FFFFFF" 
    bg_color = "#F4F6F9"
    chart_palette = ["#0056D2", "#D32F2F", "#FBC02D", "#6A1B9A", "#2E7D32", "#E65100"] 
    
st.markdown(theme_css, unsafe_allow_html=True)

# --- SECURITY UTILS ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- ENTERPRISE SECURITY: SQL DATABASE LOGIN ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
    st.session_state['role'] = None

if not st.session_state['logged_in']:
    st.markdown(f"<h1 style='text-align: center; color: {chart_palette[0]};'>🔒 Enterprise Secure Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Live Database Connection Active. Awaiting Authentication.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        with st.form("login_form"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            submit = st.form_submit_button("Authenticate via SQL")
            
            if submit:
                # Live query to the database for authentication
                conn = sqlite3.connect('enterprise_backend.db')
                cursor = conn.cursor()
                cursor.execute("SELECT role FROM users WHERE username=? AND password_hash=?", (user, hash_password(pwd)))
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    st.session_state['logged_in'] = True
                    st.session_state['role'] = result[0]
                    st.rerun()
                else:
                    st.error("❌ Invalid security credentials or database unreachable.")
    st.stop()

# --- MAIN DASHBOARD ---
st.sidebar.success(f"✅ Authenticated as: {st.session_state['role']}")
if st.sidebar.button("🚪 Secure Logout"):
    st.session_state['logged_in'] = False
    st.rerun()

st.title("🛍️ Advanced E-commerce & Customer Intelligence")
st.markdown("Full-Stack Analytics Engine powered by SQLite Relational Database.")

# --- LIVE SQL DATA FETCHING PIPELINE ---
@st.cache_data(ttl=300) # Cache clears every 5 mins to check for new DB records
def load_data_from_sql():
    try:
        conn = sqlite3.connect('enterprise_backend.db')
        df = pd.read_sql("SELECT * FROM ecommerce_sales", conn)
        conn.close()
        
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
    except Exception as e:
        return pd.DataFrame() # Return empty if DB fails

raw_df = load_data_from_sql()

if raw_df.empty:
    st.error("🚨 CRITICAL ERROR: Unable to establish connection to SQL Database.")
    st.stop()
else:
    st.sidebar.success("📡 DB Connection: STABLE")

# --- FILTERS ---
st.sidebar.header("Interactive Filters")
all_countries = sorted(raw_df['Country'].unique())
selected_countries = st.sidebar.multiselect("🌍 Filter by Region", all_countries, default=all_countries[:5])
min_date = raw_df['Date'].min()
max_date = raw_df['Date'].max()
date_range = st.sidebar.date_input("📅 Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

if len(date_range) == 2 and len(selected_countries) > 0:
    start_date, end_date = date_range
    df = raw_df[(raw_df['Date'] >= start_date) & (raw_df['Date'] <= end_date) & (raw_df['Country'].isin(selected_countries))]
else:
    st.sidebar.warning("⚠️ Please select a valid date range and at least one country.")
    st.stop()

st.sidebar.header("Machine Learning Settings")
k_value = st.sidebar.slider("Select Customer Clusters (K)", min_value=2, max_value=6, value=4)

# --- SYSTEM ALERTS AUTOMATION ---
def trigger_alert(message, alert_type="WARNING"):
    conn = sqlite3.connect('enterprise_backend.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO system_alerts (alert_type, message) VALUES (?, ?)", (alert_type, message))
    conn.commit()
    conn.close()

# --- UI TABS ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 KPIs", "🔍 Patterns", "🤖 ML Segments", "🌐 Web Analytics", "🔮 30-Day Forecast", "📩 System Alerts"])

# TAB 1: EXECUTIVE KPIs 
with tab1:
    total_revenue = df['TotalSales'].sum()
    total_buyers = df['CustomerID'].nunique()
    daily_marketing = df.groupby('Date').first().reset_index()
    total_ad_spend = daily_marketing['AdSpend'].sum()
    total_visitors = daily_marketing['WebsiteVisitors'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gross Revenue", f"${total_revenue:,.0f}")
    col2.metric("Marketing Spend", f"${total_ad_spend:,.0f}")
    col3.metric("ROI", f"{((total_revenue - total_ad_spend) / total_ad_spend) * 100 if total_ad_spend > 0 else 0:,.1f}%")
    col4.metric("Conversion Rate", f"{(total_buyers / total_visitors) * 100 if total_visitors > 0 else 0:,.2f}%")
    
    st.divider()
    daily_trend = df.groupby('Date').agg({'TotalSales': 'sum', 'AdSpend': 'first'}).reset_index()
    fig_trend = px.line(daily_trend, x='Date', y=['TotalSales', 'AdSpend'], title="Pattern Analysis: Spend vs Revenue", color_discrete_sequence=chart_palette)
    fig_trend.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
    fig_trend.update_traces(line=dict(width=3)) 
    st.plotly_chart(fig_trend, use_container_width=True)

# TAB 2: PATTERN RECOGNITION 
with tab2:
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Top Performing Products")
        top_products = df.groupby('Description')['TotalSales'].sum().sort_values(ascending=True).tail(5).reset_index()
        fig_bar = px.bar(top_products, x='TotalSales', y='Description', orientation='h', color_discrete_sequence=[chart_palette[0]])
        fig_bar.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", yaxis_title="", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
        fig_bar.update_traces(marker=dict(line=dict(color=bg_color, width=1.5)))
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        st.subheader("Revenue by Region")
        country_sales = df.groupby('Country')['TotalSales'].sum().reset_index()
        fig_pie = px.pie(country_sales, values='TotalSales', names='Country', hole=0.4, color_discrete_sequence=chart_palette)
        fig_pie.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
        fig_pie.update_traces(marker=dict(line=dict(color=bg_color, width=2.5)))
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
    fig_3d.update_traces(marker=dict(size=6, line=dict(width=1.5, color='#000000')))
    st.plotly_chart(fig_3d, use_container_width=True)

# TAB 4: WEB ANALYTICS 
with tab4:
    st.subheader("🌐 Simulated Google Analytics Dashboard")
    col1, col2, col3, col4 = st.columns(4)
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
        fig_traffic = px.area(traffic_trend, x='Date', y='WebsiteVisitors', color_discrete_sequence=[chart_palette[1]])
        fig_traffic.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
        st.plotly_chart(fig_traffic, use_container_width=True)

    with chart_col2:
        st.subheader("Traffic Acquisition")
        acquisition_data = pd.DataFrame({'Channel': ['Organic Search', 'Direct', 'Social Media', 'Referral'], 'Users': [tot_visitors * 0.45, tot_visitors * 0.30, tot_visitors * 0.15, tot_visitors * 0.10]})
        fig_acq = px.pie(acquisition_data, values='Users', names='Channel', hole=0.5, color_discrete_sequence=chart_palette)
        fig_acq.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
        fig_acq.update_traces(marker=dict(line=dict(color=bg_color, width=2.5)))
        st.plotly_chart(fig_acq, use_container_width=True)

# TAB 5: PREDICTIVE ANALYTICS & AUTOMATION TRIGGERS
with tab5:
    st.subheader("🔮 Machine Learning Sales Forecast & Automation")
    
    daily_sales = df.groupby('Date')['TotalSales'].sum().reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    daily_sales['Ordinal'] = daily_sales['Date'].apply(lambda x: x.toordinal())
    
    z = np.polyfit(daily_sales['Ordinal'], daily_sales['TotalSales'], 2)
    p = np.poly1d(z)
    
    last_date = daily_sales['Date'].max()
    future_dates = [last_date + dt.timedelta(days=x) for x in range(1, 31)]
    future_ordinals = [d.toordinal() for d in future_dates]
    predictions = p(future_ordinals)
    predictions = np.maximum(predictions, 0) # No negative sales
    
    # AUTOMATION TRIGGER: If the last predicted day is 15% lower than the first predicted day, fire an alert!
    if predictions[-1] < (predictions[0] * 0.85):
        trigger_alert(f"Automated Warning: Forecasted revenue drop detected in the next 30 days for selected regions.", "FORECAST_WARNING")
    
    fig_predict = go.Figure()
    fig_predict.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['TotalSales'], mode='lines', name='Historical Sales', line=dict(color=chart_palette[0], width=2)))
    fig_predict.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='30-Day Forecast', line=dict(color=chart_palette[1], width=3, dash='dot')))
    fig_predict.update_layout(template=chart_template, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=font_color), hoverlabel=dict(bgcolor=hover_bg, font_size=14, font_color=font_color))
    st.plotly_chart(fig_predict, use_container_width=True)

# TAB 6: BACKEND SYSTEM ALERTS
with tab6:
    st.subheader("📩 Backend Automation & System Alerts")
    st.write("Live logs of automated system triggers and warnings.")
    
    try:
        conn = sqlite3.connect('enterprise_backend.db')
        alerts_df = pd.read_sql("SELECT * FROM system_alerts ORDER BY timestamp DESC LIMIT 10", conn)
        conn.close()
        
        if not alerts_df.empty:
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No critical alerts in the system log.")
    except:
        st.error("Could not fetch alerts table.")
