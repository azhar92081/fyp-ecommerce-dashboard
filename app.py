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
import os
import google.generativeai as genai

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise Intelligence V10.0 (AI Edition)", layout="wide", page_icon="🛍️", initial_sidebar_state="expanded")

# --- SECURITY UTILS ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- CLOUD AUTO-PROVISIONING ENGINE ---
@st.cache_resource
def auto_provision_db():
    conn = sqlite3.connect('enterprise_backend.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if not cursor.fetchone():
        cursor.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE system_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, alert_type TEXT NOT NULL, message TEXT NOT NULL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", ('admin', hash_password("iub2026"), 'System Administrator'))
        conn.commit()
    conn.close()

auto_provision_db()

# --- CUSTOM CSS & DYNAMIC THEME ---
st.sidebar.header("⚙️ System Settings")
night_mode = st.sidebar.toggle("🌙 Enable Night Mode", value=True)

if night_mode:
    theme_css = "<style>.stApp { background-color: #0E1117; color: #FFFFFF; } #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
    chart_template = "plotly_dark"
    font_color = "#FFFFFF"; hover_bg = "#1E1E1E"; bg_color = "#0E1117"
    chart_palette = ["#00E5FF", "#FF007F", "#FFD60A", "#8A2BE2", "#00F5D4", "#FF4D00"] 
else:
    theme_css = "<style>.stApp { background-color: #F4F6F9; color: #000000; } #MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>"
    chart_template = "plotly_white"
    font_color = "#000000"; hover_bg = "#FFFFFF"; bg_color = "#F4F6F9"
    chart_palette = ["#0056D2", "#D32F2F", "#FBC02D", "#6A1B9A", "#2E7D32", "#E65100"] 
    
st.markdown(theme_css, unsafe_allow_html=True)

# --- ENTERPRISE SECURITY: SQL LOGIN ---
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
                    st.error("❌ Invalid security credentials.")
    st.stop()

# --- MAIN DASHBOARD ---
st.sidebar.success(f"✅ Authenticated as: {st.session_state['role']}")
if st.sidebar.button("🚪 Secure Logout"):
    st.session_state['logged_in'] = False
    st.rerun()

st.title("🛍️ Advanced E-commerce & Customer Intelligence")

# --- AI CONFIGURATION ---
st.sidebar.header("🧠 AI Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# --- DYNAMIC DATA INGESTION ---
st.sidebar.header("1. Database Management")
uploaded_file = st.sidebar.file_uploader("Upload CSV to Update SQL Database", type=['csv'])

if uploaded_file is not None:
    with st.spinner("Injecting data into SQLite Database..."):
        new_data = pd.read_csv(uploaded_file)
        conn = sqlite3.connect('enterprise_backend.db')
        new_data.to_sql('ecommerce_sales', conn, if_exists='replace', index=False)
        conn.close()
        st.cache_data.clear()
        st.sidebar.success("✅ Database Successfully Updated!")

# --- LIVE SQL DATA FETCHING ---
@st.cache_data(ttl=300) 
def load_data_from_sql():
    try:
        conn = sqlite3.connect('enterprise_backend.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ecommerce_sales'")
        if not cursor.fetchone():
            return pd.DataFrame() 

        df = pd.read_sql("SELECT * FROM ecommerce_sales", conn)
        conn.close()
        
        if df.empty: return df
            
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
        return pd.DataFrame() 

raw_df = load_data_from_sql()

if raw_df.empty:
    st.info("👈 System Architecture Online. Please upload a CSV file to initialize the SQL database.")
    st.stop()

# --- FILTERS ---
st.sidebar.header("2. Interactive Filters")
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

st.sidebar.header("3. Machine Learning Settings")
k_value = st.sidebar.slider("Select Customer Clusters (K)", min_value=2, max_value=6, value=4)

def trigger_alert(message, alert_type="WARNING"):
    conn = sqlite3.connect('enterprise_backend.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO system_alerts (alert_type, message) VALUES (?, ?)", (alert_type, message))
    conn.commit()
    conn.close()

# --- UI TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📈 KPIs", "🔍 Patterns", "🤖 ML Segments", "🌐 Web", "🔮 Forecast", "📩 Alerts", "🧠 AI Analyst"])

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
    col4.metric("Conversion", f"{(total_buyers / total_visitors) * 100 if total_visitors > 0 else 0:,.2f}%")

with tab2:
    top_products = df.groupby('Description')['TotalSales'].sum().sort_values(ascending=True).tail(5).reset_index()
    st.subheader("Top Performing Products")
    st.plotly_chart(px.bar(top_products, x='TotalSales', y='Description', orientation='h', color_discrete_sequence=[chart_palette[0]]), use_container_width=True)

with tab3:
    st.subheader("Unsupervised Customer Segmentation")
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm_df = df.groupby('CustomerID').agg({'InvoiceDate': lambda x: (snapshot_date - x.max()).days, 'InvoiceNo': 'nunique', 'TotalSales': 'sum'}).reset_index()
    rfm_df.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalSales': 'Monetary'}, inplace=True)
    scaler = StandardScaler(); scaled_features = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=k_value, random_state=42); rfm_df['Cluster'] = kmeans.fit_predict(scaled_features)
    st.plotly_chart(px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', color=rfm_df['Cluster'].astype(str), color_discrete_sequence=chart_palette), use_container_width=True)

with tab4:
    st.subheader("Simulated Google Analytics")
    st.plotly_chart(px.area(df.groupby('Date')['WebsiteVisitors'].first().reset_index(), x='Date', y='WebsiteVisitors', color_discrete_sequence=[chart_palette[1]]), use_container_width=True)

with tab5:
    daily_sales = df.groupby('Date')['TotalSales'].sum().reset_index()
    daily_sales['Ordinal'] = pd.to_datetime(daily_sales['Date']).apply(lambda x: x.toordinal())
    z = np.polyfit(daily_sales['Ordinal'], daily_sales['TotalSales'], 2); p = np.poly1d(z)
    future_dates = [daily_sales['Date'].max() + dt.timedelta(days=x) for x in range(1, 31)]
    predictions = np.maximum(p([d.toordinal() for d in pd.to_datetime(future_dates)]), 0)
    if len(predictions) > 0 and predictions[-1] < (predictions[0] * 0.85): trigger_alert("Automated Warning: Forecasted revenue drop detected.", "FORECAST_WARNING")
    fig_predict = go.Figure()
    fig_predict.add_trace(go.Scatter(x=daily_sales['Date'], y=daily_sales['TotalSales'], mode='lines', name='Historical Sales', line=dict(color=chart_palette[0])))
    fig_predict.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Forecast', line=dict(color=chart_palette[1], dash='dot')))
    st.plotly_chart(fig_predict, use_container_width=True)

with tab6:
    try:
        conn = sqlite3.connect('enterprise_backend.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='system_alerts'")
        if cursor.fetchone(): st.dataframe(pd.read_sql("SELECT * FROM system_alerts ORDER BY timestamp DESC LIMIT 10", conn), use_container_width=True, hide_index=True)
    except: pass

# --- TAB 7: GEMINI AI INTEGRATION ---
with tab7:
    st.subheader("🧠 Gemini Executive AI Analyst")
    st.write("Generative AI integration to synthesize database metrics into actionable natural language intelligence.")
    
    if api_key:
        if st.button("✨ Generate Live Executive Report"):
            with st.spinner("Connecting to Google Generative AI... analyzing database context..."):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    top_item = top_products.iloc[-1]['Description'] if not top_products.empty else "N/A"
                    context_prompt = f"""
                    Act as an expert Chief Financial Officer. I will provide you with the live metrics from my e-commerce dashboard database. 
                    Write a highly professional, 3-paragraph executive summary detailing our performance and offering one strategic recommendation.
                    
                    Here is the live data:
                    - Total Gross Revenue: ${total_revenue:,.2f}
                    - Total Unique Buyers: {total_buyers}
                    - Total Marketing Spend: ${total_ad_spend:,.2f}
                    - Highest Grossing Product: {top_item}
                    - Selected Date Range: {start_date} to {end_date}
                    """
                    response = model.generate_content(context_prompt)
                    st.success("✅ AI Analysis Complete")
                    st.markdown("### 📊 Automated Executive Intelligence Brief")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"API Error: Please check if your API key is valid. Details: {e}")
    else:
        st.warning("⚠️ Authentication Required: Please paste your Gemini API Key in the left sidebar to activate the AI Analyst.")
