"""
===============================================================================
INTEGRATED BUSINESS ANALYTICS SYSTEM
Main Application Entry Point
===============================================================================
"""

import streamlit as st
import pandas as pd
import os
import json

# Page config
st.set_page_config(
    page_title="Business Analytics System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] .stRadio label {
    color: #e0e0ff !important;
    font-weight: 500;
}

/* KPI Cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.3s, box-shadow 0.3s;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
}
.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0;
}
.kpi-label {
    font-size: 0.85rem;
    color: #a5b4fc;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.kpi-delta {
    font-size: 0.8rem;
    margin-top: 4px;
}
.delta-up { color: #34d399; }
.delta-down { color: #f87171; }

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #e0e7ff;
    margin: 2rem 0 1rem 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(129,140,248,0.3);
}

/* Chart containers */
.chart-container {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #a855f7);
    border-radius: 20px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner h1 {
    font-size: 2.5rem;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}
.hero-banner p {
    color: rgba(255,255,255,0.85);
    font-size: 1.1rem;
    font-weight: 400;
}

/* Chat styling */
.chat-msg-user {
    background: linear-gradient(135deg, #4f46e5, #6366f1);
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    color: white;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
}
.chat-msg-ai {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    color: #e0e7ff;
    margin: 8px 0;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_data():
    """Load only processed data (fast). Raw data loaded lazily on demand."""
    base = os.path.dirname(os.path.abspath(__file__))
    proc = os.path.join(base, 'data', 'processed')
    
    data = {}
    
    # Load processed data only
    merged_path = os.path.join(proc, 'merged_analytics_data.csv')
    if os.path.exists(merged_path):
        data['merged'] = pd.read_csv(merged_path, parse_dates=['order_date'])
        data['clean_transactions'] = pd.read_csv(
            os.path.join(proc, 'clean_transactions.csv'), parse_dates=['order_date']
        )
        data['clean_reviews'] = pd.read_csv(
            os.path.join(proc, 'clean_reviews.csv'), parse_dates=['review_date']
        )
    
    # Load report (small JSON)
    report_path = os.path.join(proc, 'cleaning_report.json')
    if os.path.exists(report_path):
        with open(report_path) as f:
            data['report'] = json.load(f)
    
    return data


@st.cache_data(ttl=600)
def load_raw_data():
    """Load raw data only when Pipeline page is visited."""
    base = os.path.dirname(os.path.abspath(__file__))
    raw = os.path.join(base, 'data')
    raw_trans = os.path.join(raw, 'sales_transactions.csv')
    if os.path.exists(raw_trans):
        return {
            'raw_transactions': pd.read_csv(raw_trans),
            'raw_reviews': pd.read_csv(os.path.join(raw, 'customer_reviews.csv')),
        }
    return {}


def render_kpi(label, value, delta=None, delta_dir="up"):
    """Render a KPI card."""
    delta_html = ""
    if delta:
        cls = "delta-up" if delta_dir == "up" else "delta-down"
        arrow = "↑" if delta_dir == "up" else "↓"
        delta_html = f'<div class="kpi-delta {cls}">{arrow} {delta}</div>'
    
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown("## 📊 Analytics System")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🔧 Data Pipeline",
            "💰 Sales Dashboard",
            "👥 Customer Segmentation",
            "📈 Trends & Forecasting",
            "🎯 KPI Monitor",
            "🧠 AI/ML Models",
            "🤖 AI Assistant",
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(
        "<div style='color:#a5b4fc;font-size:0.75rem;text-align:center;'>"
        "Business Analytics System<br>Data Mining & Interactive Dashboards"
        "</div>",
        unsafe_allow_html=True
    )

# ── Load Data ──
data = load_data()

if 'merged' not in data:
    st.warning("⚠️ No processed data found. Please run the data pipeline first:")
    st.code("python generate_datasets.py\npython data_pipeline.py", language="bash")
    st.stop()

# ── Page Router ──
if page == "🏠 Overview":
    from pages import overview
    overview.render(data, render_kpi)
elif page == "🔧 Data Pipeline":
    from pages import pipeline_view
    # Lazy-load raw data only for this page
    raw = load_raw_data()
    data.update(raw)
    pipeline_view.render(data)
elif page == "💰 Sales Dashboard":
    from pages import sales_dashboard
    sales_dashboard.render(data, render_kpi)
elif page == "👥 Customer Segmentation":
    from pages import customer_segmentation
    customer_segmentation.render(data, render_kpi)
elif page == "📈 Trends & Forecasting":
    from pages import trends_forecasting
    trends_forecasting.render(data, render_kpi)
elif page == "🎯 KPI Monitor":
    from pages import kpi_monitor
    kpi_monitor.render(data, render_kpi)
elif page == "🧠 AI/ML Models":
    from pages import ml_models
    ml_models.render(data, render_kpi)
elif page == "🤖 AI Assistant":
    from pages import ai_assistant
    ai_assistant.render(data)
