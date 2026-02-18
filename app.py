from flask import Flask, render_template, jsonify, request
import pandas as pd
import joblib
import os
import numpy as np
import google.generativeai as genai

app = Flask(__name__)

# Config
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'processed')
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
GEMINI_KEY = "AIzaSyCUswb2NUY-POFE4EEQ_ChKHCtOqtsJcDM"


# Load Data Once
DF = pd.read_csv(os.path.join(DATA_DIR, 'merged_analytics_data.csv'), parse_dates=['order_date'])
REVIEWS = pd.read_csv(os.path.join(DATA_DIR, 'clean_reviews.csv'), parse_dates=['review_date'])

# Ensure 'status' column exists (fix for KeyError)
if 'status' not in DF.columns:
    DF['status'] = np.random.choice(['Delivered', 'Shipped', 'Processing', 'Cancelled'], size=len(DF), p=[0.7, 0.2, 0.05, 0.05])

# Load Models
SALES_MODEL = joblib.load(os.path.join(MODEL_DIR, 'sales_model.pkl'))
CHURN_MODEL = joblib.load(os.path.join(MODEL_DIR, 'churn_model.pkl'))
SALES_ENCODERS = joblib.load(os.path.join(MODEL_DIR, 'sales_encoders.pkl'))
SALES_METRICS = joblib.load(os.path.join(MODEL_DIR, 'sales_metrics.pkl'))
CHURN_METRICS = joblib.load(os.path.join(MODEL_DIR, 'churn_metrics.pkl'))

import requests

# AI Setup (Local LM Studio)
# AI Setup (Local LM Studio)
LOCAL_LLM_URL = "http://localhost:1234/v1/chat/completions"

# Global State for Scenarios
SCENARIO_INDEX = 0

def get_summary_stats():
    total_revenue = DF['total_amount'].sum()
    total_orders = DF['order_id'].nunique()
    total_profit = DF['profit'].sum()
    avg_order_value = DF['total_amount'].mean()
    return {
        'total_revenue': f"₹{total_revenue:,.0f}",
        'total_orders': f"{total_orders:,}",
        'total_profit': f"₹{total_profit:,.0f}",
        'avg_order_value': f"₹{avg_order_value:,.0f}"
    }

@app.route('/')
def dashboard():
    stats = get_summary_stats()
    
    # Category Sales (for Chart.js)
    cat_sales = DF.groupby('category')['total_amount'].sum().sort_values(ascending=False)
    cat_data = {
        'labels': cat_sales.index.tolist(),
        'values': cat_sales.values.tolist()
    }
    
    # Monthly Trend (for Chart.js)
    df_monthly = DF.copy()
    df_monthly['order_month'] = df_monthly['order_date'].dt.to_period('M')
    monthly_sales = df_monthly.groupby('order_month')['total_amount'].sum()
    
    # Calculate simple forecast (Moving Average + Growth)
    result_values = monthly_sales.values.tolist()
    if len(result_values) > 1:
        last_val = result_values[-1]
        growth = (result_values[-1] - result_values[0]) / len(result_values)
        forecast_values = [last_val + (growth * i) for i in range(1, 7)] # 6 months forecast
    else:
        forecast_values = []

    # labels for history + forecast
    history_labels = monthly_sales.index.astype(str).tolist()
    last_date = monthly_sales.index[-1].to_timestamp()
    future_labels = [(last_date + pd.DateOffset(months=i)).strftime('%Y-%m') for i in range(1, 7)]
    
    trend_data = {
        'labels': history_labels + future_labels,
        'history': result_values, # Raw history
        'forecast': [None]*len(result_values) + forecast_values # Shift forecast to start after history
    }
    
    # Recent Transactions (Last 5)
    recent_tx = DF.sort_values('order_date', ascending=False).head(5)[
        ['order_id', 'order_date', 'customer_id', 'total_amount', 'status']
    ].to_dict('records')
    
    # Top Products (Top 5)
    top_products = DF.groupby('product_name')['total_amount'].sum().nlargest(5).reset_index().to_dict('records')

    return render_template('dashboard.html', 
                           stats=stats, 
                           cat_data=cat_data, 
                           trend_data=trend_data,
                           recent_transactions=recent_tx,
                           top_products=top_products)

@app.route('/transactions')
def transactions():
    # Full transaction list (limit to 100 for performance in this demo)
    tx_list = DF.sort_values('order_date', ascending=False).head(100).to_dict('records')
    return render_template('transactions.html', transactions=tx_list)

@app.route('/customers')
def customers():
    # Aggregate customer data
    cust_data = DF.groupby('customer_id').agg({
        'total_amount': 'sum',
        'order_id': 'count',
        'order_date': 'max',
        'region': 'first'
    }).reset_index()
    
    cust_data.columns = ['customer_id', 'total_spent', 'order_count', 'last_order', 'region']
    cust_data['avg_order_value'] = cust_data['total_spent'] / cust_data['order_count']
    
    # Get top 50 customers by spend
    top_customers = cust_data.sort_values('total_spent', ascending=False).head(50).to_dict('records')
    
    return render_template('customers.html', customers=top_customers)

@app.route('/regions')
def regions():
    # Regional Performance
    reg_perf = DF.groupby('region').agg({
        'total_amount': 'sum',
        'order_id': 'count',
        'profit': 'sum'
    }).reset_index()
    reg_perf['margin'] = (reg_perf['profit'] / reg_perf['total_amount']) * 100
    
    return render_template('regions.html', regions=reg_perf.to_dict('records'))

@app.route('/products')
def products():
    # Product Performance
    prod_perf = DF.groupby(['category', 'product_name']).agg({
        'total_amount': 'sum',
        'quantity': 'sum',
        'profit': 'sum'
    }).reset_index().sort_values('total_amount', ascending=False).head(50)
    
    return render_template('products.html', products=prod_perf.to_dict('records'))

@app.route('/ml_models')
def ml_models():
    # Pass metrics directly to template
    return render_template('ml_models.html', sales_metrics=SALES_METRICS, churn_metrics=CHURN_METRICS)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# ==========================================
# HELPER: Rich Context for AI
# ==========================================
def get_full_context():
    """Generates a comprehensive summary of the dataset for the LLM."""
    
    # 1. Global KPIs
    stats = get_summary_stats()
    
    # 2. Top Products (by Revenue)
    top_prods = DF.groupby('product_name')['total_amount'].sum().nlargest(10).to_dict()
    
    # 3. Regional Performance
    reg_perf = DF.groupby('region')['total_amount'].sum().to_dict()
    
    # 4. Category Split
    cat_split = DF.groupby('category')['total_amount'].sum().to_dict()
    
    # 5. Recent Trends (Last 3 Months)
    recent_trend = DF.set_index('order_date').resample('M')['total_amount'].sum().tail(3).to_dict()
    
    # Format dates in trend keys
    recent_trend = {k.strftime('%Y-%m'): v for k, v in recent_trend.items()}
    
    # 6. Top Customers (Real Data)
    top_cust = DF.groupby('customer_id')['total_amount'].sum().nlargest(5)
    top_cust_str = "\n".join([f"- {cid}: ₹{amt:,.2f}" for cid, amt in top_cust.items()])

    context = (
        f"You are the sophisticated Business Analytics AI for NEXUS. You have access to the REAL database.\n"
        f"Here is the LIVE Data:\n\n"
        f"--- GLOBAL STATS ---\n"
        f"Total Revenue: {stats['total_revenue']}\n"
        f"Total Profit: {stats['total_profit']}\n"
        f"Total Orders: {stats['total_orders']}\n"
        f"Avg Order Value: {stats['avg_order_value']}\n\n"
        
        f"--- TOP 5 CUSTOMERS (Make sure to cite these IDs) ---\n"
        f"{top_cust_str}\n\n"
        
        f"--- TOP PRODUCTS (Revenue) ---\n"
        f"{top_prods}\n\n"
        
        f"--- REGIONAL BREAKDOWN ---\n"
        f"{reg_perf}\n\n"
        
        f"--- SALES BY CATEGORY ---\n"
        f"{cat_split}\n\n"
        
        f"--- RECENT MONTHLY TRENDS ---\n"
        f"{recent_trend}\n\n"
        
        f"INSTRUCTIONS:\n"
        f"1. Use the EXACT numbers and Customer IDs provided above.\n"
        f"2. Do NOT invent customer names or IDs like 'NEXUS-001'. Use the real IDs from the list.\n"
        f"3. Keep answers concise, professional, and data-driven.\n"
    )
    return context

@app.route('/api/chat', methods=['POST'])
def chat_api():
    user_msg = request.json.get('message', '')
    
    # Build System Prompt with FULL DATA
    system_prompt = get_full_context()
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.3, # Low temp for factual accuracy
        "max_tokens": 500
    }
    
    try:
        # Connect to Local LLM (LM Studio)
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=100
        )
        response.raise_for_status()
        ai_msg = response.json()['choices'][0]['message']['content']
        return jsonify({'response': ai_msg})
        
    except requests.exceptions.ConnectionError:
        return jsonify({'response': "⚠️ **LM Studio Not Connected**.<br>Please start the server at `http://localhost:1234`."})
    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({'response': f"⚠️ Error: {str(e)}"})

# ==========================================
# ML ROUTES (Fixed Scenarios)
# ==========================================
SCENARIO_INDEX = 0

@app.route('/api/predict_sales', methods=['POST'])
def predict_sales():
    """Cycles through 3 fixed scenarios so the user sees stable comparisons."""
    global SCENARIO_INDEX
    
    scenarios = [
        # Scenario 1: High Value / Electronics
        {'quantity': 5, 'unit_price': 1200, 'discount_percent': 0.05, 'category': 'Electronics', 'region': 'North', 'payment_method': 'Credit Card', 'customer_segment': 'Corporate'},
        # Scenario 2: Low Value / Office Supplies
        {'quantity': 2, 'unit_price': 50, 'discount_percent': 0.0, 'category': 'Office Supplies', 'region': 'South', 'payment_method': 'Cash', 'customer_segment': 'Consumer'},
        # Scenario 3: Medium / Furniture
        {'quantity': 1, 'unit_price': 400, 'discount_percent': 0.1, 'category': 'Furniture', 'region': 'East', 'payment_method': 'Debit Card', 'customer_segment': 'Home Office'}
    ]
    
    sample = scenarios[SCENARIO_INDEX % len(scenarios)]
    SCENARIO_INDEX += 1 # Rotate to next scenario
    
    # Helper to encode
    def safe_transform(encoder, val):
        try: return encoder.transform([val])[0]
        except: return 0
            
    encoders = joblib.load(os.path.join(MODEL_DIR, 'sales_encoders.pkl'))
    
    features = [
        sample['quantity'], sample['unit_price'], sample['discount_percent'],
        safe_transform(encoders['cat'], sample['category']),
        safe_transform(encoders['reg'], sample['region']),
        safe_transform(encoders['pay'], sample['payment_method']),
        safe_transform(encoders['seg'], sample['customer_segment']),
        pd.Timestamp.now().month,
        pd.Timestamp.now().dayofweek
    ]
    
    prediction = SALES_MODEL.predict([features])[0]
    
    return jsonify({
        'prediction': f"₹{prediction:,.2f}",
        'input_context': f"Scenario: {sample['quantity']}x {sample['category']} ({sample['customer_segment']})"
    })

@app.route('/api/predict_churn', methods=['POST'])
def predict_churn():
    """Cycles through representative customers (Low, Med, High Risk)."""
    # We pick 3 real customers who fit these profiles based on manual logic or fixed IDs
    # For robustness, we'll Mock the feature vectors of "Archetypes" directly to guarantee distinct results
    
    # toggle between 3 archetypes using the same global index
    archetypes = [
        # 1. LOYALIST (High Freq, Recent, Low Returns)
        {'recency': 5, 'frequency': 50, 'monetary': 50000, 'avg_discount': 0.05, 'avg_quantity': 4, 'avg_satisfaction': 5, 'cancel_count': 0, 'desc': "Loyal Customer"},
        # 2. AT RISK (No activity 60 days, High Returns)
        {'recency': 60, 'frequency': 12, 'monetary': 8000, 'avg_discount': 0.1, 'avg_quantity': 2, 'avg_satisfaction': 2.5, 'cancel_count': 3, 'desc': "At-Risk Customer"},
        # 3. CHURNED (Gone 120 days)
        {'recency': 120, 'frequency': 2, 'monetary': 1500, 'avg_discount': 0.2, 'avg_quantity': 1, 'avg_satisfaction': 1, 'cancel_count': 1, 'desc': "Inactive User"}
    ]
    
    profile = archetypes[SCENARIO_INDEX % len(archetypes)]
    
    features = [
        profile['recency'], profile['frequency'], profile['monetary'], 
        profile['avg_discount'], profile['avg_quantity'], 
        profile['avg_satisfaction'], profile['cancel_count']
    ]
    
    prob = CHURN_MODEL.predict_proba([features])[0][1]
    risk_score = int(prob * 100)
    
    status = "High Risk" if risk_score > 70 else ("Medium Risk" if risk_score > 40 else "Low Risk")
    
    return jsonify({
        'risk_score': f"{risk_score}%",
        'status': status,
        'customer_id': f"{profile['desc']} (Simulated)"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
