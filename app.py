import streamlit as st
import pandas as pd
from src.customer_churn.exception.exception import CustomerChurnException
import sys
import requests


API_URL = "http://127.0.0.1:8000/"

# Navigation section:
page = st.sidebar.selectbox("Navigation Menu", ["🏠 Home", "📊 Predict", 
                                                "📖 Explain", "📑 Generate Report", "ℹ️ About"], key="navigation_target")
st.sidebar.markdown("**🔍 Navigate through the sections to explore customer churn insights!**")
st.sidebar.markdown("")


# 1. Home Page:
if page == "🏠 Home":
    st.write("Home Page")



# 2. Prediction Page:
if page == "📊 Predict":
    st.title("Customer Churn Prediction")
    st.markdown("Enter customer details below to predict churn risk:")
    st.markdown("----")
    tab1, tab2 = st.tabs(["Manual Input-DEMO VERSION", "Upload Transactions"])

    with tab1:
        customer_id = st.number_input(label="Customer ID", min_value=1, value=120)
        col1, col2 = st.columns(2)
        with col1:
            order_frequency = st.number_input(label="Order Frequency", min_value=1, value=25)
            total_monetary_value = st.number_input(label="Total Monetary Value", value=-850.0)
            total_quantity_abs = st.number_input(label="Total Quantity (abs)", min_value=1, value=10)
            total_order_issues = st.number_input(label="Total Order Issue", min_value=1, value=8)
            country = st.text_input(label="Country", value="United Kingdom")
            avg_order_value = st.number_input(label="Average Order Value", min_value=-850.0)

        with col2:
            avg_quantity_per_order = st.number_input(label="Average Quantity Per Order", min_value=1.0, value=250.0)
            customer_order_issue_rate = st.number_input(label="Customer Order Issue Rate", min_value=0.0, value=13.0)
            recency = st.number_input(label="Recency", min_value=1, value=25)
            tenure = st.number_input(label="Tenure", min_value=1, value=97)
            avg_stockcode_issue_rate = st.number_input(label="Average Stockcode Issue Rate", min_value=0.0, value=1.0)
            max_stockcode_issue_rate = st.number_input(label="Maximum Stockcode Issue Rate", min_value=0.0, value=1.02)
            total_orders_made_for_stock = st.number_input(label="Total Orders Made For Stock", min_value=1.0, value=195.0)
        try:
            payload = {
                'order_frequency': order_frequency,
                'total_monetary_value': total_monetary_value,
                'total_quantity_abs': total_quantity_abs,
                'total_order_issues': total_order_issues,
                'country': country,
                'avg_order_value': avg_order_value,
                'avg_quantity_per_order': avg_quantity_per_order,
                'customer_order_issue_rate': customer_order_issue_rate,
                'recency': recency,
                'tenure': tenure,
                'avg_stockcode_issue_rate' : avg_stockcode_issue_rate,
                'max_stockcode_issue_rate': max_stockcode_issue_rate,
                'total_orders_made_for_stock': total_orders_made_for_stock
            }

            if st.button("Predict", type="primary"):
                st.write(f"**Customer ID:** `{customer_id}`")
                st.write("**Result:**")
                response = requests.post(f"{API_URL}/predict", json=payload)
                st.write(f"Status code: {response.status_code}")
                st.write(f"Response text: {response.text}")
                result = response.json()
                st.write(result)
                
        except Exception as e:
            raise CustomerChurnException(e, sys)


        

# 3. Explain Page:
if page == "📖 Explain":
     st.write("Explain Page")

# 4. LLM Report Generation:
if page == "📑 Generate Report":
    st.write("LLM Report Generation Page")

# 5. About Page:
if page == "ℹ️ About":
    st.write("About Page")