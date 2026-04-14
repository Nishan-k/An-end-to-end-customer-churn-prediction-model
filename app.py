import streamlit as st
import pandas as pd
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.components.shap_visualization import ShapVisualization
from src.customer_churn.llm.report_generation import report_generation
import sys
import requests


#################################################
# Session initialization for LLM report:
#################################################
if 'llm_report_ready' not in st.session_state:
    st.session_state['llm_report_ready'] = False


#################################################
# API endpoint:
#################################################
API_URL = "http://127.0.0.1:8000/"


#################################################
# Navigation section:
#################################################
page = st.sidebar.selectbox("Navigation Menu", ["🏠 Home", "📊 Predict", 
                                                "📑 Generate Report", "ℹ️ About"], key="navigation_target")
st.sidebar.markdown("**🔍 Navigate through the sections to explore customer churn insights!**")
st.sidebar.markdown("")




# 1. Home Page:
if page == "🏠 Home":
    st.title("🛒 Customer Churn Prediction for E‑commerce")
    st.markdown("""
    ### Predict, Explain, and Act – End‑to‑End ML Pipeline

    Welcome to the **Customer Churn Prediction System**.  
    This application helps you:
    - ✅ **Predict** whether a customer will churn based on 13 key features.
    - 📊 **Explain** predictions using SHAP values (interactive charts).
    - 📄 **Generate** AI‑powered business reports (OpenAI GPT‑4o‑mini).
    - 💡 **Recommend** retention actions tailored to each customer.

    **How it works:**  
    1. Go to the **Predict** page → enter customer features (or upload a file – coming soon).  
    2. Get instant churn probability and risk label.  
    3. Explore SHAP contributions and download an AI‑generated PDF report.  

    🚀 Built with **XGBoost**, **FastAPI**, **Streamlit**, **SHAP**, and **OpenAI**.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model F1 Score", "84.9%", "on test set")
    with col2:
        st.metric("Recall (Churn)", "98.8%", "catches almost all churners")
    with col3:
        st.metric("Cost Savings", "~96%", "vs no model")
    
    st.markdown("---")
    st.markdown("**👉 Ready to start?** Go to the **📊 Predict** page in the sidebar.")



# 2. Prediction Page:
if page == "📊 Predict":
    st.title("Customer Churn Prediction")
    st.markdown("Enter customer details below to predict churn risk:")
    st.markdown("----")
    manual_input_tab1, batch_tab2 = st.tabs(["Manual Input-DEMO VERSION", "Upload Transactions"])

    with manual_input_tab1:
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
                response = requests.post(f"{API_URL}/predict_single_feature", json=payload)
                results = response.json()
                # Update the session state for LLM report:
                st.session_state['llm_report_ready'] = True
                # Store the session state:
                st.session_state['prediction_results'] = results
                # Session storage for customerid:
                st.session_state['customer_id'] = customer_id

            if 'prediction_results' in st.session_state:
                results = st.session_state['prediction_results']
                
                with st.expander("Raw API Response (For Debugging)"):
                    st.json(results)

                # Display prediction result:
                st.subheader(f"Prediction For Customer: {customer_id}")
                pred = results['prediction_results'][0]
                prob_churn = pred['prediction_probability']
                prob_retain = pred['retention_probability']

                col1, col2 = st.columns(2)
                col1.metric("Churn Probability", f"{prob_churn*100:.2f}%")
                col2.metric("Retention Probability", f"{prob_retain*100:.2f}%")

                st.progress(prob_churn, text=f"Churn risk level: {prob_churn*100:.2f}%")

                if pred['prediction'] == "Stay":
                    st.success("🟢 No Risk — Customer likely to stay")
                else:
                    st.error(f"🔴 {pred['risk_label']} — Customer likely to churn")

                # SHAP visualization and Dataframe:
                shab_tab, shap_df_tab = st.tabs(["SHAP Visualization", "SHAP DataFrame"])
                with shab_tab:
                    # `ShapVisualization` class initialization:
                    shap_visualization = ShapVisualization(results=results)          
                    fig, net_shap, shap_df = shap_visualization.plot_shap_feature_importance()
                    st.plotly_chart(fig)
                    if net_shap < 0:
                        st.info(f"📉 Net SHAP: {net_shap:.3f} → Retention signals dominate")
                    else:
                        st.warning(f"📈 Net SHAP: {net_shap:.3f} → Churn signals dominate")
                with shap_df_tab:
                    display_df = shap_df.drop('Color', axis=1).copy()
                    display_df['Input_features'] = display_df['Input_features'].astype(str)
                    st.dataframe(display_df, use_container_width=True, height=400)
                    st.caption("Negative SHAP = Decreases Churn Risk ↓, Positive SHAP = Increases churn risk ↑")
                    csv = display_df.to_csv(index=False)
                    st.download_button("Download SHAP data (CSV)", csv, "shap_values.csv", "text/csv")

                # To reset the session:        
                if st.button("Reset Session", type="primary"):
                    if 'prediction_results' in st.session_state:
                        del st.session_state['prediction_results']
                    st.rerun()
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    with batch_tab2:
        st.info("🚧 **Batch prediction via CSV upload is under development.** In the future, " \
        "you will be able to upload raw transaction files (with original features like invoice, stock code, etc.) "
        "and the system will automatically generate all engineered features before predicting. " \
        "For now, please use the **Manual Input** tab to test the model.")



        


# 3. LLM Report Generation:
if page == "📑 Generate Report":
   if st.session_state.get('llm_report_ready', False):
       # Retrieve the prediction results from the session 'prediction_results':
       results = st.session_state.get('prediction_results')
       if results:
           report_generation()
       else:
           st.warning("No prediction data found. Please make a prediction first.")
   else:
       st.info("Please go to the **Predict** page and run a prediction first.")

# 4. About Page:
if page == "ℹ️ About":
    st.write("About Page")