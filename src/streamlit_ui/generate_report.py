from src.customer_churn.llm.report_generation import report_generation
import streamlit as st 






def generate_report():
    if st.session_state.get('llm_report_ready', False):
       # Retrieve the prediction results from the session 'prediction_results':
       results = st.session_state.get('prediction_results')
       if results:
           report_generation()
       else:
           st.warning("No prediction data found. Please make a prediction first.")
           if st.button("Predict", type="primary"):
            st.session_state.page = "📊 Predict"
            st.rerun()
    else:
       st.info("Please go to the **Predict** page and run a prediction first.")
       if st.button("Predict", type="primary"):
            st.session_state.page = "📊 Predict"
            st.rerun()
       