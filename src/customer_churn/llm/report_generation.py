import streamlit as st  
from datetime import datetime
import os 
import time
from src.customer_churn.llm.llm_call import get_report
from src.customer_churn.llm.pdf_generator import save_report_as_pdf




def report_generation():
    """
    Responsible for generating the report using OpenAI.
    """
    st.header("🧠 AI-Powered Churn Analysis Report")
    st.markdown("""
    Generate a comprehensive analysis of customer churn prediction with actionable insights 
    for stakeholders. The AI will analyze the prediction factors and provide tailored recommendations.
    """)

    results = st.session_state['prediction_results']
    customer_id = st.session_state.get('customer_id', 'Unknown')
    pred = results['prediction_results'][0]
    prediction = pred['prediction']         
    prob_churn = pred['prediction_probability']
    prob_retain = pred['retention_probability']

    st.subheader("Prediction Result")
    if prediction == "Churn":
        st.error(f"⚠️ Customer Likely to Churn")
        st.markdown(f"**Churn Probability:** {prob_churn:.2%}")
    else:
        st.success(f"✅ Customer Likely to Stay")
        st.markdown(f"**Retention Probability:** {prob_retain:.2%}")

    st.markdown("---")
    st.subheader("Customize Your Report")

    col1, col2 = st.columns(2)
    with col1:
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Detailed Analysis", "Technical Deep Dive", "Action Plan"]
        )
    with col2:
        audience = st.selectbox(
            "Target Audience",
            ["Management", "Customer Service Team", "Technical Team", "Marketing Team"]
        )

    include_recommendations = st.checkbox("Include Actionable Recommendations", value=True)

    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            # Call the updated get_report function that takes the full results dict
            response = get_report(
                results=results,
                customer_id=customer_id,
                report_type=report_type,
                audience=audience,
                include_recommendations=include_recommendations
            )
            st.session_state.report_content = response
        
        current_date = datetime.now().strftime("%Y%m%d")
        pdf_filename = f"{customer_id}_{current_date}.pdf"
        pdf_path = save_report_as_pdf(report_text=response, pdf_filename=pdf_filename)

        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as file:
                st.download_button(
                    label="📥 Download as PDF",
                    data=file,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    key="download_pdf"
                )
        else:
            st.error("Failed to generate PDF.")

       