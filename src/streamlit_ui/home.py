import streamlit as st 



def home():
    st.title("🛒 Customer Churn Prediction for E‑commerce")
    st.markdown("""
    ### Predict, Explain, and Act: An End‑to‑End ML Pipeline with AI integration

    Welcome to the **Customer Churn Prediction System**.  
                
    This application helps you:
    - **Predict** whether a customer will churn based on 13 key features.
    - **Explain** predictions using SHAP values (interactive charts).
    - **Generate** AI‑powered business reports (OpenAI GPT‑4o‑mini).
    - **Recommend** retention actions tailored to each customer.

    **How it works:**  
    1. Go to the **Predict** page → enter customer features (or upload a file – coming soon).  
    2. Get instant churn probability and risk label.  
    3. Explore **SHAP contributions** and download an **AI‑generated PDF report**.  

    🚀 Built with **XGBoost**, **FastAPI**, **Streamlit**, **SHAP**, **MlFlow**, and **OpenAI**.
    """)
    st.divider()
    st.subheader("🎯 Model Performance at a Glance")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CV F1(XGBoost)", "84.81%", "on test set")
    with col2:
        st.metric("Recall (Churn)", "98.81%", "on test set")
    with col3:
        st.metric("Precision (Churn)", "72.96%", "on test set")

    with st.container(border=True):
        st.subheader("🧪 Experiment Tracking")
        st.write("All model training runs, hyperparameters, and metrics are tracked with MLflow.")
        st.link_button("📈 View Experiments on DagsHub",
                       "https://dagshub.com/Nishan-k/An-end-to-end-customer-churn-prediction-model/experiments#/experiment/m_e3821cd47cee41b9a2a46b5e69bcc151",
                        help="Open MLflow experiments in DagsHub",
                        use_container_width=True)
    st.divider()
    if st.button("🚀 Get Started", type="primary"):
        st.session_state.page = "📊 Predict"
        st.rerun()
