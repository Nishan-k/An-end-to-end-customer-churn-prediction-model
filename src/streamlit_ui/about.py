import streamlit as st

def about():
    st.title("📌 About This Project")
    st.markdown("---")

    with st.expander("📊 Dataset & Churn Definition", expanded=True):
        st.markdown("""**Dataset**: UCI Online Retail II (1,067,371 transactions, 2009–2011).""")
        st.page_link("https://archive.ics.uci.edu/dataset/502/online+retail+ii", label="UCI ML Repository", icon="🌐")
        st.markdown("""
        **Churn definition**:  
        - Observation window: 2009‑01‑01 to 2010‑12‑31 (features).  
        - Churn window: 2011‑01‑01 to 2011‑03‑31 (label = 1 if no purchase).  
        """)

    with st.expander("🏗️ Architecture Overview"):
        st.image("architecture/churn_architecture.png", caption="End‑to‑end ML pipeline", use_container_width=True)
        st.markdown("""
        - **Data Ingestion** → MongoDB → Parquet feature store.  
        - **Data Validation** → schema check (columns, dtypes).  
        - **Data Cleaning** → duplicates, impute missing price, flag returns/cancellations, outlier capping (two branches).  
        - **Feature Engineering** → RFM, product‑level risk scores, customer‑level aggregates.  
        - **Model Training** → XGBoost with hyperparameter tuning (MLflow).  
        - **API** → FastAPI serves predictions + SHAP explanations.  
        - **UI** → Streamlit for manual input, SHAP plots, LLM reports.  
        - **Deployment** → Streamlit (UI), Render(Backend), and Corn-Job(Pings /health every 5 minutes) to stop Render from freezing.
        """)

    with st.expander("🧪 Experiments & Model Selection"):
        st.markdown("""
        - **Spot-Checking Algorithms(Total 9 algorithms)**: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier
                     GradientBoostingClassifier, XGBClassifier, KNeighborsClassifier, LinearDiscriminantAnalysis, and SVC.
        - **Mlflow Experiment tracked(Total 5 Alogrithms) **: GradientBoostingClassifier, SVC, RandomForestClassifier, ExtraTreesClassifier, and
                    XGBClassifier.
        - **Two data branches**: clipped (outliers capped at 99th percentile) vs. unclipped.  
        - **Winner**: XGBClassifier on **unclipped** data.  
        - **Key metrics** (test set, 4,141 customers):  
          - F1 = 84.81%  
          - Recall (churn) = 98.81%  
          - Precision (churn) = 72.96%  
        - **Business‑driven selection**: Used cost of false positives (€5) vs. false negatives (€200). XGBoost unclipped minimised a total cost of €17,270.
        """)

    with st.expander("🛠️ Tech Stack"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data & ML**\n- pandas / NumPy\n- scikit‑learn\n- XGBoost\n- SHAP\n- MLflow\n- Plotly")
        with col2:
            st.markdown("**DevOps & Deployment**\n- FastAPI / Uvicorn\n- Streamlit\n- Render\n- Cron-Job \n- Git / DagsHub")

    with st.expander("🚀 Future Improvements"):
        st.markdown("""
        - Batch prediction (CSV upload with automatic feature engineering).  
        - CI/CD pipeline (GitHub Actions).  
        - Automated retraining with Apache Airflow.  
        - Model performance monitoring on Grafana.
        - Deployment on AWS.
        """)

    st.markdown("---")
    st.markdown("**👤 Author** – Nishan Karki  \n🔗 [GitHub Repository](https://github.com/Nishan-k/An-end-to-end-customer-churn-prediction-model)")
