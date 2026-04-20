import streamlit as st
from src.customer_churn.llm.report_generation import report_generation
from src.streamlit_ui.home import home
from src.streamlit_ui.predict import predict
from src.streamlit_ui.generate_report import generate_report
from src.streamlit_ui.about import about



#################################################
# Session initialization:
#################################################

# For pages:
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Home"

# For LLM report:
if 'llm_report_ready' not in st.session_state:
    st.session_state['llm_report_ready'] = False


#################################################
# Navigation section:
#################################################
# page = st.sidebar.selectbox("Navigation Menu", ["🏠 Home", "📊 Predict", 
#                                                 "📑 Generate Report", "ℹ️ About"], key="navigation_target")
# st.sidebar.markdown("**🔍 Navigate through the sections to explore customer churn insights!**")
# st.sidebar.markdown("")

page_options = ["🏠 Home", "📊 Predict", "📑 Generate Report", "ℹ️ About"]
current_index = page_options.index(st.session_state.page)
selected_page = st.sidebar.selectbox(
    "Navigation Menu",
    page_options,
    index=current_index,
    key="nav_selectbox"  
)

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()


if st.session_state.page == "🏠 Home":
    home()
elif st.session_state.page == "📊 Predict":
    predict()
elif st.session_state.page == "📑 Generate Report":
    generate_report()
elif st.session_state.page == "ℹ️ About":
    about()