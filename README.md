# 🛒 Customer Churn Prediction System
### End-to-end production ML pipeline with SHAP explanations and AI-generated reports
[![Live Demo](https://img.shields.io/badge/demo-LIVE-brightgreen)](https://e2e-customer-churn-ai.streamlit.app/) 

[![Backend API](https://img.shields.io/badge/backend-Render-blue)](https://an-end-to-end-customer-churn-prediction.onrender.com/health) 
[![Dataset](https://img.shields.io/badge/dataset-UC_Irvine_ML_Repository-blue)](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

> Built on unlabelled retail transaction data: churn labels were engineered 
> from scratch using observation and churn window methodology. 
> Model selected on business cost minimisation (€17,270 at risk) 
> rather than F1 score, precision or recall alone.


## 🖥️ Application Preview
**Churn Prediction with SHAP Explanations**
![App Screenshot](icons/app_demo.png)

**AI-Generated Retention Report**
![Report Page](icons/report_demo.png)
## Table of contents

<ol>
<li><a href="#problemstatement"><b> Problem Statement </b> </a></li>
<li><a href="#businessimpact"><b> Business Impact </b> </a></li>
<li><a href="#dataset"><b> Dataset </b> </a></li>
<li><a href="#keydesigndecisions"><b> Key Design Decisions </b> </a></li>
<li><a href="#experimentandmodelselection"><b> Experiment Tracking & Model Selection </b> </a></li>
<li><a href="#modelperformance"><b> Model Performance </b> </a></li>
<li><a href="#appfeatures"><b> Application Features </b> </a></li>
<li><a href="#techstack"><b> Tech Stack </b> </a></li>
<li><a href="#mlpipepline"><b>ML Pipeline Components </b> </a></li>
<li><a href="#projectstructure"><b> Project Structure </b> </a></li>
<li><a href="#runlocally"><b>Run Locally </b> </a></li>
<li><a href="#futureimprovements"><b>Future Improvements </b> </a></li>
<li><a href="#connect"><b> Connect </b> </a></li>
</ol>


<h2 id='problemstatement'> 1. Problem Statement </h2>
E‑commerce companies lose valuable revenue when customers stop buying. Early identification of <b>churn risk</b> allows targeted retention campaigns, reducing lost customers and marketing waste.

**Challenge:** Raw transaction logs contain no churn label. This project:
- **Engineers a churn label** from scratch (observation window 2009‑2010, churn window Jan–Mar 2011).
- **Builds an end‑to‑end MLOps pipeline** to predict, explain, and act on churn.
- **Goes beyond model metrics**: optimises for real business cost (€5 per wasted offer, €200 per lost customer).



<h2 id='businessimpact'>2. Business Impact</h2>
For the project, a hypothetical cost framework was applied:

- **Retention outreach cost (False Positive):** €10 per customer
- **Customer Lifetime Value lost (False Negative):** €200 per customer

> Without any intervention, all 2,861 actual churners in the <b>test set</b> would be lost, costing <b>€572,200</b>. Using the XGBoost model reduces the total cost to <b>€17,270</b>, a saving of <b>€554,930</b> (≈ 97% reduction).



<h2 id='dataset'>3. Dataset</h2>

- <b>Data Source:</b> <a href="https://archive.ics.uci.edu/dataset/502/online+retail+ii"> UCI Online Retail II</a>
- <b>Instances:</b> 1,067,371
- <b>Number of features:</b> 9
- <b>Unlabelled:</b> No explicit churn column, churn label created via time-based split:
    - <b>Observation window:</b> 2009‑01‑01 to 2010‑12‑31 (features).
    - <b> Churn window:</b> 2011‑01‑01 to 2011‑03‑31 (label = 1 if no purchase).




<h2 id='keydesigndecisions'>4. Key Design Decisions</h2>

1.






