import os
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime





# 1. Load the dotenv file:
# load OpenAI key for running locally:
if os.path.exists(".env"):
    load_dotenv(override=True)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in the environment.")

MODEL = 'gpt-4o-mini'
client = OpenAI(api_key=api_key)


def system_prompt(report_type=None, audience=None, include_recommendations=True):
    """
    System prompt for the LLM to instruct the AI on how to behave.
    """
    prompt = f"""
You are a **Customer Retention Analyst** AI. Your task is to generate a concise, actionable report explaining customer churn risk based on SHAP values and model predictions.

**Rules**:
- Positive SHAP values **increase** churn risk; negative SHAP values **decrease** churn risk.
- Focus on the top 3-5 features with the highest absolute SHAP values.
- Express SHAP values as percentages (e.g., 0.05 = 5% impact).
- Base all insights strictly on the provided data.

**Output Structure**:
- **Title**: "Customer Churn Risk Report"
- **Prediction**: State the prediction and probability (e.g., "High Risk – 62% chance of churn").
- **Top Drivers**: List the most impactful features, explaining whether they increase or decrease risk, and by how much.
- **Business Interpretation**: Explain what each driver means in practical business terms.
"""
    if include_recommendations:
        prompt += """
- **Recommendations**: Suggest 2-3 specific, actionable interventions based on the top drivers. Each recommendation should include estimated impact (high/medium/low) and implementation difficulty (easy/moderate/difficult).
"""
    if report_type == "Executive Summary":
        prompt += "\nKeep the report very brief, focusing on business impact and ROI. Avoid technical details."
    elif report_type == "Detailed Analysis":
        prompt += "\nProvide deeper explanations, more context, and supporting details. Include possible underlying causes."
    elif report_type == "Technical Deep Dive":
        prompt += "\nInclude technical details about SHAP values, feature engineering, and model behavior. Use precise data science terminology."
    elif report_type == "Action Plan":
        prompt += "\nFormat the report as a step-by-step action plan with specific actions, responsible roles, and success metrics."

    if audience == "Management":
        prompt += "\nUse business language, highlight financial implications and strategic recommendations."
    elif audience == "Customer Service Team":
        prompt += "\nFocus on customer experience, practical guidance for interactions, and service improvements."
    elif audience == "Technical Team":
        prompt += "\nUse technical language, discuss implementation details and system improvements."
    elif audience == "Marketing Team":
        prompt += "\nFocus on customer segmentation, messaging, and campaign ideas to reduce churn."

    return prompt


def user_prompt(results, customer_id, report_type=None, audience=None, include_recommendations=True):
    """
    User prompt containing the actual data.
    results: dict from API response containing 'prediction_results' and 'shap_values'
    """
    pred = results['prediction_results'][0]
    prediction = pred['prediction']
    prob_churn = pred['prediction_probability']
    risk_label = pred['risk_label']
    
    shap_list = results['shap_values']  # list of dicts: feature, value, shap_value
    
    # Build dictionaries for SHAP values and customer feature values
    shap_dict = {item['feature']: item['shap_value'] for item in shap_list}
    feature_values = {item['feature']: item['value'] for item in shap_list}
    
    # Sort by absolute SHAP value
    sorted_shap = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True))
    
    prompt = f"""
Generate a customer churn report based on the following data:
**Customer ID:** {customer_id}
**Current date**: {datetime.now().strftime('%B %d, %Y')}
**Prediction**: {prediction}
**Churn Probability**: {prob_churn:.2%}
**Risk Label**: {risk_label}

**Feature values (customer data)**: {feature_values}

**SHAP values** (positive = increases churn risk, negative = decreases risk):
{sorted_shap}

**Current date**: {datetime.now().strftime('%B %d, %Y')}
"""
    if report_type:
        prompt += f"\n**Report Type**: {report_type}"
    if audience:
        prompt += f"\n**Target Audience**: {audience}"
    if not include_recommendations:
        prompt += "\n**Do NOT include recommendations** in this report."

    prompt += """
\nInterpretation guidelines:
- Positive SHAP → increases churn risk; negative → decreases.
- Use absolute SHAP values to rank importance.
- Convert SHAP values to percentages (e.g., 0.05 = 5% impact).
- Keep the report clear and actionable based on the data provided.
"""
    return prompt




def get_report(results, customer_id, report_type=None, audience=None, include_recommendations=True):
    """
    Stream the report from OpenAI.
    """
    system = system_prompt(report_type, audience, include_recommendations)
    user = user_prompt(results, customer_id, report_type, audience, include_recommendations)
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    
    report_placeholder = st.empty()
    full_response = ""
    
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            report_placeholder.markdown(full_response)
    
    current_date = datetime.now().strftime('%B %d, %Y')
    header = f"**Customer ID:** {customer_id}\n**Date:** {current_date}\n\n"
    full_response = header + full_response
    report_placeholder.markdown(full_response)
    return full_response