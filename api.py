from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sys
from src.customer_churn.logging.logger import logging
from src.customer_churn.components.prediction_pipeline import Prediction
from src.customer_churn.components.SHAP import ShapExplainer




# Initialize Prediction class:
prediction = Prediction()
shap_explainer = ShapExplainer()

app = FastAPI(title="Customer Churn Prediction API")



# Input Schema:
class CustomerFeatures(BaseModel):
    order_frequency: int
    total_monetary_value: float
    total_quantity_abs: int
    total_order_issues: int
    country: str
    avg_order_value: float
    avg_quantity_per_order: float
    customer_order_issue_rate: float
    recency: int
    tenure: int
    avg_stockcode_issue_rate: float
    max_stockcode_issue_rate: float
    total_orders_made_for_stock: float


# Route for Home:
@app.get("/")
def root():
    return {"status": "running", "model": "XGBoost"}


# Route to predict:
@app.post("/predict_single_feature")
def predict_single_feature(input_feature: CustomerFeatures):
    try:
        # Convert the input feature to Dataframe:
        input_feature_df = pd.DataFrame([input_feature.model_dump()])

        # Call the predict method of the prediction pipeline class:
        prediction_results = prediction.predict(X_features=input_feature_df)

        # Call the explain method of ShapExplainer class to get the Shapely Additive Explanation values:
        shap_values = shap_explainer.explain(X_raw=input_feature_df)['aggregated_contributions']
        return {
            'prediction_results': prediction_results,
            'shap_values': shap_values
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Health endpoint:
@app.get("/health")
async def health_check():
    return {"status":"active"}