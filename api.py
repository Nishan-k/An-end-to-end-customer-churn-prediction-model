from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sys
from src.customer_churn.logging.logger import logging
from src.customer_churn.exception.exception import CustomerChurnException


app = FastAPI(title="Customer Churn Prediction API")

# Load the necessary artifacts:
preprocessor = joblib.load("Artifacts/pre_processing/pre_processor.pkl")
model = joblib.load("Artifacts/model_training/final_model.pkl")

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
@app.post("/predict")
def predict(input_feature: CustomerFeatures):
    try:
        # Convert the input feature to Dataframe:
        input_feature_df = pd.DataFrame([input_feature.model_dump()])

        input_feature_transformed = preprocessor.transform(input_feature_df)
        prediction = model.predict(input_feature_transformed)
        pred = int(prediction[0]) if isinstance(prediction, np.ndarray) else int(prediction)
        return {
            'prediction': pred
        }

    except Exception as e:
        raise CustomerChurnException(e, sys)