import pandas as pd
import joblib
import numpy as np
from src.customer_churn.logging.logger import logging
from src.customer_churn.exception.exception import CustomerChurnException
import sys


# Path initialization:
pre_processing_artifacts = "Artifacts/pre_processing/pre_processor.pkl"
model_trainer_artifacts = "Artifacts/model_training/final_model.pkl"


class Prediction:
    
    def __init__(self):
        try:
            self.pre_processor = joblib.load(pre_processing_artifacts)
            self.model = joblib.load(model_trainer_artifacts)
            logging.info("Prediction class initialization success!")
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def predict(self, X_features) -> dict:
        """
        Takes in the features and predicts on it.

        Returns:
        ---------
        results: a dictionary consisiting of the predicted result, predictied probability, and risk label
        """
        try:
            results = []
            logging.info(f"Provided features for prediction:\n{X_features.to_string()}")
            X_features_transformed = self.pre_processor.transform(X_features)
            logging.info("Features transformation success!")
            for prob_array, prediction in zip(self.model.predict_proba(X_features_transformed), self.model.predict(X_features_transformed)):
                churn_probability = round(float(prob_array[1]), 4)
                retention_probability = round(float(prob_array[0]), 4)
                if prediction == 1:
                    if churn_probability < 0.65:
                        risk_label = 'Low Risk'
                    elif churn_probability < 0.80:
                        risk_label = 'Medium Risk'
                    else:
                        risk_label = 'High Risk'
                else:
                    risk_label = 'No Risk'
                results.append({
                    'prediction_probability': churn_probability,
                    'retention_probability': retention_probability,
                    'prediction': 'Churn'if prediction == 1 else 'Stay',
                    'risk_label': risk_label
                })
            logging.info(f"The result of the prediction:\n{results}")
            return results
        except Exception as e:
            raise CustomerChurnException(e, sys)
        