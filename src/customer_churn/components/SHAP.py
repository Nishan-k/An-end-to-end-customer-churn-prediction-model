import shap
import joblib
import numpy as np
import pandas as pd
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import logging
import sys



# Path initialization:
pre_processing_artifacts = "Artifacts/pre_processing/pre_processor.pkl"
model_trainer_artifacts = "Artifacts/model_training/final_model.pkl"

class ShapExplainer:
    
    def __init__(self):
        self.model = joblib.load(model_trainer_artifacts)
        self.preprocessor = joblib.load(pre_processing_artifacts)
        self.explainer = shap.TreeExplainer(self.model)

    
    def get_feature_names(self) -> list:
        """
        Get feature names after preprocessing.
        """
        try:
            num_transformer = self.preprocessor.named_transformers_['numerical']
            cat_transformer = self.preprocessor.named_transformers_['categorical']
            num_names = (
                num_transformer.get_feature_names_out().tolist()
                if hasattr(num_transformer, 'get_feature_names_out')
                else self.preprocessor.transformers[0][2]
            )
            cat_names = cat_transformer.get_feature_names_out().tolist()
            return num_names + cat_names
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def _get_feature_value(self, feature, X_raw):
        """
        Get original feature value from raw input.
        """
        try:
            val = X_raw.iloc[0][feature]
            if isinstance(val, (np.integer, int)):
                return int(val)
            elif isinstance(val, float):
                return round(val, 4)
            else:
                return val
        except Exception:
            return None
            

    def _get_shap_row(self, shap_values):
        """
        Handle different SHAP output formats.
        """
        if isinstance(shap_values, list):
            return shap_values[1][0]
        elif len(shap_values.shape) == 2:
            return shap_values[0]
        else:
            return shap_values
            

    def _aggregate_one_hot(self, contributions):
        """
        Aggregate ONLY one-hot encoded categorical features.
        """
        grouped = {}
        cat_transformer = self.preprocessor.named_transformers_['categorical']
        cat_feature_names = cat_transformer.feature_names_in_
    
        for feature, value in contributions:    
            matched = False
            for cat_col in cat_feature_names:
                prefix = f"{cat_col}_"
    
                if feature.startswith(prefix):
                    grouped[cat_col] = grouped.get(cat_col, 0.0) + value
                    matched = True
                    break
            if not matched:
                grouped[feature] = grouped.get(feature, 0.0) + value
    
        return sorted(
            grouped.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
    
    
    def explain(self, X_raw: pd.DataFrame) -> dict:
        """
        Takes raw features and returns SHAP explanations.
        """
        try:
            X_transformed = self.preprocessor.transform(X_raw)
            shap_values = self.explainer.shap_values(X_transformed)
            feature_names = self.get_feature_names()
            shap_row = self._get_shap_row(shap_values)
            raw_contributions = list(zip(feature_names, shap_row))
            raw_sorted = sorted(
                raw_contributions,
                key=lambda x: abs(x[1]),
                reverse=True
            )
            aggregated = self._aggregate_one_hot(raw_contributions)
            return {
                "shap_values": shap_row.tolist(),
                "feature_names": feature_names,
                "raw_contributions": [
                    {"feature": f, "shap_value": round(float(v), 4)}
                    for f, v in raw_sorted
                ],
               "aggregated_contributions": [
                                                {
                                                    "feature": f,
                                                    "value": self._get_feature_value(f, X_raw),
                                                    "shap_value": round(float(v), 4)
                                                }
                                                for f, v in aggregated
                                            ]
            }
        except Exception as e:
            raise CustomerChurnException(e, sys)