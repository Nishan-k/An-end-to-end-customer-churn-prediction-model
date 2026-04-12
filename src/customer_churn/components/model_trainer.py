import os
import pandas as pd
import numpy as np
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import logging
from xgboost import XGBClassifier
import joblib
from src.customer_churn.utils.main_utils.common import read_yaml_file
from src.customer_churn.entity.artifact_entity import ModelTrainerArtifacts, PreProcessingArtifacts
from src.customer_churn.config.configuration import ModelTrainingConfig
import time
import sys


class ModelTraining:

    def __init__(self, model_training_config: ModelTrainingConfig,                 
                 pre_processing_artifacts: PreProcessingArtifacts):
        try:
            self.model_training_config = model_training_config
            self.model_file_path = self.model_training_config.model_training_file_model_name
            self.pre_processing_artifacts = pre_processing_artifacts
            self.pre_processor_file_path = self.pre_processing_artifacts.pre_processor_file_path
            self.X_path = self.pre_processing_artifacts.X_path
            self.y_path = self.pre_processing_artifacts.y_path          
            self.best_params = read_yaml_file("params.yaml")['model_trainer']['best_params']
            logging.info(f"ModelTraining class initialization completed!")
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def load_data(self):
        """
        Loads the parquet file X and y.

        Returns:
        ---------
        X: pd.DataFrame
        y: pd.series
        """
        try:
            X = pd.read_parquet(self.X_path)
            y = pd.read_parquet(self.y_path).squeeze()
            logging.info(f"Data loading success! | X:{X.shape} | y: {y.shape}")
            return X, y
        except Exception as e:
            raise CustomerChurnException(e, sys)

    
    def load_pre_processor(self):
        """
        Loads the pre-processor(.pkl) file

        Returns:
        --------
        pre_processor: the finalized pre-processor (made up of RobustScaler() and OneHot Encoder(), an artifact of preprocessor.py)
        """
        try:
            pre_processor = joblib.load(self.pre_processor_file_path)
            logging.info("Pre-processor loading success!")
            return pre_processor
        except Exception as e:
            raise CustomerChurnException(e, sys)

            
    def train_and_save_model(self, pre_processor, X, y):
        try:
            logging.info("Transforming data features using the saved preprocessor.")
            X_transformed = pre_processor.transform(X)

            logging.info("Training the full data on XGBoost.")
            final_model = XGBClassifier(**self.best_params,
                                       eval_metric = 'logloss',
                                       random_state = 42,
                                       verbosity = 0)
            final_model.fit(X_transformed, y)
            logging.info("Model Training Completed!")

            # Making the directory for the artifacts:
            os.makedirs(self.model_training_config.model_training_dir, exist_ok=True)
            joblib.dump(final_model, self.model_file_path)
            logging.info(f"Final model saved to: {self.model_file_path}")
            return self.model_file_path           
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def initiate_model_training(self):
        """
        Initiates the model training pipeline:
        
        Returns:
        --------
        model_trainer_artifacts: an object of ModelTrainerArtifacts class that contains the artifacts of the model training pipeline
        
        """
        try:
            starting_time = time.perf_counter()
            logging.info("Model Training Pipeline Trigerred.")
            X, y = self.load_data()
            pre_processor = self.load_pre_processor()
            model_file_path = self.train_and_save_model(pre_processor=pre_processor, X=X, y=y)
            model_trainer_artifacts = ModelTrainerArtifacts(
                final_model_path=model_file_path
            )

            ending_time = time.perf_counter()
            execution_time = round((ending_time - starting_time)/60, 3)  
            logging.info(f"Model Trainer Artifacts:\n{model_trainer_artifacts}")
            logging.info(f"Model Training Completed | Total Execution Time: {execution_time} min.")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomerChurnException(e, sys)          