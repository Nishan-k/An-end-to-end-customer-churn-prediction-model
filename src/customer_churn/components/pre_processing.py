import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sys
import joblib
from src.customer_churn.logging.logger import logging
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.config.configuration import DataPreprocessingConfig
from src.customer_churn.entity.artifact_entity import FeatureEngineeringArtifacts
import os

class DataPreProcessing:
    def __init__(self, data_pre_processing_config: DataPreprocessingConfig,
                 feature_engineering_artifacts: FeatureEngineeringArtifacts
                 ):
        try:
            self.data_pre_processing_config = data_pre_processing_config
            self.feature_engineering_artifacts = feature_engineering_artifacts
            self.unclipped_train_file_path = self.feature_engineering_artifacts.unclipped_training_file_path
            self.unclipped_test_file_path = self.feature_engineering_artifacts.unclipped_testing_file_path
            logging.info("Data Pre-Processing Class Initialization Success!")
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    def create_full_training_dataset(self) -> pd.DataFrame: 
        """
        Concatenates and returns the finalized data branch i.e. Unclipped data.
        """
        try:
            logging.info("Concatenating the final dataset.")
            unclipped_train = pd.read_parquet(self.unclipped_train_file_path)
            unclipped_test = pd.read_parquet(self.unclipped_test_file_path)
            full_data = pd.concat([unclipped_train, unclipped_test])
            full_data.dropna(inplace=True)
            logging.info(f"Concatenation completed. Final data size: {full_data.shape}")
            return full_data
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    def separate_col_by_dtypes(full_dataset:pd.DataFrame):
        """
        Returns the numerical and categorical feature names as list:
        """
        try:
            numerical_features = full_dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = full_dataset.select_dtypes(include=['object', 'string']).columns.tolist()
            logging.info(f"The finalized dataset has a total of {len(numerical_features)} numerical features and a total of {len(categorical_features)} categorical features.")
            return (numerical_features, categorical_features)
        except Exception as e:
            raise CustomerChurnException(e, sys)

    
    def create_dep_indep_features(full_dataset:pd.DataFrame):
        """
        Splits the full data into dependent(y) and independent(X) features:
        """
        try:
            full_dataset.dropna(inplace=True)
            X = full_dataset.drop(columns=['customerid', 'churn'])
            y = full_dataset['churn']
            logging.info(f"Dependent and independent features created. Independent: {X.shape} and dependent: {y.shape}")
            return X, y   
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    def create_preprocessor(self, X, y, numerical_features, categorical_features):
        """
        Is responsible to fit and save the finalized
        pre-processor and encoder as a .pkl file.
        """
        try:
            preprocessor = ColumnTransformer([
                ('numerical', RobustScaler(), numerical_features),
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ], remainder='drop')
            preprocessor.fit(X, y)
            logging.info(f"Preprocessor fitted successfully.\n{preprocessor}")

            # Create the directory to save the artifacts:
            os.makedirs(os.path.dirname(self.data_pre_processing_config.pre_processing_file_name), exist_ok=True)

            # Save the fitted pre-processor:
            joblib.dump(preprocessor, self.data_pre_processing_config.pre_processing_file_name)
            logging.info(f"Preprocessor saved to {self.data_pre_processing_config.pre_processing_file_name}")

            return self.data_pre_processing_config.pre_processing_file_name


        except Exception as e:
            raise CustomerChurnException(e, sys)

    def initiate_pre_processor(self):
        """
        Is responsible to trigger the entire data preprocessing pipeline.
        """
        


# if __name__ == "__main__":
#     data_pre_processing_config = DataPreprocessingConfig()
#     unclipped_training_file_path = "Artifacts/feature_engineering/data_unclipped/train.parquet"
#     unclipped_testing_file_path =  "Artifacts/feature_engineering/data_unclipped/test.parquet"
#     data_preprocessing = DataPreProcessing(data_pre_processing_config=data_pre_processing_config,
#                                            unclipped_testing_file_path=unclipped_testing_file_path,
#                                            unclipped_training_file_path=unclipped_training_file_path)
#     data_preprocessing.create_full_training_dataset()


    


