import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import sys
import joblib
from src.customer_churn.logging.logger import logging
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.config.configuration import DataPreprocessingConfig
from src.customer_churn.entity.artifact_entity import FeatureEngineeringArtifacts, PreProcessingArtifacts
import os
import time


class DataPreProcessing:

    def __init__(self, data_pre_processing_config: DataPreprocessingConfig, 
                 feature_engineering_artifacts: FeatureEngineeringArtifacts):
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
        Concatenates the finalized data branch i.e. Unclipped data (training and testing data).

        Returns:
        ---------
        full_data: A full concatenated unclipped dataset that will be used to fit and create the pre-processor.
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


    def separate_col_by_dtypes(self, full_dataset:pd.DataFrame):
        """
        Finds the numerical and categorical features for the given dataframe.

        Returns:
        --------
        numerical_features: A list of numerical features
        categorical_features: A list of categorical features
        """
        try:
            feature_to_drop = {'customerid', 'churn'}
            numerical_features = full_dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numerical_features = [feature for feature in numerical_features if feature not in feature_to_drop]
            categorical_features = full_dataset.select_dtypes(include=['object', 'string']).columns.tolist()
            logging.info(f"The numerical features are:\n{numerical_features} and the categoical features are:\n{categorical_features}")
            logging.info(f"The dataset has {len(numerical_features)} numerical features and {len(categorical_features)} categorical features.")
            return numerical_features, categorical_features
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def create_dep_indep_features(self, full_dataset:pd.DataFrame):
        """
        Splits the full data into dependent(y) and independent(X) features.

        Returns:
        ---------
        X: Independent data
        y: Dependent data
        X_path: The path where the independent data is stored
        y_path: The path where the dependent data is stored
        """
        try:
            X = full_dataset.drop(columns=['customerid', 'churn'])
            y = full_dataset['churn']
            logging.info(f"Dependent and independent features created. Independent: {X.shape} and dependent: {y.shape}")

            # Create the directory to store the dependent and independent data:
            os.makedirs(self.data_pre_processing_config.pre_processing_full_data_dir, exist_ok=True)
            X_path = os.path.join(self.data_pre_processing_config.pre_processing_full_data_dir, "X.parquet")
            y_path = os.path.join(self.data_pre_processing_config.pre_processing_full_data_dir, "y.parquet")
            X.to_parquet(X_path, index=False)
            y.to_frame().to_parquet(y_path, index=False)          
            logging.info(f"Independent feature saved at: {X_path}")
            logging.info(f"Dependent features saved at: {y_path}")
            return X, y, X_path, y_path
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def create_preprocessor(self, X, y, numerical_features, categorical_features):
        """
        Is responsible to fit and save the finalized
        pre-processor and encoder as a .pkl file.


        Returns:
        --------
        pre_processor_path: Path where the fitted pre-processor will be stored.
        """
        try:
            logging.info("Creating the pre-processing file:")
            preprocessor = ColumnTransformer([
                ('numerical', RobustScaler(), numerical_features),
                ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ])
            preprocessor.fit(X, y)
            logging.info(f"Preprocessor fitted successfully.\n{preprocessor}")

            # Create the directory to save the artifacts:
            os.makedirs(os.path.dirname(self.data_pre_processing_config.pre_processing_file_name), exist_ok=True)

            # Save the fitted pre-processor:
            joblib.dump(preprocessor, self.data_pre_processing_config.pre_processing_file_name)
            logging.info(f"Preprocessor saved to {self.data_pre_processing_config.pre_processing_file_name}")
            
            pre_processor_path = self.data_pre_processing_config.pre_processing_file_name
            return pre_processor_path
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def get_artifacts(self, pre_processor_file_path, X_path, y_path) -> PreProcessingArtifacts:
        """
        Is responsible to generate the artifacts for the DataPreProcessing Pipeline.

        Returns:
        ---------
        pre_processing_artifacts: An object of PreProcessingArtifacts class
        """
        try:
            pre_processing_artifacts = PreProcessingArtifacts(
                pre_processor_file_path=pre_processor_file_path,
                X_path=X_path,
                y_path=y_path
            )
            return pre_processing_artifacts
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def initiate_pre_processor(self):
        """
        Is responsible to trigger the entire data preprocessing pipeline.

        Returns:
        ----------
        pre_processing_artifacts: The artifact for DataPreProcessing component
        """
        try:
            starting_time = time.perf_counter()
            logging.info("Pre-Processor Pipeline Trigerred.")
            
            full_data = self.create_full_training_dataset()
            numerical_features, categorical_features = self.separate_col_by_dtypes(full_dataset=full_data)
            
            X, y, X_path, y_path = self.create_dep_indep_features(full_dataset=full_data)
            pre_processor_file_path = self.create_preprocessor(X=X, y=y, 
                                                           numerical_features=numerical_features, 
                                                           categorical_features=categorical_features)
            pre_processing_artifacts = self.get_artifacts(pre_processor_file_path=pre_processor_file_path,
                                                          X_path=X_path, y_path=y_path)
            ending_time = time.perf_counter()
            execution_time = round((ending_time - starting_time)/60, 3)    
            logging.info(f"Pre-Processing Artifacts:\n{pre_processing_artifacts}")
            logging.info(f"Pre-processor completed! | Total Execution Time: {execution_time} min.")
            return pre_processing_artifacts

        except Exception as e:
            raise CustomerChurnException(e, sys)