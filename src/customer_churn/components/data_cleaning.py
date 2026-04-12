import pandas as pd
import os
import numpy as np
import sys
from sklearn.impute import SimpleImputer
from src.customer_churn.entity.artifact_entity import DataValidationArtifacts, DataCleaningArtifacts
from src.customer_churn.config.configuration import DataCleaningConfig, TrainingPipelineConfig
from src.customer_churn.logging.logger import logging
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.utils.main_utils.common import read_yaml_file
import joblib
import time
import yaml



class DataCleaning:
    def __init__(self, data_cleaning_config: DataCleaningConfig, data_validation_artifacts: DataValidationArtifacts):
        try:
            self.data_cleaning_config = data_cleaning_config
            self.data_validation_artifacts = data_validation_artifacts
            self.train_file_path = self.data_validation_artifacts.valid_train_file_path
            self.test_file_path = self.data_validation_artifacts.valid_test_file_path
            # self.train_file_path = train_file_path
            # self.test_file_path = test_file_path
            self.data_cleaning_params = read_yaml_file("params.yaml")['data_cleaning']
            logging.info(f"Data cleaning parameters loaded with success: {self.data_cleaning_params}")
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def handle_duplicates(self, df:pd.DataFrame, df_name:str) -> pd.DataFrame:
        try:
            duplicates = df.duplicated()
            if duplicates.any():
                num_of_duplicates = df[duplicates].shape[0]
                df.drop_duplicates(inplace=True)
                logging.info(f"Total number of duplicates in the given {df_name} dataframe: {num_of_duplicates:,} were dropped.")
            else:
                logging.info(f"No duplicates in {df_name} dataframe.")
            return df
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def fit_price_imputer(self, train_df: pd.DataFrame):
        """
        This function fits and saves the imputer as .pkl file on the training data.
        """
        try:
            strategy = self.data_cleaning_params['imputation_strategy']
            imputer_path = self.data_cleaning_config.missing_data_imputer_file

            # Fit the imputer:
            price_imputer = SimpleImputer(strategy=strategy)
            price_imputer.fit(train_df[['price']])
            
            # Create the directory to store the imputer:
            os.makedirs(os.path.dirname(imputer_path), exist_ok=True)            
            joblib.dump(price_imputer, imputer_path)
            
            logging.info(f"SimpleImputer used {strategy} as strategy and saved the imputer at: {imputer_path} path.")
            return imputer_path
        except Exception as e:
            raise CustomerChurnException(e, sys)

            
    def transform_price_imputer(self, df:pd.DataFrame, df_name:str) -> pd.DataFrame:
        """
        This function loads the trained/fitted imputer and transforms on the unseen/test data:
        """
        try:
            imputer_path = self.data_cleaning_config.missing_data_imputer_file
            imputer = joblib.load(imputer_path)
            logging.info(f"Imputer loading Success from {imputer_path}")
            df['price'] = imputer.transform(df[['price']]).ravel()
            logging.info(f"Dataframe: {df_name} on column Price, imputation success!")
            return df
        except Exception as e:
            raise CustomerChurnException(e, sys)

    
    def report_missing_values(self, df:pd.DataFrame, df_name:str) -> bool:
        missing_report = False
        try:
            missing_vals = df.isna().sum()            
            if (missing_vals > 0).any():
                missing_report = True
                logging.info(f"{df_name} dataframe missing value report:\n{missing_vals}")
                return missing_report
            else:
                logging.info(f"{df_name} has no missing(nan) values.")
                return missing_report
        except Exception as e:
            raise CustomerChurnException(e, sys)
        

    def handle_cancelled_orders(self, df:pd.DataFrame, df_name:str) -> pd.DataFrame:
        """
        For returned orders flag 1 else 0
        For cancelled orders flag 1 else 0
        Creates a new column "Quantity_abs" with absolute values for Quantity column for feature selection in later stage.
        """
        try:
            df['order_issue'] = ((df['quantity'] < 0) | (df['invoice'].str.startswith('C'))).astype(int)
            df['quantity_abs'] = df['quantity'].map(lambda q: np.abs(q))
            logging.info(f"Total number of cancelled orders in {df_name} dataframe: {df['order_issue'].sum():,}")
            return df
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def handle_outliers_w_clipping(self, df:pd.DataFrame, caps=None) -> pd.DataFrame:
        """
        Clips two columns: ['quantity_abs', 'price'] with 99th Quantile to test linear models.
        """
        
        try:
            # For training data:
            if caps is None:
                capping_quantile = self.data_cleaning_params['capping_quantiles']
                caps = {
                    'quantity_abs': float(df['quantity_abs'].quantile(capping_quantile)), 
                    'price': float(df['price'].quantile(capping_quantile))
                    }
                logging.info(f"Clipping values calculated on training data: {caps}")

                # Save caps in the .yaml file to be used for testing file:
                caps_file_path_yaml = self.data_cleaning_config.caps_yaml_file
                with open(caps_file_path_yaml, "w") as f:
                    yaml.dump(caps, f)
                logging.info(f"Clipping data from training file saved at: {caps_file_path_yaml}")

                # Clip the training data:
                for col, cap in caps.items():
                    df[f"{col}_clipped"] = df[col].clip(upper=cap)
                    logging.info(f"Column: {col} has been clipped by 99th quantile: {cap}")                
                return df, caps_file_path_yaml

            # For testing data:
            for col, cap in caps.items():
                df[f"{col}_clipped"] = df[col].clip(upper=cap)
                logging.info(f"Column: {col} has been clipped using {cap}")
            return df
        except Exception as e:
            raise CustomerChurnException(e, sys)

    
    def initiate_cleaning_training_data(self):
        """
        Cleans the training data:
        """
        imputer_path=None
        logging.info("Cleaning Training Data Pipeline Started:")
        starting_time = time.perf_counter()
        try:
            train_df = pd.read_parquet(self.train_file_path)
            logging.info("Reading training data success!")
            train_df_duplicates_removed = self.handle_duplicates(df=train_df, df_name='training')
            logging.info("Duplicates handled in the training data.")
            data_missing = self.report_missing_values(df=train_df_duplicates_removed, df_name='training')
            
            # Imputer is fitted regardles of missing values on training data to handle future missing values on test data:
            imputer_path = self.fit_price_imputer(train_df=train_df_duplicates_removed)
            if data_missing:
                missing_data_imputed_df = self.transform_price_imputer(df=train_df_duplicates_removed, df_name='training')
                clean_df = self.handle_cancelled_orders(df=missing_data_imputed_df, df_name='training')
            else:
                clean_df = self.handle_cancelled_orders(df=train_df_duplicates_removed, df_name='training')
                
            # Create a copy of the dataframe for linear and non-linear models:
            unclipped_df = clean_df.copy()
            
             # Save the uclipped training df file:
            unclipped_training_filepath = self.data_cleaning_config.unclipped_train_file_path
            os.makedirs(os.path.dirname(unclipped_training_filepath), exist_ok=True)         
            unclipped_df.to_parquet(unclipped_training_filepath, index=False)
            logging.info(f"Unclipped training data saved for tree models in: {unclipped_training_filepath}")

            # Clip the training data:
            clipped_training_filepath = self.data_cleaning_config.clipped_train_file_path
            os.makedirs(os.path.dirname(clipped_training_filepath), exist_ok=True)
            clipped_df, caps_file_path_yaml = self.handle_outliers_w_clipping(df=clean_df)
            clipped_df.to_parquet(clipped_training_filepath, index=False)
            logging.info(f"Clipped training data saved for linear models in: {clipped_training_filepath}")

            # Return the artifacts:
            data_cleaning_training_df_artifacts = DataCleaningArtifacts(
                clipped_training_file_path=clipped_training_filepath,
                clipped_testing_file_path=None,
                unclipped_training_file_path=unclipped_training_filepath,
                unclipped_testing_file_path=None,
                simple_imputer_file_path=imputer_path,
                caps_file_path_yaml = caps_file_path_yaml
            )
            ending_time = time.perf_counter()
            execution_time = round((ending_time - starting_time)/60, 3)
            logging.info(f"Data Cleaning Artifacts:\n{data_cleaning_training_df_artifacts}\n")
            logging.info(f"Cleaning Training Data Completed | Total Execution Time: {execution_time} min.")
            return data_cleaning_training_df_artifacts
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
        
    def initiate_cleaning_test_data(self):
        """
        Cleans the testing data:
        """
        logging.info("Cleaning Testing Data Pipeline Started:")
        starting_time = time.perf_counter()      
        try:
            test_df = pd.read_parquet(self.test_file_path)
            logging.info("Reading testing data success!!")
            test_df_duplicates_removed = self.handle_duplicates(df=test_df, df_name='testing')
            logging.info("Duplicates removed in the testing data.")
            has_missing_price = self.report_missing_values(df=test_df_duplicates_removed, df_name='testing')
            if has_missing_price:
                missing_data_imputed_df = self.transform_price_imputer(df=test_df_duplicates_removed, df_name='testing')
                clean_df = self.handle_cancelled_orders(df=missing_data_imputed_df, df_name='testing')
            else:
                logging.info("Testing data has no missing values (NaN)")
                clean_df = self.handle_cancelled_orders(df=test_df_duplicates_removed,  df_name='testing')

            # Create a copy of the dataframe for linear and non-linear model:
            unclipped_df = clean_df.copy()

            # Save the unclipped testing dataframe:
            unclipped_testing_filepath = self.data_cleaning_config.unclipped_test_file_path
            os.makedirs(os.path.dirname(unclipped_testing_filepath), exist_ok=True)
            unclipped_df.to_parquet(unclipped_testing_filepath, index=False)
            logging.info(f"Unclipped test file saved at: {unclipped_testing_filepath}")

            # Clip the testing file:
            clipped_testing_filepath = self.data_cleaning_config.clipped_test_file_path
            os.makedirs(os.path.dirname(clipped_testing_filepath), exist_ok=True)
            cap_file = read_yaml_file(self.data_cleaning_config.caps_yaml_file)
            clipped_df = self.handle_outliers_w_clipping(df=clean_df, caps=cap_file)
            clipped_df.to_parquet(clipped_testing_filepath, index=False)
            logging.info(f"Clipped testing data saved at: {clipped_testing_filepath}")

            data_cleaning_testing_df_artifacts = DataCleaningArtifacts(
                clipped_training_file_path=None,
                clipped_testing_file_path=clipped_testing_filepath,
                unclipped_training_file_path=None,
                unclipped_testing_file_path=unclipped_testing_filepath,
                simple_imputer_file_path=None,
                caps_file_path_yaml = None
            )
            ending_time = time.perf_counter()
            execution_time = round((ending_time - starting_time)/60, 3)
            logging.info(f"Data Cleaning Artifacts:\n{data_cleaning_testing_df_artifacts}\n")
            logging.info(f"Cleaning Testing Data Completed | Total Execution Time: {execution_time} min.")
            return data_cleaning_testing_df_artifacts
        except Exception as e:
            raise CustomerChurnException(e, sys)
            
        

# if __name__ == "__main__":
    # Test the training data:
    # data_cleaning_config = DataCleaningConfig()
    # train_file_path = "Artifacts/data_validation/valid/train.parquet"
    # test_file_path = "Artifacts/data_validation/valid/test.parquet"
    # log_separator(section_name="Data Cleaning: Training Data", stage="start")
    # data_cleaning = DataCleaning(data_cleaning_config=data_cleaning_config,
    #                              train_file_path=train_file_path,
    #                              test_file_path=test_file_path)    
    # data_cleaning.initiate_cleaning_training_data()
    # log_separator(section_name="Data Cleaning: Training Data", stage="end")


    # Test the testing data:
    # data_cleaning_config = DataCleaningConfig()
    # train_file_path = "Artifacts/data_validation/valid/train.parquet"
    # test_file_path = "Artifacts/data_validation/valid/test.parquet"
    # log_separator(section_name="Data Cleaning: Testing Data", stage="start")
    # data_cleaning = DataCleaning(data_cleaning_config=data_cleaning_config,
    #                              train_file_path=train_file_path,
    #                              test_file_path=test_file_path)    
    # data_cleaning.initiate_cleaning_test_data()
    # log_separator(section_name="Data Cleaning: Testing Data", stage="end")

