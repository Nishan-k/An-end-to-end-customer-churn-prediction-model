from src.customer_churn.config.configuration import TrainingPipelineConfig, DataValidationConfig
from src.customer_churn.entity.artifact_entity import DataIngestionArtifacts, DataValidationArtifacts   
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging.logger import log_separator, logging
from src.customer_churn.utils.main_utils.common import read_yaml_file, read_parquet_file
import sys
import pandas as pd
from scipy.stats import ks_2samp
import time
import os



class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.train_file_path = self.data_ingestion_artifacts.training_file_path
            self.test_file_path = self.data_ingestion_artifacts.test_file_path

            # Read the required files:
            self.schema_raw = read_yaml_file(self.data_validation_config.schema_file_path)

            # A dictionary of column name and dtype:
            self.schema = {col['name']: col['dtype'] for col in self.schema_raw['columns']}
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def are_column_len_valid(self, df:pd.DataFrame, df_name:str)->bool:
        """
        Takes in the dataframe, checks the number of columns with the expected number of columns defined in the schema (data_schema/schema.yaml)
        """
        try:
            actual_cols = set(df.columns)
            expected_cols = set(self.schema.keys())

            if expected_cols != actual_cols:
                missing_cols = expected_cols - actual_cols
                extra_cols = actual_cols - expected_cols
                error_message = f"Columns Mismatch in {df_name} dataframe. Missing columns: {missing_cols} | Extra columns: {extra_cols}"
                logging.error(error_message)
                raise CustomerChurnException(error_message, sys)
            logging.info(f"{df_name} dataframe: Column length validation success!")
            return True

        except Exception as e:
            if not isinstance(e, CustomerChurnException):
                raise CustomerChurnException(e, sys)
            raise


    def are_dtypes_valid(self, df:pd.DataFrame, df_name:str) -> bool:
        """
        Validates the dtypes of the data frame passed with the dtypes defined in the schema:
        """
        try:
            for col, expected_dtype in self.schema.items():
                actual_dtype = df[col].dtype
                if expected_dtype != actual_dtype:
                    error_message = f"Data-type Mismatch in {df_name} dataframe. Column: {col} | Expected: {expected_dtype} | Got: {actual_dtype}"
                    logging.error(error_message)
                    raise CustomerChurnException(error_message, sys)
                else:
                    logging.info(f"{df_name} dataframe: Data types validation success!")
                    return True
        except Exception as e:
            if not isinstance(e, CustomerChurnException):
                raise CustomerChurnException(e, sys)
            raise
    
    def get_artifacts(self, data_valid_status: bool) -> DataValidationArtifacts:
        if data_valid_status:
                valid_train_path = self.data_validation_config.valid_train_file_path
                valid_test_path = self.data_validation_config.valid_test_file_path
                os.makedirs(os.path.dirname(valid_train_path), exist_ok=True)
                os.makedirs(os.path.dirname(valid_test_path), exist_ok=True)
                self.train_df.to_parquet(valid_train_path)
                self.test_df.to_parquet(valid_test_path)
                data_validation_artifacts = DataValidationArtifacts(
                    validation_status=data_valid_status,
                    valid_train_file_path=valid_train_path,
                    valid_test_file_path=valid_test_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    drift_report_file_path=None
                )
                return data_validation_artifacts
        else:
            invalid_train_path = self.data_validation_config.invalid_train_file_path
            invalid_test_path = self.data_validation_config.invalid_test_file_path
            os.makedirs(os.path.dirname(invalid_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(invalid_test_path), exist_ok=True)
            self.train_df.to_parquet(invalid_train_path)
            self.test_df.to_parquet(invalid_test_path)
            data_validation_artifacts = DataValidationArtifacts(
                validation_status=data_valid_status,
                valid_train_file_path=None,
                valid_test_file_path=None,
                invalid_train_file_path=invalid_train_path,
                invalid_test_file_path=invalid_test_path,
                drift_report_file_path=None
            )
            return data_validation_artifacts
                


    def initiate_data_validation(self):
        """
        Initiates the data-validation pipeline:
        """
        starting_time = time.perf_counter()
        logging.info("Data Validation Pipeline Started:")
        self.train_df = read_parquet_file(self.train_file_path)
        self.test_df = read_parquet_file(self.test_file_path)

        try:          
            # Validate number of columns:
            train_col_status = self.are_column_len_valid(df=self.train_df, df_name='Training')
            test_col_status = self.are_column_len_valid(df=self.test_df, df_name='Testing')

            # Validate data-type:
            train_dtype_status = self.are_dtypes_valid(df=self.train_df, df_name='Training')     
            test_dtype_status = self.are_dtypes_valid(df=self.test_df, df_name='Testing')

            # Prepare for the artifacts:
            if train_col_status and test_col_status and train_dtype_status and test_dtype_status:
                data_validation_artifacts = self.get_artifacts(data_valid_status=True)
            else:
                data_validation_artifacts = self.get_artifacts(data_valid_status=False)
            ending_time = time.perf_counter()
            execution_time = round((ending_time - starting_time)/60, 3)    
            logging.info(f"Data Validation Artifacts:\n{data_validation_artifacts}\n")
            logging.info(f"Data Validation Completed | Total Execution Time: {execution_time} min.")
            return data_validation_artifacts
                     
        except Exception as e:
            raise CustomerChurnException(e, sys)

    