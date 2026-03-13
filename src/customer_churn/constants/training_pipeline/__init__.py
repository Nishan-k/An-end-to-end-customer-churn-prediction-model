import os
import numpy as np



###############################################
# Constant variables for the Training Pipeline:
###############################################

PIPELINE_NAME: str = "Customer_Churn"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "online_retail.parquet"
TRAIN_FILE_NAME: str = "train.parquet"
TEST_FILE_NAME: str = "test.parquet"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")

############################################
# Constant variables for the Data Ingestion:
############################################
DATA_INGESTION_COLLECION_NAME: str = "customer_data"
DATA_INGESTION_DATABASE_NAME: str = "online_retail"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_FEATURE_STOR_DIR: str = "feature_store"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2


############################################
# Constant variables for the Data Validation:
############################################
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALIDATED_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRFIT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRFIT_REPORT_FILE_NAME: str = "report.yaml"


############################################
# Constant variables for the Data Cleaning:
############################################
DATA_CLEANING_DIR_NAME: str = "data_cleaning"
DATA_CLEANING_CLIPPED_DATA: str = "data_clipped"
DATA_CLEANING_UNCLIPPED_DATA: str = "data_unclipped"
DATA_CLEANING_IMPUTER_FOR_MISSING_DATA: str = "missing_data_imputer"
DATA_CLEANING_IMPUTER_FILE_NAME: str = "SimpleImputerForPrice.joblib"
DATA_CLEANING_CAPS_FILE: str = "caps.yaml"


################################################
# Constant variables for the Feature Engineering:
################################################
FEATURE_ENGINEERING_DIR_NAME: str = "feature_engineering"
FEATURE_ENGINEERING_CLIPPED_DIR_NAME: str = "data_clipped"
FEATURE_ENGINEERING_UNCLIPPED_DIR_NAME: str = "data_unclipped"
