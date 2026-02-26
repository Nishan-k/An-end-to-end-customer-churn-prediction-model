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

############################################
# Constant variables for the Data Ingestion:
############################################
DATA_INGESTION_COLLECION_NAME: str = "customer_data"
DATA_INGESTION_DATABASE_NAME: str = "online_retail"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_FEATURE_STOR_DIR: str = "feature_store"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2