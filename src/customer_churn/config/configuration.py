from src.customer_churn.constants import training_pipeline
from datetime import datetime
import os


# Training Pipeline Config:
class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        self.timestamp:str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_dir = training_pipeline.ARTIFACT_DIR


# Data Ingestion Config:
class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR,
                                                    training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir,
                                                         training_pipeline.DATA_INGESTION_FEATURE_STOR_DIR,
                                                         training_pipeline.FILE_NAME)
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,
                                                    training_pipeline.TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,
                                                   training_pipeline.TEST_FILE_NAME)
        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECION_NAME
        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME


# Data Validation Config:
class DataValidationConfig:
    def __init__(self):
        self.data_validation_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR, 
                                                training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALIDATED_DIR)
        self.invalid_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.drift_report_file_path: str = os.path.join(self.data_validation_dir,
                                                   training_pipeline.DATA_VALIDATION_DRFIT_REPORT_DIR,
                                                   training_pipeline.DATA_VALIDATION_DRFIT_REPORT_FILE_NAME)
        self.valid_train_file_path: str = os.path.join(self.valid_dir,
                                                       training_pipeline.TRAIN_FILE_NAME) 
        self.valid_test_file_path: str = os.path.join(self.valid_dir,
                                                      training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_dir,
                                                         training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_dir,
                                                        training_pipeline.TEST_FILE_NAME)
        self.schema_file_path: str = training_pipeline.SCHEMA_FILE_PATH
        
