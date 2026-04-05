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
        

# Data Clearning Config:
class DataCleaningConfig:
    def __init__(self):
        self.data_cleaning_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR,
                                                   training_pipeline.DATA_CLEANING_DIR_NAME)
        self.clipped_dir: str = os.path.join(self.data_cleaning_dir, training_pipeline.DATA_CLEANING_CLIPPED_DATA)
        self.clipped_train_file_path: str = os.path.join(self.clipped_dir, training_pipeline.TRAIN_FILE_NAME)
        self.clipped_test_file_path: str = os.path.join(self.clipped_dir, training_pipeline.TEST_FILE_NAME)

        self.unclipped_dir: str = os.path.join(self.data_cleaning_dir, training_pipeline.DATA_CLEANING_UNCLIPPED_DATA)        
        self.unclipped_train_file_path: str = os.path.join(self.unclipped_dir, training_pipeline.TRAIN_FILE_NAME)
        self.unclipped_test_file_path: str = os.path.join(self.unclipped_dir, training_pipeline.TEST_FILE_NAME)

        self.missing_data_imputer: str = os.path.join(self.data_cleaning_dir, training_pipeline.DATA_CLEANING_IMPUTER_FOR_MISSING_DATA)
        self.missing_data_imputer_file: str = os.path.join(self.missing_data_imputer, training_pipeline.DATA_CLEANING_IMPUTER_FILE_NAME)

        self.caps_yaml_file: str = os.path.join(self.data_cleaning_dir, training_pipeline.DATA_CLEANING_CAPS_FILE)
        

# Feature Engineering Config:        
class FeatureEngineeringConfig:
    def __init__(self):
        self.feature_engineer_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR,
                                                      training_pipeline.FEATURE_ENGINEERING_DIR_NAME)
        self.clipped_dir: str = os.path.join(self.feature_engineer_dir, training_pipeline.FEATURE_ENGINEERING_CLIPPED_DIR_NAME)
        self.clipped_train_file_path: str = os.path.join(self.clipped_dir, training_pipeline.TRAIN_FILE_NAME)
        self.clipped_test_file_path: str = os.path.join(self.clipped_dir, training_pipeline.TEST_FILE_NAME)

        self.unclipped_dir: str = os.path.join(self.feature_engineer_dir, training_pipeline.FEATURE_ENGINEERING_UNCLIPPED_DIR_NAME)
        self.unclipped_train_file_path: str = os.path.join(self.unclipped_dir, training_pipeline.TRAIN_FILE_NAME)
        self.unclipped_test_file_path: str = os.path.join(self.unclipped_dir, training_pipeline.TEST_FILE_NAME)



# Pre Processing Config:
class DataPreprocessingConfig:
    def __init__(self):
        self.pre_processing_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR,
                                                    training_pipeline.PRE_PROCESSING_DIR_NAME)
        self.pre_processing_file_name: str = os.path.join(self.pre_processing_dir, 
                                                          training_pipeline.PRE_PROCESSING_FILE_NAME)
        self.pre_processing_full_data_dir:str = os.path.join(self.pre_processing_dir, training_pipeline.PRE_PROCESSING_FULL_DATASET)

        