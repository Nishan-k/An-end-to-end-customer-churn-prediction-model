from src.customer_churn.constants import training_pipeline
from datetime import datetime
import os



class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        self.timestamp:str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_dir = training_pipeline.ARTIFACT_DIR


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir,
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
