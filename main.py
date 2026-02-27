from src.customer_churn.config.configuration import (TrainingPipelineConfig, DataIngestionConfig)
from src.customer_churn.components.data_ingestion import DataIngestion
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging import logger



if __name__ == "__main__":
    # 1. Create an object for TrainingPipelineConfig():
    training_pipeline_config = TrainingPipelineConfig()


    # 2. Create an object for DataIngestionConfig():
    data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    logger.log_separator(section_name="Data Ingsetion")
    data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
    logger.logging.info(f"Data Ingestion Artifacts:\n{data_ingestion_artifacts}")
    logger.logging.info("Data Ingestion Completed with Success !!")

