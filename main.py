from src.customer_churn.config.configuration import (TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig)
from src.customer_churn.components.data_ingestion import DataIngestion
from src.customer_churn.components.data_validation import DataValidation

from src.customer_churn.logging.logger import log_separator



if __name__ == "__main__":
    # 1. Create an object for TrainingPipelineConfig():
    training_pipeline_config = TrainingPipelineConfig()

    # 2. Create an object for DataIngestionConfig():
    data_ingestion_config = DataIngestionConfig()
    log_separator(section_name="Data Ingsetion", stage="Start")
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config, cut_off_date='2011-03-31')    
    data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
    log_separator(section_name="Data Ingsetion", stage="End")

    # 3. Create an object for DataValidationConfig():
    data_validation_config = DataValidationConfig()
    log_separator(section_name="Data Validation", stage="start")
    data_validation = DataValidation(data_validation_config=data_validation_config,
                                     data_ingestion_artifacts=data_ingestion_artifacts)    
    data_validation_artifacts = data_validation.initiate_data_validation()
    log_separator(section_name="Data Validation", stage="end")
    
    

