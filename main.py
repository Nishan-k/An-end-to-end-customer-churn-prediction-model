from src.customer_churn.config.configuration import (DataIngestionConfig, DataValidationConfig,
                                                     DataCleaningConfig)
from src.customer_churn.components.data_ingestion import DataIngestion
from src.customer_churn.components.data_validation import DataValidation
from src.customer_churn.components.data_cleaning import DataCleaning
from src.customer_churn.utils.main_utils.common import read_yaml_file
from src.customer_churn.logging.logger import log_separator

# Read the params:
params = read_yaml_file("params.yaml")

# Cut-off date for data-ingestion :
churn_cutoff_date = params['data_ingestion']['churn_cutoff_date']




if __name__ == "__main__":
    # 2. Data Ingestion:
    log_separator(section_name="Data Ingsetion", stage="Start")
    data_ingestion_config = DataIngestionConfig()    
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config, cut_off_date=churn_cutoff_date)    
    data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
    log_separator(section_name="Data Ingsetion", stage="End")

    # 2. Data Validation:
    log_separator(section_name="Data Validation", stage="start")
    data_validation_config = DataValidationConfig()
    data_validation = DataValidation(data_validation_config=data_validation_config,
                                     data_ingestion_artifacts=data_ingestion_artifacts)    
    data_validation_artifacts = data_validation.initiate_data_validation()
    log_separator(section_name="Data Validation", stage="end")
    
    # 3. Data Cleaning:
    # 3.1 Training Data:
    log_separator(section_name="Data Cleaning: Training Data", stage="start")
    data_cleaning_config = DataCleaningConfig()    
    data_cleaning = DataCleaning(data_cleaning_config=data_cleaning_config, data_validation_artifacts=data_validation_artifacts)
    data_cleaning_training_data_artifacts = data_cleaning.initiate_cleaning_training_data()
    log_separator(section_name="Data Cleaning: Training Data", stage="end")
    

