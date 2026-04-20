from src.customer_churn.config.configuration import (DataIngestionConfig, DataValidationConfig,
                                                     DataCleaningConfig, FeatureEngineeringConfig, 
                                                     DataPreprocessingConfig, ModelTrainingConfig)
from src.customer_churn.components.data_ingestion import DataIngestion
from src.customer_churn.components.data_validation import DataValidation
from src.customer_churn.components.data_cleaning import DataCleaning
from src.customer_churn.components.preprocessor import DataPreProcessing
from src.customer_churn.components.feature_engineering import FeatureEngineering
from src.customer_churn.components.model_trainer import ModelTraining
from src.customer_churn.utils.main_utils.common import read_yaml_file
from src.customer_churn.logging.logger import log_separator
from src.customer_churn.entity.artifact_entity import DataCleaningArtifacts

# Read the params:
params = read_yaml_file("params.yaml")

# Cut-off date for data-ingestion :
churn_cutoff_date = params['data_ingestion']['churn_cutoff_date']




if __name__ == "__main__":
    # 1. Data Ingestion:
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
    

    # 3.2 Testing Data:
    log_separator(section_name="Data Cleaning: Testing Data", stage="start")
    data_cleaning_config = DataCleaningConfig()    
    data_cleaning = DataCleaning(data_cleaning_config=data_cleaning_config, data_validation_artifacts=data_validation_artifacts)
    data_cleaning_testing_data_artifacts = data_cleaning.initiate_cleaning_test_data()
    log_separator(section_name="Data Cleaning: Testing Data", stage="end")
    

    # 3.3 Combining the data cleaning artifacts:
    combined_data_cleaning_artifacts = DataCleaningArtifacts(
        clipped_training_file_path=data_cleaning_training_data_artifacts.clipped_training_file_path,
        clipped_testing_file_path=data_cleaning_testing_data_artifacts.clipped_testing_file_path,
        unclipped_training_file_path=data_cleaning_training_data_artifacts.unclipped_training_file_path,
        unclipped_testing_file_path=data_cleaning_testing_data_artifacts.unclipped_testing_file_path)

    
    # 4. Feature Engineering:
    # # 4.1 For clipped data:
    # log_separator(section_name="Feature Engineering: clipped dataset", stage="start")
    # feature_engineering_config = FeatureEngineeringConfig()
    # feature_engineering = FeatureEngineering(feature_engineering_config=feature_engineering_config,
    #                                          data_cleaning_artifacts=combined_data_cleaning_artifacts,
    #                                          churn_cutoff_date=churn_cutoff_date)
    # feature_engineering_artifacts_clipped = feature_engineering.initiate_feature_engineering(branch_name='clipped')
    # log_separator(section_name="Feature Engineering", stage="end")

    # # 4.2 For unclipped data:
    log_separator(section_name="Feature Engineering", stage="start")
    feature_engineering_config = FeatureEngineeringConfig()
    feature_engineering = FeatureEngineering(feature_engineering_config=feature_engineering_config,
                                             data_cleaning_artifacts=combined_data_cleaning_artifacts,
                                             churn_cutoff_date=churn_cutoff_date)
    feature_engineering_artifacts_unclipped = feature_engineering.initiate_feature_engineering(branch_name='unclipped')
    log_separator(section_name="Feature Engineering: unclipped dataset", stage="end")
    


    # 5. Creating Pre-processor:
    log_separator(section_name="Creating Pre-processor", stage="start")
    pre_processing_config = DataPreprocessingConfig()
    data_pre_processing = DataPreProcessing(data_pre_processing_config=pre_processing_config,
                                            feature_engineering_artifacts=feature_engineering_artifacts_unclipped)
    pre_processor_artifacts = data_pre_processing.initiate_pre_processor()
    log_separator(section_name="Creating Pre-processor", stage="end")


    # 6. Model Training:
    log_separator(section_name="Model Training", stage="Start")
    model_training_config = ModelTrainingConfig()
    model_trainer = ModelTraining(model_training_config=model_training_config,
                                  pre_processing_artifacts=pre_processor_artifacts)
    model_trainer_artifacts = model_trainer.initiate_model_training()
    log_separator(section_name="Model Training", stage="end")
    

