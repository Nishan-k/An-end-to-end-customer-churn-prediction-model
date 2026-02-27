from src.customer_churn.logging import logger
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.entity.artifact_entity import DataIngestionArtifacts
from src.customer_churn.config.configuration import DataIngestionConfig
import sys
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import time
import os



# Load .env values:
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


# Create a class for the Data Ingestion Process:

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    
    def import_collection_as_df(self):
        """
        Loads the collection from MongoDB:
        """
        start_time = time.perf_counter()
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            # Necessary data-types changes:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], unit='ms')
            df['Customer ID'] = df['Customer ID'].astype('Int64') 

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])

            df.replace({"na": np.nan}, inplace=True)
            end_time = time.perf_counter()
            execution_time = round((end_time - start_time)/60, 3)
            logger.logging.info(
                f"Data Import From MongoDB As DataFrame Success | Records: {df.shape[0]:,}"
                f" Total Time Execution: {execution_time} min"
            )
            return df
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    
    def export_data_into_feature_store(self, dataframe:pd.DataFrame):
        """
        Stores the main data as a backup file in the feature store as Parquet
        """
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # Create the folder:
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_parquet(feature_store_file_path, index=False)
            logger.logging.info(f"Dataframe saved as Parquet file in feature store in the path: {dir_path} as a backup file.")
            return dataframe
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Splits the dataframe into train and test file as Parquet
        """
        try:
            split_ratio = self.data_ingestion_config.train_test_split_ratio
            train_set, test_set = train_test_split(dataframe, test_size=split_ratio)
            logger.logging.info(f"Train-test split completed | Training: {(1-split_ratio) * 100}% and Test: {split_ratio * 100}%")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_parquet(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_parquet(self.data_ingestion_config.testing_file_path, index=False)
            logger.logging.info(f"Exporting train and test data as Parquet completed.")
        except Exception as e:
            raise CustomerChurnException(e, sys)
    
    def initiate_data_ingestion(self):
        """
        Trigger the entire data ingestion process.
        """
        try:
            starting_time = time.perf_counter()
            logger.logging.info("Data Ingestion Pipeline Started.")
            df = self.import_collection_as_df()
            dataframe = self.export_data_into_feature_store(dataframe=df)
            self.split_data_as_train_test(dataframe=dataframe)
            ending_time = time.perf_counter()
            execution_time = round((ending_time - starting_time)/60, 3)
            logger.logging.info(f"Data Ingestion Completed | Total Execution Time: {execution_time} min")
            # Artifacts:
            data_ingestion_artifact = DataIngestionArtifacts(
                training_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomerChurnException(e, sys)

