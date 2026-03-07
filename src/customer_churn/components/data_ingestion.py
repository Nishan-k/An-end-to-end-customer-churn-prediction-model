from src.customer_churn.logging.logger import log_separator, logging
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
from typing import Optional


# Load .env values:
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


# Create a class for the Data Ingestion Process:

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig, cut_off_date: str):
        """
        Parameters:
        ===========
        data_ingestion_config: An object of DataIngestionConfig class.
        cut_off_date: samples to be dropped which are outside the observation and churn window.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.cut_off_date_dt = pd.to_datetime(cut_off_date)
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    
    def import_collection_as_df(self):
        """
        Loads the collection from MongoDB:
        """

        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            mongo_client = MongoClient(MONGO_DB_URL)

            # Pull the raw data from MongoDO:
            collection = mongo_client[database_name][collection_name]
            raw_df = pd.DataFrame(list(collection.find()))
            logging.info(f"RAW data shape: {raw_df.shape} pulled from MongoDB.")

            # Drop the rows with missing Customer ID:
            initial_count = len(raw_df)
            df = raw_df.dropna(subset=['customerid']).copy()
            dropped = initial_count - len(df)
            logging.info(f"Total number of dropped rows: {dropped:,} or {(dropped/initial_count)*100:.2f}% with missing Customer ID")

            # Necessary data-types changes:
            df['invoicedate'] = pd.to_datetime(df['invoicedate'], unit='ms')
            df['customerid'] = df['customerid'].astype('string')

            # Drop the indexes outside the Observation and churn window:
            indexes_to_drop = df[df['invoicedate'] > self.cut_off_date_dt].index
            before_cutoff = df.shape[0]
            df.drop(index=indexes_to_drop, inplace=True)
            after_cutoff = df.shape[0]
            logging.info(f"Cut-off date: {self.cut_off_date_dt} | Before: {before_cutoff:,} samples | After: {after_cutoff:,} samples. ")
            df.reset_index(drop=True, inplace=True)

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])

            df.replace({"na": np.nan}, inplace=True)
            logging.info(
                f"Data Import From MongoDB As DataFrame Success | Records: {df.shape[0]:,} & Features: {df.shape[1]}")
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
            logging.info(f"Dataframe saved as Parquet file in feature store in the path: {dir_path} as a backup file.")
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
            logging.info(f"Train-test split completed | Training: {train_set.shape} ({(1-split_ratio) * 100}%)  and Test: {test_set.shape} ({split_ratio * 100}%).")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_set.to_parquet(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_parquet(self.data_ingestion_config.testing_file_path, index=False)
            logging.info(f"Exporting train and test data as Parquet completed.")
        except Exception as e:
            raise CustomerChurnException(e, sys)
        
    
    def initiate_data_ingestion(self):
        """
        Trigger the entire data ingestion process.        
        """

        starting_time = time.perf_counter()
        logging.info("Data Ingestion Pipeline Started:")
        
        try:            
            df = self.import_collection_as_df()
            dataframe = self.export_data_into_feature_store(dataframe=df)
            self.split_data_as_train_test(dataframe=dataframe)

            ending_time = time.perf_counter()
            execution_time = round((ending_time - starting_time)/60, 3)            
            
            # Artifacts:
            data_ingestion_artifact = DataIngestionArtifacts(
                training_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info(f"Data Ingestion Artifacts:\n{data_ingestion_artifact}\n")
            logging.info(f"Data Ingestion Completed | Total Execution Time: {execution_time} min.")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomerChurnException(e, sys)



