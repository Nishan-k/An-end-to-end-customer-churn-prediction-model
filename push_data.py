import pandas as pd
import json
import os
import sys
from dotenv import load_dotenv
import certifi
import numpy as np
import pymongo
from src.customer_churn.logging import logger
from src.customer_churn.utils.main_utils.common import standardize_column_names
from src.customer_churn.exception.exception import CustomerChurnException



# Load the .env file:
load_dotenv()

mongo_uri = os.getenv("MONGO_DB_URL")
ca = certifi.where()


class LoadRetailData:

    def __init__(self):
        pass
    
    def convert_xlsx_to_parquet_and_load(self, file_path, sheet_names):
        """
        This function reads the .xlsx file as Pandas DF, saves it as a Parquet file, reads the parquet file and
        returns as Pandas DF:
        """
        self.file_path = file_path
        self.sheet_names = sheet_names
        df = pd.concat(pd.read_excel(self.file_path, sheet_name=self.sheet_names).values(),
                       ignore_index=True)
        
        # Rename the columns for consistency:
        df = standardize_column_names(df=df)
        
        # Chnage the necessary dtypes:
        df['invoice'] = df['invoice'].astype('str')
        df['stockcode'] = df['stockcode'].astype('str')
        df['description'] = df['description'].astype('str')
        df['price'] = df['price'].replace(0.0, np.nan)
        
     

        # Save the DF as Parquet file:
        df.to_parquet("data/online_retail.parquet")

        # Read the Parquet file:
        df = pd.read_parquet("data/online_retail.parquet")

        return df


    def parquest_data_to_json_converter(self, df):
        self.df = df
        try:
            df.reset_index(drop=True, inplace=True)
            # Convert the DF to json file:
            records = list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomerChurnException(e, sys)


    def insert_data_to_mongodb(self, records, database, collection):
        try:
            self.records = records
            self.database = database
            self.collection = collection

            self.mongo_client = pymongo.MongoClient(mongo_uri)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return (len(self.records))
        except Exception as e:
            raise CustomerChurnException(e, sys)


if __name__ == "__main__":
    FILE_PATH = "data/online_retail_II.xlsx"
    SHEET_NAMES = ['Year 2009-2010', 'Year 2010-2011']
    DATABASE = "online_retail"
    COLLECTION = "customer_data"
    retail_obj = LoadRetailData()
    parq_as_df = retail_obj.convert_xlsx_to_parquet_and_load(file_path=FILE_PATH, sheet_names=SHEET_NAMES)
    data_as_json = retail_obj.parquest_data_to_json_converter(df=parq_as_df)
    num_of_records = retail_obj.insert_data_to_mongodb(records=data_as_json, 
                                                       database=DATABASE, 
                                                       collection=COLLECTION)
    print(f"A total of: {num_of_records} were inserted.")

