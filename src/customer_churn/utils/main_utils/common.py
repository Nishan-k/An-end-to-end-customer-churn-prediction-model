import yaml
from src.customer_churn.exception.exception import CustomerChurnException
from src.customer_churn.logging import logger
import os, sys
import pandas as pd


# 1. To read YAML file:
def read_yaml_file(file_path:str) -> dict:
    """
    Reads the YAML file and returns as Python Dictionary
    """
    try:
        logger.logging.info(f"Reading YAML file from the path: {file_path}")
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomerChurnException(e, sys)
    
# 2. To write the YAML file:
def write_yaml_file(file_path:str, content:object, replace:bool=False) -> None:
    """
    Writes the YAML file.
    """
    try:
        logger.logging.info(f"Writing the YAML file in the given path: {file_path}")
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomerChurnException(e, sys)


# 3. Read the parquet file and return as df:
def read_parquet_file(file_path: str) -> pd.DataFrame:
    """
    Takes in the file path of parquet file, reads it and returns a DF:
    """
    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        raise CustomerChurnException(e, sys)
    