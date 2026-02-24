import pandas as pd
import json
import os
import sys
from dotenv import load_dotenv
import certifi
import numpy as np
import pymongo
from src.customer_churn.logging import logger
from src.customer_churn.exception.exception import CustomerChurnException


# 1. Read the xlsx file as a dataframe:
sheet_names = ['Year 2009-2010', 'Year 2010-2011']
df = pd.concat(
    pd.read_excel("data/online_retail_II.xlsx", sheet_name=sheet_names).values(),
    ignore_index=True
)




