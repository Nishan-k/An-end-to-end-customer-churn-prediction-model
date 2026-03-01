import logging
import os
from datetime import datetime
from src.customer_churn.exception.exception import CustomerChurnException
import sys

# Create the format for the log folder and file:
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"


# Create a path for the log folder to store the log files:
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the directory for the logs:
os.makedirs(logs_path, exist_ok=True)

# Create a full path to store the logs:
LOGS_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Logs configuration setting:
logging.basicConfig(
    filename=LOGS_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


# A function to separate the logs based on the provided section names:
def log_separator(section_name, stage):
    """
    This function creates a separator lines for different logs for ease of visual understanding:
    """
    separator_line = "-" * 100  
    try:
        if section_name and stage:
            if stage.lower() == 'start':
                logging.info(f"{separator_line}")
                logging.info(f"{stage.upper()} OF: {section_name.upper()}")
                logging.info(f"{separator_line}\n")
            elif stage.lower() == 'end':
                logging.info(f"{separator_line}")
                logging.info(f"{stage.upper()} OF: {section_name.upper()}")
                logging.info(f"{separator_line}\n\n\n")
            else:
                error_message = "Please define the stage name from one of these options: [start or End]"
                raise CustomerChurnException(error_message, sys)
    except Exception as e:
        raise CustomerChurnException(e, sys)



