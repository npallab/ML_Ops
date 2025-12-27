import numpy 
import pandas as pd
import os
import yaml
from src.logger import get_logger
from src.customexception import CustomException
import sys

logger=get_logger(__name__)

def read_yaml_file(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The specified YAML file does not exist")
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info(f"YAML file {file_path} loaded successfully")
            return config
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise CustomException(e)
    