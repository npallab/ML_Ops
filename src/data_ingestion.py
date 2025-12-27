import os
import sys
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# Project specific imports
from config.path_config import RAW_FILE_PATH, TRAIN_FILE_PATH, TEST_FILE_PATH, RAW_DIR, CONFIG_PATH
from src.logger import get_logger
from src.customexception import CustomException
from utils.common_functions import read_yaml_file

# Load environment variables (optional since we hardcoded the JSON path for now)
# load_dotenv()
logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        try:
            self.config = config['data_ingestion']
            self.bucket_name = self.config['bucket_name']
            self.bucket_file_name = self.config['bucket_file_name']
            self.train_test_ratio = self.config['train_ratio']
            
            # Ensure the directory exists (Check spelling of RAW_DIR in path_config.py)
            os.makedirs(RAW_DIR, exist_ok=True)
            logger.info(f"Directory verified/created at: {RAW_DIR}")
        except Exception as e:
            raise CustomException(e, sys)

    def download_data_from_gcs(self):
        try:
            # Explicit path to your JSON key
            key_path = r'C:\Users\user\Desktop\strategic-reef-479708-u4-ee7574828564.json'
            
            if not os.path.exists(key_path):
                raise FileNotFoundError(f"JSON Key not found at {key_path}")

            logger.info(f"Authenticating GCS using key: {key_path}")
            client = storage.Client.from_service_account_json(key_path)
            
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.bucket_file_name)
            
            logger.info(f"Downloading {self.bucket_file_name} from bucket {self.bucket_name}...")
            blob.download_to_filename(RAW_FILE_PATH)
            
            logger.info(f"File downloaded successfully to: {RAW_FILE_PATH}")
        except Exception as e:
            logger.error(f"GCS Download Error: {str(e)}")
            raise CustomException(e, sys)

    def split_data_as_train_test(self):
        try:
            logger.info("Reading raw data for splitting...")
            df = pd.read_csv(RAW_FILE_PATH)
            
            train_set, test_set = train_test_split(
                df, 
                test_size=self.train_test_ratio, 
                random_state=42
            )
            
            train_set.to_csv(TRAIN_FILE_PATH, index=False)
            test_set.to_csv(TEST_FILE_PATH, index=False)
            
            logger.info(f"Split complete. Train: {len(train_set)} rows, Test: {len(test_set)} rows.")
        except Exception as e:
            logger.error(f"Split Error: {str(e)}")
            raise CustomException(e, sys)

    def run(self):
        try:
            self.download_data_from_gcs()
            self.split_data_as_train_test()
            logger.info("Data Ingestion Pipeline finished successfully.")
        except CustomException as ce:
            logger.error(f"Pipeline failed: {ce}")

if __name__ == "__main__":
    # Load config and execute
    config_data = read_yaml_file(CONFIG_PATH)
    data_ingestion = DataIngestion(config_data)
    data_ingestion.run()
    print("Data Ingestion Process Completed. Check your artifacts folder.")