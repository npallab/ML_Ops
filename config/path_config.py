import os

###############################DATA IGESTION PATHS##################################

RAW_DIR ='artifacts/raw_data'
RAW_FILE_PATH = os.path.join(RAW_DIR,'RAW.csv')
TRAIN_FILE_PATH = os.path.join(RAW_DIR,'TRAIN.csv')
TEST_FILE_PATH = os.path.join(RAW_DIR,'TEST.csv')

CONFIG_PATH='config/config.yaml'

###############################DATA PROCESSING PATHS##################################

PROCESSED_DIR = 'artifacts/processed_data'
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, 'PROCESSED_TRAIN.csv')
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, 'PROCESSED_TEST.csv')

