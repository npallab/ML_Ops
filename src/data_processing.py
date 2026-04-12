import os 
import sys
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.customexception import CustomException
from config.path_config import *
from utils.common_functions import read_yaml_file, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE # Fixed typo: imblearn

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        try:
            self.train_path = train_path
            self.test_path = test_path
            self.processed_dir = processed_dir
            self.config_path = config_path

            self.config = read_yaml_file(self.config_path)
            
            if not os.path.exists(self.processed_dir):
                os.makedirs(self.processed_dir)
                logger.info(f"Created directory for processed data at {self.processed_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_data(self, df):
        try:
            logger.info('Starting data preprocessing...')
            # Drop unnecessary columns
            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True, errors='ignore')
            df.drop_duplicates(inplace=True)
            
            cat_columns = self.config['data_processing']['categorical_columns']
            num_columns = self.config['data_processing']['numerical_columns']
            
            logger.info('Applying label encoding to categorical columns...')
            encoder = LabelEncoder()
            mappings = {}
            for col in cat_columns:
                if col in df.columns:
                    df[col] = encoder.fit_transform(df[col])
                    mappings[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            
            logger.info(f'Label Mappings: {mappings}')
            
            logger.info('Skewness Handling for numerical columns...')
            for col in num_columns:
                if col in df.columns:
                    if df[col].skew() > 0.5:
                        df[col] = np.log1p(df[col])
            
            logger.info('Data preprocessing completed successfully.')
            return df
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise CustomException(e, sys)

    def balance_data(self, df):
        try:
            logger.info('Starting data balancing using SMOTE...') # Fixed: lowercase logger
            X = df.drop(columns=['booking_status'], errors='ignore')
            y = df['booking_status']
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df['booking_status'] = y_resampled
            
            logger.info('Data balancing completed successfully.')
            return balanced_df
        except Exception as e:
            logger.error(f"Error during data balancing: {e}")
            raise CustomException(e, sys)

    def feature_selection(self, balanced_df):
        try:
            logger.info('Starting feature selection using Random Forest...')
            X = balanced_df.drop(columns=['booking_status'], errors='ignore')
            y = balanced_df['booking_status']
            
            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)
            
            feature_importances = model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
            feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
            
            selected_features = feature_importance_df.head(10)['Feature'].tolist()
            logger.info(f'Selected features: {selected_features}')
            
            return selected_features # Return the LIST of names to apply consistently
        except Exception as e:
            logger.error(f"Error during feature selection: {e}")
            raise CustomException(e, sys)

    def save_processed_data(self, df, file_path):
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise CustomException(e, sys)

    def process(self):
        try:
            # 1. Load Data
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # 2. Preprocess
            processed_train_df = self.preprocess_data(train_df)
            processed_test_df = self.preprocess_data(test_df)

            # 3. Balance Training Data (Avoid balancing test data to keep it realistic)
            balanced_train_df = self.balance_data(processed_train_df)

            # 4. Feature Selection (Determine features using TRAIN only)
            top_features = self.feature_selection(balanced_train_df)
            
            # 5. Filter both DataFrames based on selected features
            final_train_df = balanced_train_df[top_features + ['booking_status']]
            final_test_df = processed_test_df[top_features + ['booking_status']]

            # 6. Save Files
            self.save_processed_data(final_train_df, os.path.join(self.processed_dir, 'processed_train.csv'))
            self.save_processed_data(final_test_df, os.path.join(self.processed_dir, 'processed_test.csv'))
            
            logger.info("Entire pipeline completed successfully.")

        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            raise CustomException(e, sys)      

if __name__ == "__main__":
    # Ensure these variable names match exactly what is in your path_config.py
    data_processor = DataProcessor(
        train_path=TRAIN_FILE_PATH, 
        test_path=TEST_FILE_PATH, 
        processed_dir=PROCESSED_DIR, 
        config_path=CONFIG_PATH
    )
    data_processor.process()