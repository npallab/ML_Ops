import numpy as np
import pandas as pd
import os
import joblib
import lightgbm as lgb
import sys
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from src.logger import get_logger
from src.customexception import CustomException
from config.model_params import LIGHTGBM_PARAMS, RANDOM_SEARCH_PARAMS
from config.path_config import MODEL_OUTPUT_PATH, PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH
from utils.common_functions import load_data
import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, train_path, test_path, model_output_path):
        """
        Initializes the ModelTrainer with paths and parameters.
        """
        try:
            self.train_path = train_path
            self.test_path = test_path
            self.model_output_path = model_output_path
            self.param_distributions = LIGHTGBM_PARAMS
            self.random_search_params = RANDOM_SEARCH_PARAMS
            logger.info("ModelTrainer initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def load_and_split_data(self):
        """
        Loads train and test datasets and splits them into X and y.
        """
        try:
            logger.info(f'Loading training data from {self.train_path}')
            train_df = load_data(self.train_path)
            logger.info(f'Loading testing data from {self.test_path}')
            test_df = load_data(self.test_path)
            
            # Splitting Features and Target
            target_column = 'booking_status'
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]
            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]
            
            logger.info('Data loaded and split successfully.')
            return X_train, y_train, X_test, y_test
        except Exception as e:
            raise CustomException(e, sys)

    def train_lgbm(self, X_train, y_train):
        """
        Trains a LightGBM model using RandomizedSearchCV.
        """
        try:
            logger.info('Starting LightGBM model training with RandomizedSearchCV...')
            lgbm = lgb.LGBMClassifier(random_state=42)
            
            # Unpack parameters using ** to handle n_iter, cv, etc.
            random_search = RandomizedSearchCV(
                estimator=lgbm, 
                param_distributions=self.param_distributions, 
                **self.random_search_params
            )
            
            random_search.fit(X_train, y_train)
            logger.info(f'Best parameters found: {random_search.best_params_}')
            
            return random_search.best_estimator_
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates the trained model and returns a dictionary of metrics.
        """
        try:
            logger.info('Evaluating model performance...')
            y_pred = model.predict(X_test)
            
            # Get probabilities for ROC AUC (handling class index robustly)
            classes = list(model.classes_)
            pos_idx = classes.index(1) if 1 in classes else (classes.index('1') if '1' in classes else 0)
            y_proba = model.predict_proba(X_test)[:, pos_idx]

            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Look for the positive label in the classification report
            pos_label = None
            for candidate in [1, '1', '1.0', 1.0]:
                if str(candidate) in report or candidate in report:
                    pos_label = candidate
                    break
            
            if pos_label is not None:
                metrics = {
                    'accuracy': accuracy,
                    'precision': report[str(pos_label) if str(pos_label) in report else pos_label]['precision'],
                    'recall': report[str(pos_label) if str(pos_label) in report else pos_label]['recall'],
                    'f1_score': report[str(pos_label) if str(pos_label) in report else pos_label]['f1-score'],
                    'roc_auc': roc_auc
                }
            else:
                metrics = {'accuracy': accuracy, 'roc_auc': roc_auc}

            logger.info(f'Evaluation metrics: {metrics}')
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    def save_model(self, model):
        """
        Saves the model to the local artifacts folder.
        """
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info(f'Model saved locally to {self.model_output_path}')
        except Exception as e:
            raise CustomException(e, sys)

    def run(self):
        """
        Executes the entire training pipeline and logs to MLflow.
        """
        try:
            # Note: Ensure an MLflow tracking server is running or uses local 'mlruns'
            with mlflow.start_run():
                logger.info("MLflow Run Started")
                
                # 1. Log dataset artifacts
                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')

                # 2. Training and Evaluation
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_model = self.train_lgbm(X_train, y_train)
                evaluation_results = self.evaluate_model(best_model, X_test, y_test)
                
                # 3. Save and Log Model
                self.save_model(best_model)
                mlflow.log_artifact(self.model_output_path)
                
                # 4. Log Hyperparameters and Metrics
                mlflow.log_params(best_model.get_params())
                mlflow.log_metrics(evaluation_results)
                
                logger.info("MLflow logging completed.")
                return evaluation_results

        except Exception as e:
            # Ensure sys is passed to CustomException for traceback details
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        # Initialize trainer with constants from path_config
        trainer = ModelTrainer(
            train_path=PROCESSED_TRAIN_DATA_PATH, 
            test_path=PROCESSED_TEST_DATA_PATH, 
            model_output_path=MODEL_OUTPUT_PATH
        )
        
        # Execute the pipeline
        results = trainer.run()
        print("\n--- Training Pipeline Results ---")
        print(results)

    except Exception as e:
        # Logged by the run method or initialization
        print(f"Pipeline Execution Failed: {e}")