from config.path_config import MODEL_OUTPUT_PATH, PROCESSED_TEST_DATA_PATH, PROCESSED_TRAIN_DATA_PATH
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from config.path_config import TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR
from config.path_config import CONFIG_PATH

if __name__ == "__main__":
    data_processor = DataProcessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=PROCESSED_DIR,
        config_path=CONFIG_PATH
    )
    data_processor.process()

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