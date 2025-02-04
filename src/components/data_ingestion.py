import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from dataclasses import dataclass
import os

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('data', 'raw', 'data.csv') 
    train_data_path: str = os.path.join('data', 'processed', 'train.csv')  
    test_data_path: str = os.path.join('data', 'processed', 'test.csv')  # Path to save the test data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Load the raw data
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info('Read the dataset as a DataFrame')

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)

            # Split the data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train-test split completed")

            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)