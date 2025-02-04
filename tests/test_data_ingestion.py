import unittest
from src.components.data_ingestion import DataIngestion

class TestDataIngestion(unittest.TestCase):
    def test_data_ingestion(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        self.assertIsNotNone(train_path)
        self.assertIsNotNone(test_path)

if __name__ == "__main__":
    unittest.main()