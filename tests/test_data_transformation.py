import unittest
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

class TestDataTransformation(unittest.TestCase):
    def test_data_transformation(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
        self.assertIsNotNone(train_arr)
        self.assertIsNotNone(test_arr)

if __name__ == "__main__":
    unittest.main()