import unittest
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def test_model_trainer(self):
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)
        trainer = ModelTrainer()
        r2_score = trainer.initiate_model_trainer(train_arr, test_arr)
        self.assertGreater(r2_score, 0.6)

if __name__ == "__main__":
    unittest.main()