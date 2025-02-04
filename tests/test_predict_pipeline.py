import unittest
import pandas as pd
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

class TestPredictPipeline(unittest.TestCase):
    def test_predict_pipeline(self):
        data = CustomData(
            gender="male",
            race_ethnicity="group A",
            parental_level_of_education="bachelor's degree",
            lunch="standard",
            test_preparation_course="none",
            reading_score=70,
            writing_score=80
        )
        df = data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)
        self.assertIsNotNone(prediction)

if __name__ == "__main__":
    unittest.main()