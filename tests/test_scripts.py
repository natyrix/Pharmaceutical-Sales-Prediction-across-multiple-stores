import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../Pharmaceutical-Sales-Prediction-across-multiple-stores/")))

from scripts import cleaning_pipeline
df = pd.read_csv("./tests/train_sample_data.csv")

class TestDataClean(unittest.TestCase):
    def setUp(self) -> None:
        # self.df = df_g.copy()
        self.cleaner = cleaning_pipeline.CleaningPipeline()

    def test_convert_to_datetime(self):
        self.cleaner.convert_to_datetime(df, ['Date'])
        self.assertEqual(df['Date'].dtype, "datetime64[ns]")
    
    def test_convert_to_string(self):
        self.cleaner.convert_to_string(df, ['StateHoliday'])
        self.assertEqual(df['StateHoliday'].dtype, "string")


if __name__ == "__main__":
    unittest.main()