import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join("../Pharmaceutical-Sales-Prediction-across-multiple-stores/")))

from scripts import cleaning_pipeline

class TestDataClean(unittest.TestCase):
    def setUp(self) -> None:
        # self.df = df_g.copy()
        self.cleaner = cleaning_pipeline.CleaningPipeline()