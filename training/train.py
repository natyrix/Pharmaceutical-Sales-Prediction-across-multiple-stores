import sys
from urllib.parse import urlparse
# import mlflow
from matplotlib import pyplot as plt
from regex import P
# import mlflow
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd
import seaborn as sns
import sys
import warnings
import os

data = pd.read_csv('data/train_merged.csv', sep=',')

# mlflow.set_experiment('Sales Prediction')

if __name__ == '__main__':
    # with mlflow.start_run():
    warnings.filterwarnings("ignore")
    
    data.set_index('Year', inplace=True)

    data.drop(['StateHoliday'], axis=1, inplace=True)

    data = data[data['Open'] == 1]
    data = data[data['Sales'] > 0.0]

    X = data.drop('Sales', axis=1)
    y = data['Sales']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)

    data_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', data_transformer), ('regressor',
                                            RandomForestRegressor(n_estimators=12, random_state=42))
    ])

    rf_model = pipeline.fit(X_train, y_train)
    val_accuracy = rf_model.score(X_test, y_test)
    with open("train/metrics.txt", 'w') as outfile:
        outfile.write(
            f"Validation data accuracy: {val_accuracy}")