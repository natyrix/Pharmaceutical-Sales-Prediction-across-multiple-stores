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
from time import gmtime, strftime
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import warnings
import os
from time import gmtime, strftime
import pickle

data = pd.read_csv('data/train_processed.csv', sep=',')


if __name__ == '__main__':
    
    data.set_index('Date', inplace=True)

    data.drop(['StateHoliday'], axis=1, inplace=True)

    data = data[data['Open'] == 1]
    data = data[data['Customers'] > 0.0]

    X = data.drop(columns=['Sales','Customers'], axis=1)
    y = data['Customers']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)

    data_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', data_transformer), ('regressor',
                                            RandomForestRegressor(n_estimators=12, random_state=42))
    ])

    model = pipeline.fit(X_train, y_train)
    
    time = strftime("%Y-%m-%d-%H:%M:%S", gmtime())
    with open(f'../models/random_forest_customers_{time}.pkl', 'wb') as f:
        pickle.dump(model, f)