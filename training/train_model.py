import sys
import mlflow.sklearn
import mlflow
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import sys
import warnings
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from ml_utils import cross_validation, plot_result

data = pd.read_csv('data/train_processed.csv', sep=',')

data.set_index('Date', inplace=True)

data.drop(['StateHoliday'], axis=1, inplace=True)

data = data[data['Open'] == 1]
data = data[data['Sales'] > 0.0]

X = data.drop(columns=['Sales','Customers'], axis=1)
y = data['Sales']
"""
RandomForestClassifier
"""
model = RandomForestClassifier(max_depth=20)

model_result = cross_validation(model, X, y, 5)


with open("training/random_forest_classifier_result.txt", 'w') as outfile:
    outfile.write(
        f"Training data accuracy: {model_result['Training Accuracy scores'][0]}\n")
    outfile.write(
        f"Validation data accuracy: {model_result['Validation Accuracy scores'][0]}")


model_name = "Random Forest Classifier"
plot_result(model_name, "Accuracy", "Accuracy scores in 5 Folds",
               model_result["Training Accuracy scores"],
               model_result["Validation Accuracy scores"],
               'training/random_forest_classifier_accuracy.png')

# Precision Results

# Plot Precision Result
plot_result(model_name, "Precision", "Precision scores in 5 Folds",
               model_result["Training Precision scores"],
               model_result["Validation Precision scores"],
               'training/random_forest_classifier_preicision.png')

# Recall Results plot

# Plot Recall Result
plot_result(model_name, "Recall", "Recall scores in 5 Folds",
               model_result["Training Recall scores"],
               model_result["Validation Recall scores"],
               'training/random_forest_classifier_recall.png')


# f1 Score Results

# Plot F1-Score Result
plot_result(model_name, "F1", "F1 Scores in 5 Folds",
               model_result["Training F1 scores"],
               model_result["Validation F1 scores"],
               'training/random_forest_classifier_f1_score.png')



