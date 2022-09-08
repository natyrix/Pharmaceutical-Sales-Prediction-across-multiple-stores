import pandas as pd
import dvc.api

import datetime
import sys
import os
import pickle
sys.path.append(os.path.abspath(os.path.join("./scripts/")))
from logger import logger

class ReadWriteUtil():
    def dvc_get_data(self, path, version='v1') :
        try:
            repo = "/home/n/Documents/10_Academy/Pharmaceutical-Sales-Prediction-across-multiple-stores"
            data_url = dvc.api.get_url(path=path, repo=repo, rev=version)
            data_url = str(data_url)[6:]
            df = pd.read_csv(data_url, sep=",", low_memory=False)
            logger.info(f"{path} with version {version} Loaded")
        except Exception as e:
            df = None
            logger.error(e)
        return df
    
    def to_csv(self, df, csv_path, index=False):
        try:
            df.to_csv(csv_path, index=index)
            logger.info(f"Saved to path {csv_path}")

        except Exception as e:
            logger.error(e)

    def save_model(self, model):
        try:
            model_path = '../models/' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '.pkl'
            pickle.dump(model, open(model_path, 'wb'))
            logger.info("Model saved to ../models/ folder")
        except Exception as e:
            logger.error(e)

# if __name__ == "__main__":
#     print(dvc_get_data("../data/store.csv", "v1").shape)
#     print(dvc_get_data("../data/test.csv", "v1").shape)
#     print(dvc_get_data("../data/train.csv", "v1").shape)