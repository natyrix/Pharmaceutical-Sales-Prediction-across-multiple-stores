import pandas as pd
import dvc.api


import sys
import os

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


# if __name__ == "__main__":
#     print(dvc_get_data("../data/store.csv", "v1").shape)
#     print(dvc_get_data("../data/test.csv", "v1").shape)
#     print(dvc_get_data("../data/train.csv", "v1").shape)