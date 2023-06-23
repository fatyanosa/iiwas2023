import os
import time
import warnings
import pathlib
import glob
from utils import remove_date, metric_calc
import pandas as pd
from collections import defaultdict
from missingpy import MissForest
import numpy as np
from sklearn.model_selection import train_test_split

np.seterr(invalid="ignore")
warnings.filterwarnings("ignore")

datasets = {
        "ozone": ["onehr","eighthr"],
        "pm2.5": ["Beijing", "Chengdu", "Guangzhou", "Shanghai", "Shenyang"],
}

for folder, files in datasets.items():
    for data in files:
        df = pd.read_csv("data/" + folder + "/" + data + ".csv")
        print(df.shape)
        # print(df.isnull().any().any())

        df = remove_date(df)

        # Fill the missing values first with imputation
        imp = MissForest()
        df = imp.fit_transform(df)

        # create a test data with missing last row to predict
        test = df.copy()

        test.iloc[-1:, :] = np.nan
        print(test.iloc[-1:, :])

        # predict the last row using a prediction method
        imp = MissForest()
        test = imp.fit_transform(test)

        rmse, mae, mape = metric_calc(test, df)





