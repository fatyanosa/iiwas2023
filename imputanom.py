import pandas as pd
import scipy.stats as stats
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score
from missingpy import MissForest
import sys
import sklearn.neighbors._base

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np
from utils import metric_calc


df = pd.read_csv("data/nyc_taxi.csv",low_memory=False)


anomaly_points = [
    [
        "2014-10-30 15:30:00.000000",
        "2014-11-03 22:30:00.000000"
    ],
    [
        "2014-11-25 12:00:00.000000",
        "2014-11-29 19:00:00.000000"
    ],
    [
        "2014-12-23 11:30:00.000000",
        "2014-12-27 18:30:00.000000"
    ],
    [
        "2014-12-29 21:30:00.000000",
        "2015-01-03 04:30:00.000000"
    ],
    [
        "2015-01-24 20:30:00.000000",
        "2015-01-29 03:30:00.000000"
    ]  
]


df['timestamp'] = pd.to_datetime(df['timestamp'])
#is anomaly? : True => 1, False => 0
df['anomaly'] = 0
for start, end in anomaly_points:
    df.loc[((df['timestamp'] >= start) & (df['timestamp'] <= end)), 'anomaly'] = 1

df.index = df['timestamp']
df.drop(['timestamp'], axis=1, inplace=True)


ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
ocsvm_ret = ocsvm_model.fit_predict(df['value'].values.reshape(-1, 1))
ocsvm_df = pd.DataFrame()
ocsvm_df['value'] = df['value']
ocsvm_df['anomaly']  = [1 if i==-1 else 0 for i in ocsvm_ret]
ocsvm_f1 = f1_score(df['anomaly'], ocsvm_df['anomaly'])
print(f'One-Class SVM F1 Score : {ocsvm_f1}')

rng = np.random.RandomState(42)
def impute(df, i):
    miss_data = df.copy()
    miss_data = miss_data.mask(rng.random(df.shape) < i)
    miss_data.loc[miss_data["anomaly"] == 1]

    return pd.DataFrame(miss_data.copy()).interpolate(
                method="linear", limit_direction="both"
            )

for i in np.arange(0.1, 1, 0.1):
    X_filled = impute(df, i)
    y = pd.DataFrame({'y_true':df['value'],
    'y_pred':X_filled['value']})
    rmse, mae, mape = metric_calc(y)

    print(i)
    print("rmse",rmse)

    ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
    ocsvm_ret = ocsvm_model.fit_predict(X_filled['value'].values.reshape(-1, 1))
    ocsvm_df = pd.DataFrame()
    ocsvm_df['value'] = X_filled['value']
    ocsvm_df['anomaly']  = [1 if i==-1 else 0 for i in ocsvm_ret]
    ocsvm_f1 = f1_score(df['anomaly'], ocsvm_df['anomaly'])
    print(f'One-Class SVM F1 Score : {ocsvm_f1}')
