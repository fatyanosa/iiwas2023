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
np.random.seed(1234)
from utils import metric_calc


df = pd.read_csv("data/cpu_utilization_asg_misconfiguration.csv",low_memory=False)


anomaly_points = [
    [
            "2014-07-10 12:29:00.000000",
            "2014-07-15 17:19:00.000000"
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
best_error = 1
best_mask = 0.1

def impute(df, i):
    miss_data = df.copy()
    miss_data = miss_data.mask(rng.random(df.shape) < i)
    miss_data.loc[miss_data["anomaly"] == 1]

    return pd.DataFrame(miss_data.copy()).interpolate(
                method="linear", limit_direction="both"
            )

for i in np.arange(0.1, 1, 0.1):
    y = pd.DataFrame({'y_true':df['value'],
    'y_pred':impute(df, i)['value']})
    rmse, mae, mape = metric_calc(y)

    if rmse < best_error:
        best_error = rmse
        best_mask = i
        print(i)
        print(rmse, mae, mape)

X_filled = impute(df, best_mask)
ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
ocsvm_ret = ocsvm_model.fit_predict(X_filled['value'].values.reshape(-1, 1))
ocsvm_df = pd.DataFrame()
ocsvm_df['value'] = X_filled['value']
ocsvm_df['anomaly']  = [1 if i==-1 else 0 for i in ocsvm_ret]
ocsvm_f1 = f1_score(df['anomaly'], ocsvm_df['anomaly'])
print(f'One-Class SVM F1 Score : {ocsvm_f1}')
