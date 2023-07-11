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

datasets = {
    "ambient_temperature_system_failure": [
        [
            "2013-12-15 07:00:00.000000",
            "2013-12-30 09:00:00.000000"
        ],
        [
            "2014-03-29 15:00:00.000000",
            "2014-04-20 22:00:00.000000"
        ]
    ],
    "cpu_utilization_asg_misconfiguration": [
        [
            "2014-07-10 12:29:00.000000",
            "2014-07-15 17:19:00.000000"
        ]
    ],
    "ec2_request_latency_system_failure": [
        [
            "2014-03-14 03:31:00.000000",
            "2014-03-14 14:41:00.000000"
        ],
        [
            "2014-03-18 17:06:00.000000",
            "2014-03-19 04:16:00.000000"
        ],
        [
            "2014-03-20 21:26:00.000000",
            "2014-03-21 03:41:00.000000"
        ]
    ],
    "machine_temperature_system_failure": [
        [
            "2013-12-10 06:25:00.000000",
            "2013-12-12 05:35:00.000000"
        ],
        [
            "2013-12-15 17:50:00.000000",
            "2013-12-17 17:00:00.000000"
        ],
        [
            "2014-01-27 14:20:00.000000",
            "2014-01-29 13:30:00.000000"
        ],
        [
            "2014-02-07 14:55:00.000000",
            "2014-02-09 14:05:00.000000"
        ]
    ],
    "nyc_taxi": [
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
    ],
    "rogue_agent_key_hold": [
        [
            "2014-07-15 04:35:00.000000",
            "2014-07-15 13:25:00.000000"
        ],
        [
            "2014-07-17 05:50:00.000000",
            "2014-07-18 06:45:00.000000"
        ]
    ],
    "rogue_agent_key_updown": [
        [
            "2014-07-14 17:00:00.000000",
            "2014-07-15 15:00:00.000000"
        ],
        [
            "2014-07-16 21:50:00.000000",
            "2014-07-17 19:50:00.000000"
        ]
    ],
}

for data, anomaly_points in datasets.items():
    print(data)
    df = pd.read_csv("data/"+data+".csv",low_memory=False)

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
    print(f'One-Class SVM F1 Score (Before) : {ocsvm_f1}')

    best_error = 1
    best_mask = 0.1

    def impute(df, i):
        miss_data = df.copy()
        miss_data.loc[df.sample(frac=i, random_state=42).index, "value"] = np.nan

        return pd.DataFrame(miss_data["value"]).interpolate(
            method="linear", limit_direction="both"
        )

    # for i in np.arange(0.1, 1, 0.1):
    #     X_filled = impute(df, i)
    #     y = pd.DataFrame({'y_true':df['value'],
    #     'y_pred':X_filled['value']})
    #     rmse, mae, mape = metric_calc(y)
    #     print(i)
    #     print(rmse, mae, mape)
    #     if rmse < best_error:
    #         best_error = rmse
    #         best_mask = i

    #     ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
    #     ocsvm_ret = ocsvm_model.fit_predict(X_filled['value'].values.reshape(-1, 1))
    #     ocsvm_df = pd.DataFrame()
    #     ocsvm_df['value'] = X_filled['value']
    #     ocsvm_df['anomaly']  = [1 if i==-1 else 0 for i in ocsvm_ret]
    #     ocsvm_f1 = f1_score(df['anomaly'], ocsvm_df['anomaly'])
    #     print(f'One-Class SVM F1 Score : {ocsvm_f1}')

    # print(best_mask)
    X_filled = impute(df, 0.4)
    ocsvm_model = OneClassSVM(nu=0.2, gamma=0.001, kernel='rbf')
    ocsvm_ret = ocsvm_model.fit_predict(X_filled['value'].values.reshape(-1, 1))
    ocsvm_df = pd.DataFrame()
    ocsvm_df['value'] = X_filled['value']
    ocsvm_df['anomaly']  = [1 if i==-1 else 0 for i in ocsvm_ret]
    ocsvm_f1 = f1_score(df['anomaly'], ocsvm_df['anomaly'])
    print(f'One-Class SVM F1 Score : {ocsvm_f1}')
