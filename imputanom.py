import pandas as pd
import scipy.stats as stats
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from missingpy import MissForest
import sys
import sklearn.neighbors._base

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
from sklearn.impute import SimpleImputer, KNNImputer
import numpy as np

np.random.seed(1234)
from src.utils import select_method, metric_calc, imputation
import time
import os
import pathlib

datasets = {
    "nyc_taxi": [
        ["2014-10-30 15:30:00.000000", "2014-11-03 22:30:00.000000"],
        ["2014-11-25 12:00:00.000000", "2014-11-29 19:00:00.000000"],
        ["2014-12-23 11:30:00.000000", "2014-12-27 18:30:00.000000"],
        ["2014-12-29 21:30:00.000000", "2015-01-03 04:30:00.000000"],
        ["2015-01-24 20:30:00.000000", "2015-01-29 03:30:00.000000"],
    ],
    "ambient_temperature_system_failure": [
        ["2013-12-15 07:00:00.000000", "2013-12-30 09:00:00.000000"],
        ["2014-03-29 15:00:00.000000", "2014-04-20 22:00:00.000000"],
    ],
    "cpu_utilization_asg_misconfiguration": [
        ["2014-07-10 12:29:00.000000", "2014-07-15 17:19:00.000000"]
    ],
    "ec2_request_latency_system_failure": [
        ["2014-03-14 03:31:00.000000", "2014-03-14 14:41:00.000000"],
        ["2014-03-18 17:06:00.000000", "2014-03-19 04:16:00.000000"],
        ["2014-03-20 21:26:00.000000", "2014-03-21 03:41:00.000000"],
    ],
    "machine_temperature_system_failure": [
        ["2013-12-10 06:25:00.000000", "2013-12-12 05:35:00.000000"],
        ["2013-12-15 17:50:00.000000", "2013-12-17 17:00:00.000000"],
        ["2014-01-27 14:20:00.000000", "2014-01-29 13:30:00.000000"],
        ["2014-02-07 14:55:00.000000", "2014-02-09 14:05:00.000000"],
    ],
    "rogue_agent_key_hold": [
        ["2014-07-15 04:35:00.000000", "2014-07-15 13:25:00.000000"],
        ["2014-07-17 05:50:00.000000", "2014-07-18 06:45:00.000000"],
    ],
    "rogue_agent_key_updown": [
        ["2014-07-14 17:00:00.000000", "2014-07-15 15:00:00.000000"],
        ["2014-07-16 21:50:00.000000", "2014-07-17 19:50:00.000000"],
    ],
}

ad_methods = [
    "Hotelling T2",
    "One-Class SVM",
    "Isolation Forest",
    "LOF",
    "ChangeFinder",
    "Variance Based Method",
]

imp_methods = [
    "MEAN",
    "MODE",
    "MEDIAN",
    "INTERPOLATION",
    "GAIN",
    "MissForest",
    "KNN",
    "miceforest",
]

output_file = "results.csv"
pathlib.Path(output_file).touch(exist_ok=True)

if os.stat(output_file).st_size == 0:
    f_results = open(output_file, "a")
    f_results.write(
        "data,fraction,imp_method,ad_method,f1_before,f1_after,recall_before,recall_after,precision_before,precision_after,accuracy_before,accuracy_after,rmse,mae,mape,exec_time"
        + "\n"
    )
    f_results.close()

for data, anomaly_points in datasets.items():
    print(data)
    print("Data= ", data)
    df = pd.read_csv("data/" + data + ".csv", low_memory=False)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # is anomaly? : True => 1, False => 0
    df["anomaly"] = 0
    for start, end in anomaly_points:
        df.loc[((df["timestamp"] >= start) & (df["timestamp"] <= end)), "anomaly"] = 1

    df.index = df["timestamp"]
    df.drop(["timestamp"], axis=1, inplace=True)

    for fraction in np.arange(0.1, 1, 0.1):
        print("Fraction= ", fraction)
        miss_data = df.copy()
        miss_data.loc[df.sample(frac=fraction, random_state=42).index, "value"] = np.nan
        for imp in imp_methods:
            try:
                X_filled = imputation(imp, miss_data).ravel()
                y = pd.DataFrame({"y_true": df["value"], "y_pred": X_filled})
                rmse, mae, mape = metric_calc(y)
            except ValueError as e:
                print(e)
                continue

            for method in ad_methods:
                print("method=", method)
                start_time = time.time()
                (
                    f1_before,
                    recall_before,
                    precision_before,
                    accuracy_before,
                ) = select_method(method, df)
                f1_after, recall_after, precision_after, accuracy_after = select_method(
                    method, df, X_filled
                )

                end_time = time.time()
                exec_time = round(end_time - start_time)

                new_row = {
                    "data": data,
                    "fraction": fraction,
                    "imp_method": imp,
                    "ad_method": method,
                    "f1_before": f1_before,
                    "f1_after": f1_after,
                    "recall_before": recall_before,
                    "recall_after": recall_after,
                    "precision_before": precision_before,
                    "precision_after": precision_after,
                    "accuracy_before": accuracy_before,
                    "accuracy_after": accuracy_after,
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "exec_time": exec_time,
                }
                df_metrics = pd.read_csv(output_file)

                df_metrics.loc[len(df_metrics)] = new_row

                df_metrics.to_csv(output_file, index=False)
