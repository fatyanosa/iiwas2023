import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def remove_date(df):
    cat_columns = df.select_dtypes(["object", "category", "<M8[ns]"]).columns

    for col in cat_columns:
        # check if the object type is datetime
        try:
            df[col] = pd.to_datetime(df[col])

            # Drop date and time column
            df.drop([col], axis=1, inplace=True)

        except Exception:
            df[col] = df[col].replace(np.nan, "", regex=True)

            # remove unnecessary space in categorical df
            df[col] = df[col].map(str.strip)

            # replace empty string with Nan
            df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
            df[col] = df[col].astype("category")

    cat_columns = df.select_dtypes(["category"]).columns
    if not cat_columns.empty:
        df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
        df[cat_columns] = df[cat_columns].astype("category")
        df[cat_columns] = df[cat_columns].replace([-1], np.nan)
    return df

def RMSE(original, filled):
    from sklearn.metrics import mean_squared_error

    score = np.sqrt(mean_squared_error(original, filled))
    return score


def MAE(original, filled):
    from sklearn.metrics import mean_absolute_error

    score = mean_absolute_error(original, filled)
    return score


def MAPE(original, filled):
    from sklearn.metrics import mean_absolute_percentage_error

    score = mean_absolute_percentage_error(original, filled)
    return score

def metric_calc(X_filled, complete_data):
    scaler = MinMaxScaler()
    scaled_complete_data = scaler.fit_transform(complete_data)
    scaled_X_filled = scaler.fit_transform(X_filled)

    rmse = RMSE(scaled_complete_data, scaled_X_filled)
    print("RMSE=", rmse)

    mae = MAE(scaled_complete_data, scaled_X_filled)
    print("MAE=", mae)

    mape = MAPE(scaled_complete_data, scaled_X_filled)
    print("MAPE=", mape)

    return rmse, mae, mape