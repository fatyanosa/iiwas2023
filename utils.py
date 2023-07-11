import numpy as np
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

def metric_calc(y):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    y[['y_true', 'y_pred']] = scaler.fit_transform(y[['y_true', 'y_pred']])
    y_true = y['y_true']
    y_pred = y['y_pred']

    rmse = RMSE(y_true, y_pred)
    # print("RMSE=", rmse)

    mae = MAE(y_true, y_pred)
    # print("MAE=", mae)

    mape = MAPE(y_true, y_pred)
    # print("MAPE=", mape)
    return rmse, mae, mape