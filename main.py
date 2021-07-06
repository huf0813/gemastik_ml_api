from fastapi import FastAPI
import csv
import numpy as np
import tensorflow as tf

app = FastAPI()


def success_response(data):
    return {"success": True,
            "message": "success",
            "data": data}


def commodity_dataset(path):
    series = []

    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)
        step = 0
        for row in reader:
            price_close = row[5]
            if price_close == "null":
                continue
            series.append(float(price_close))
            step = step + 1

    return series


def commodity_predict(days, model, name):
    path = 'dataset/{}.csv'.format(name)
    series = commodity_dataset(path)

    result = price_prediction(days, model, series)
    return result


@app.get("/commodity")
def commodity(name: str, days: int):
    dataset_path = "dataset/{}.csv".format(name)
    dataset = commodity_dataset(dataset_path)
    return success_response(dataset[-days:])


@app.get("/commodity/predict")
def commodity_future_price(name: str, days: int = 0):
    model = load_model(name)
    result = commodity_predict(days, model, name)
    return success_response(result)


# Machine Learning
def load_model(name):
    path = 'models/{}_model.h5'.format(name)
    model = tf.keras.models.load_model(path)
    return model


def price_prediction(days: int, model, commodity_series):
    split_time = 2500
    x_valid = commodity_series[split_time:]

    temp = len(x_valid) - days
    temp = x_valid[temp:]

    result = []

    # for i in range(days):
    #     rnn_forecast = model_forecast(model, temp[..., np.newaxis], len(temp))
    #     index_value = rnn_forecast[0][-1][0]
    #     result.append(index_value)
    #
    #     # delete first value
    #     temp = np.delete(temp, 0)
    #
    #     # add last value
    #     temp = np.append(temp, index_value)

    rnn_forecast = model_forecast(model, temp[..., np.newaxis], len(temp))
    index_value = rnn_forecast[0][-1][0]
    result.append(index_value)

    # delete first value
    temp = np.delete(temp, 0)

    # add last value
    np.append(temp, index_value)
    return result


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


if __name__ == "__main__":
    name = "sugar"
    days = 30
    model = load_model(name)
    result = commodity_predict(days, model, name)