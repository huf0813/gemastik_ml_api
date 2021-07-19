import datetime

import numpy
from fastapi import FastAPI
import csv
import numpy as np
import tensorflow as tf
from datetime import date, timedelta
import pandas as pd

app = FastAPI()


def success_response(data):
    print(type(data))
    return {"success": True,
            "message": "success",
            "data": data}


def commodity_dataset(path):
    series = []
    response_series = []

    with open(path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        next(reader)
        step = 0
        for row in reader:
            price_close = row[5]
            if price_close == "null":
                continue
            price = float(price_close)
            series.append(price)

            origin = row[0]
            range_day = len(pd.date_range(start=origin, end=date.today().strftime("%Y-%m-%d")))
            obj = {
                "time": origin,
                "value": price
            }
            response_series.append(obj)
            step = step + 1

    return series, response_series


def commodity_predict(day, model, commodity_name):
    path = 'dataset/{}.csv'.format(commodity_name)
    series, _ = commodity_dataset(path)
    commodity_result = price_prediction(day, model, series)
    return commodity_result


@app.get("/commodity")
def commodity(commodity_name: str, day: int):
    dataset_path = "dataset/{}.csv".format(commodity_name)
    _, dataset = commodity_dataset(dataset_path)
    return success_response(dataset[-day:])


@app.get("/commodity/predict")
def commodity_future_price(commodity_name: str, day: int = 0):
    model = load_model(commodity_name)
    commodity_result = commodity_predict(day, model, commodity_name)
    return success_response(commodity_result)


# Machine Learning
def load_model(name):
    path = 'models/{}_model.h5'.format(name)
    model = tf.keras.models.load_model(path)
    return model


def price_prediction(day_future: int, model, commodity_series):
    split_time = 2500
    x_valid = numpy.array(commodity_series[split_time:])
    scalar_days = len(x_valid) - day_future
    temp = x_valid[scalar_days:]

    final_predict = []

    for i in range(day_future):
        rnn_forecast = model_forecast(model, temp[..., np.newaxis], len(temp))
        index_value = float(rnn_forecast[0][-1][0])

        today = date.today() + timedelta(days=day_future + i - 1)
        obj = {
            "time": today.strftime("%Y-%m-%d"),
            "value": index_value
        }

        final_predict.append(obj)

        # delete first value
        temp = np.delete(temp, 0)
        # add last value
        temp = np.append(temp, 1)

    return final_predict


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
