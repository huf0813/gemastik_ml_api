from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File
import csv
from io import BytesIO
import numpy as np
import tensorflow as tf
from datetime import date, timedelta
import pandas as pd
from PIL import Image, ImageOps

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post('/commodity_image/predict')
async def commodity_image(commodity_image: bytes = File(...)):
    np.set_printoptions(suppress=True)

    model = tf.keras.models.load_model('models/keras_model.h5')

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    image = Image.open(BytesIO(commodity_image))

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)

    pr = np.argmax(prediction, axis=1)

    if pr[0] == 0:
        result = {"commodity_name": "kedelai", "deskripsi": "Kedelai dengan jenis Edamame, yang sekarang sedang naik daun. Edamame terbukti mengandung isoflavon tertinggi dibandingkan jenis kedelai lain. Kandungan protein edamame mencapai 36%, jauh lebih tinggi dibanding olahan kedelai lain.Edamame sangat ideal untuk Anda yang ingin mencari camilan rendah lemak, tetapi tinggi protein."}
    elif pr[0] == 1:
        result = {"commodity_name": "jagung", "deskripsi": "Jagung dengan jenis Jagung gigi kuda (dent corn). memiliki kandungan pati yang lebih tinggi daripada jagung manis, namun kadar gulanya justru lebih rendah. Jagung gigi kuda memiliki dua jenis warna, yaitu kuning dan putih. Biasanya jagung berwarna kuning digunakan sebagai pakan ternak. Sementara jagung berwarna putih dimanfaatkan untuk membuat roti ataupun tepung jagung. Di Amerika, jenis jagung ini digunakan untuk membuat keripik tortila."}
    else:
        result = {"commodity_name": "kopi", "deskripsi": "Kopi dengan jenis Papua Wamena. Kopi ini merupakan Jenis kopi arabika yang ditanam di Lembah Baliem pegunungan Jayawijaya. Kopi Arabika ini memiliki cita rasa yang sangat khas di banding dengan cita rasa Arabika lainnya, aroma kopinya harum halus dan memiliki after taste yang sangat manis. Kopi Arabika Papua Wamena juga memiliki kadar asam yang rendah sehingga bisa dikonsumsi oleh semua orang. Memiliki kadar air 12 persen dan difermentasi selama 8 hingga 10 jam."}

    return success_response(result)


# Machine Learning
def load_model(name):
    path = 'models/{}_model.h5'.format(name)
    model = tf.keras.models.load_model(path)
    return model


def price_prediction(day_future: int, model, commodity_series):
    split_time = 2500
    x_valid = np.array(commodity_series[split_time:])
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
