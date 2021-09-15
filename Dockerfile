FROM python:3.8.10

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# update command linux
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install python-multipart

# copy project
COPY . .