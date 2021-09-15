FROM python:3.8-slim

# set work directory
WORKDIR /app

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
# update command linux
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && pip install -r requirements.txt && pip install python-multipart

# copy project
COPY . .