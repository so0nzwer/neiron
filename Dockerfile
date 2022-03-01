FROM python:3.7-slim as builder

ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    apt-get nvidia-cuda-toolkit

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "main.py" ]