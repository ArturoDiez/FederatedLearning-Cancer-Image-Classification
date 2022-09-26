# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster

WORKDIR /app
ADD client.py /
COPY requirements.txt requirements.txt
COPY federatedlearninginsa-0caa7695b46b.json federatedlearninginsa-0caa7695b46b.json
RUN pip3 install -r requirements.txt

COPY . .

ENTRYPOINT [ "python3", "./client.py"]