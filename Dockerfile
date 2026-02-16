FROM python:3.10-slim

RUN apt update &&  apt upgrade -y

RUN pip install opencv-contrib-python && \
    apt-get install -y ffmpeg libsm6 libxext6  

WORKDIR /app

CMD ["python3", "main.py"]