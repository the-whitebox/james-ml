FROM python:3.10-slim-buster
RUN echo 'deb http://deb.debian.org/debian testing main' >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y --no-install-recommends -o APT::Immediate-Configure=false gcc g++
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
EXPOSE 5005
ENTRYPOINT ["python3"]

CMD ["train_yolo.py"]