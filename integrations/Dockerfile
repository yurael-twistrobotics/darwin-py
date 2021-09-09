FROM darwin-pytorch1.9.0-cuda10.2:latest

ARG NAME

ADD ${NAME} /app/${NAME}

WORKDIR /app/${NAME}
RUN pip --no-cache-dir install -r requirements.txt

ADD entry.py .
ADD sanity_check.py .
ADD test.jpg .

# ENV GUST_ID test-gust-id
# ENV GUST_MODE inference
# ENV MODEL_ID test-model-id
# ENV REQUEST_ID test-request-id
# ENV WIND_URL wind:4001

RUN python sanity_check.py

ENTRYPOINT ["python", "-u", "entry.py"]