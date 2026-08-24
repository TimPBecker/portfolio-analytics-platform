FROM python:3.12-slim

RUN pip install --no-cache-dir \
    dagster==1.13.17 \
    dagster-webserver==1.13.17 \
    dask[distributed]==2026.7.1 \
    pandas==3.0.3 \
    pyarrow==25.0.1

WORKDIR /opt/dagster/app
ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p $DAGSTER_HOME
EXPOSE 3000
