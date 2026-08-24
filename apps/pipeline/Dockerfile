FROM python:3.12-slim

# Install system dependencies for cryptography, build tools, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/dagster/app

# Copy shared core package and install it
COPY packages/portfolio-core /packages/portfolio-core
RUN pip install --no-cache-dir -e /packages/portfolio-core

# Copy pipeline requirements and install
COPY apps/pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline application source code
COPY apps/pipeline /opt/dagster/app

ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p $DAGSTER_HOME

EXPOSE 3000

ENTRYPOINT ["dagster", "dev", "-h", "0.0.0.0", "-p", "3000", "-f", "repo.py"]
