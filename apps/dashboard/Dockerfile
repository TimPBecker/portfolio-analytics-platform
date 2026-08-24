FROM python:3.12-slim

# Install system dependencies for cryptography, build tools, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy shared core package and install it
COPY packages/portfolio-core /packages/portfolio-core
RUN pip install --no-cache-dir -e /packages/portfolio-core

# Copy dashboard dependencies and install
COPY apps/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy dashboard application source code
COPY apps/dashboard /app

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
