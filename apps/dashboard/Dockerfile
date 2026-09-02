FROM python:3.12-slim

# Install system dependencies for cryptography, build tools, curl, fonts, wget
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    libfontconfig1 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Install Quarto CLI (includes Typst compiler) - multi-arch support for amd64 and arm64
ARG QUARTO_VERSION=1.6.42
RUN ARCH=$(dpkg --print-architecture) \
    && wget -q https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${ARCH}.deb \
    && dpkg -i quarto-${QUARTO_VERSION}-linux-${ARCH}.deb \
    && rm quarto-${QUARTO_VERSION}-linux-${ARCH}.deb

# Disable any telemetry in Quarto and Streamlit
ENV QUARTO_DENY_TELEMETRY=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Copy shared core package and install it
COPY packages/portfolio-core /packages/portfolio-core
RUN pip install --no-cache-dir -e /packages/portfolio-core

# Copy dashboard dependencies and install
COPY apps/dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy reports templates & dashboard application source code
COPY reports /reports
COPY apps/dashboard /app

# Expose Streamlit default and development ports
EXPOSE 8501 8502

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
