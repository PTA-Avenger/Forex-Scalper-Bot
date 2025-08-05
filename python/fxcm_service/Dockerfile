# Multi-stage Dockerfile for FXCM Trading Service
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Create non-root user for development
RUN groupadd -r fxcm && useradd -r -g fxcm fxcm \
    && chown -R fxcm:fxcm /app

USER fxcm

# Expose port
EXPOSE 5003

# Development command
CMD ["python", "app.py"]

# Production stage
FROM base as production

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r fxcm && useradd -r -g fxcm fxcm

# Copy only necessary files
COPY app.py /app/
COPY fxcm_client.py /app/

# Create directories for logs
RUN mkdir -p /app/logs \
    && chown -R fxcm:fxcm /app

# Switch to non-root user
USER fxcm

# Expose port
EXPOSE 5003

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5003/health || exit 1

# Environment variables
ENV FLASK_ENV=production \
    FLASK_DEBUG=0 \
    WORKERS=2 \
    TIMEOUT=60 \
    KEEP_ALIVE=2 \
    LOG_LEVEL=info

# Use Gunicorn for production
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5003", \
     "--workers", "2", \
     "--timeout", "60", \
     "--keep-alive", "2", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--preload", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "app:app"]