# Lightweight Dockerfile for Gemini-powered Price Predictor
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies (removed heavy ML dependencies)
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r forex && useradd -r -g forex -u 1001 forex

# Dependencies stage
FROM base AS dependencies

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Development stage
FROM dependencies AS development

WORKDIR /app

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest-xdist \
    pytest-mock \
    jupyterlab \
    ipdb

# Copy source code
COPY . /app/

# Change ownership to non-root user
RUN chown -R forex:forex /app

# Switch to non-root user
USER forex

# Environment variables
ENV FLASK_ENV=development \
    FLASK_DEBUG=1

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Default command
CMD ["python", "app.py"]

# Production stage
FROM dependencies AS production

WORKDIR /app

# Copy only necessary files (removed model directories as we use Gemini API)
COPY app.py /app/
COPY gemini_predictor.py /app/
COPY sentiment_analyzer/ /app/sentiment_analyzer/

# Create directories for logs (no need for model storage with Gemini API)
RUN mkdir -p /app/logs \
    && chown -R forex:forex /app

# Switch to non-root user
USER forex

# Environment variables
ENV FLASK_ENV=production \
    FLASK_DEBUG=0 \
    WORKERS=2 \
    TIMEOUT=60 \
    KEEP_ALIVE=2 \
    LOG_LEVEL=info

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Use Gunicorn for production
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5001", \
     "--workers", "2", \
     "--timeout", "60", \
     "--keep-alive", "2", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]

# Testing stage
FROM dependencies AS testing

WORKDIR /app

# Copy all source code for testing
COPY . /app/

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest-cov \
    pytest-benchmark \
    pytest-mock

# Change ownership
RUN chown -R forex:forex /app

# Switch to non-root user
USER forex

# Run tests
CMD ["pytest", "-v", "--tb=short", "--cov=.", "--cov-report=xml", "--cov-report=html"]