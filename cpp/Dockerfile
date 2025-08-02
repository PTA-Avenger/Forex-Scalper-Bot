# Multi-stage Dockerfile for C++ Trading Engine
FROM ubuntu:22.04 AS base

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libboost-all-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev \
    libhiredis-dev \
    postgresql-server-dev-all \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r forex && useradd -r -g forex -u 1001 forex

# Build stage
FROM base AS builder

WORKDIR /src

# Copy source code
COPY CMakeLists.txt ./
COPY src/ ./src/
COPY include/ ./include/
COPY tests/ ./tests/

# Build the application
RUN mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release \
             -DCMAKE_INSTALL_PREFIX=/app \
             -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG -march=native" \
             .. \
    && make -j$(nproc) \
    && make install

# Test stage (optional, can be skipped in production builds)
FROM builder AS tester

RUN cd build && make test

# Production stage
FROM ubuntu:22.04 AS production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libboost-system1.74.0 \
    libboost-thread1.74.0 \
    libboost-chrono1.74.0 \
    libboost-filesystem1.74.0 \
    libssl3 \
    libcurl4 \
    libhiredis0.14 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directory
WORKDIR /app

# Create non-root user
RUN groupadd -r forex && useradd -r -g forex -u 1001 forex \
    && chown -R forex:forex /app

# Copy built application from builder stage
COPY --from=builder --chown=forex:forex /app/ /app/

# Copy configuration files
COPY --chown=forex:forex config/ /app/config/

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data \
    && chown -R forex:forex /app/logs /app/data

# Switch to non-root user
USER forex

# Environment variables
ENV PATH="/app/bin:${PATH}"
ENV LOG_LEVEL=info
ENV CONFIG_PATH=/app/config/trading_config.json

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["ForexScalpingBot", "--config", "/app/config/trading_config.json"]

# Development stage (for development builds)
FROM base AS development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    gdb \
    valgrind \
    strace \
    htop \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install development dependencies
RUN apt-get update && apt-get install -y \
    libgtest-dev \
    libgmock-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy source for development
COPY . .

# Build in debug mode
RUN mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Debug \
             -DCMAKE_CXX_FLAGS="-g -O0 -DDEBUG" \
             .. \
    && make -j$(nproc)

# Switch to non-root user
RUN chown -R forex:forex /app
USER forex

# Default command for development
CMD ["./build/ForexScalpingBot", "--config", "config/trading_config.json", "--verbose"]