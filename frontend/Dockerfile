# Multi-stage Dockerfile for React Frontend
FROM node:18-alpine AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache \
    curl \
    && addgroup -g 1001 -S forex \
    && adduser -S forex -u 1001

# Dependencies stage
FROM base AS dependencies

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production \
    && cp -R node_modules /tmp/node_modules

# Install all dependencies (including dev)
RUN npm ci

# Development stage
FROM dependencies AS development

# Copy source code
COPY . .

# Change ownership to non-root user
RUN chown -R forex:forex /app

# Switch to non-root user
USER forex

# Environment variables
ENV NODE_ENV=development \
    REACT_APP_API_URL=http://localhost:8080 \
    REACT_APP_WS_URL=ws://localhost:8080

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Default command
CMD ["npm", "start"]

# Build stage
FROM dependencies AS builder

# Copy source code
COPY . .

# Set production environment variables
ENV NODE_ENV=production \
    REACT_APP_API_URL=/api \
    REACT_APP_WS_URL=/ws

# Build the application
RUN npm run build

# Production stage with NGINX
FROM nginx:1.25-alpine AS production

# Install curl for health checks
RUN apk add --no-cache curl

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built application from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy NGINX configuration template
COPY nginx.template /etc/nginx/conf.d/default.template

# Create non-root user
RUN addgroup -g 1001 -S forex \
    && adduser -S forex -u 1001 \
    && chown -R forex:forex /usr/share/nginx/html \
    && chown -R forex:forex /var/cache/nginx \
    && chown -R forex:forex /var/log/nginx \
    && chown -R forex:forex /etc/nginx/conf.d \
    && touch /var/run/nginx.pid \
    && chown -R forex:forex /var/run/nginx.pid

# Switch to non-root user
USER forex

# Environment variables
ENV NGINX_PORT=3000

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Start NGINX
CMD ["sh", "-c", "envsubst '$$NGINX_PORT' < /etc/nginx/conf.d/default.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]

# Testing stage
FROM dependencies AS testing

# Copy all source code for testing
COPY . .

# Change ownership
RUN chown -R forex:forex /app

# Switch to non-root user
USER forex

# Run tests
CMD ["npm", "run", "test:coverage"]