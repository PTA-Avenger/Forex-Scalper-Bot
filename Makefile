# Forex Scalping Bot Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help build test clean install deploy docker-build docker-test docker-deploy

# Default target
help:
	@echo "Forex Scalping Bot - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  install          Install all dependencies"
	@echo "  build            Build all components"
	@echo "  test             Run all tests"
	@echo "  test-cpp         Run C++ tests only"
	@echo "  test-python      Run Python tests only"
	@echo "  test-frontend    Run frontend tests only"
	@echo "  lint             Run code linting"
	@echo "  format           Format code"
	@echo "  clean            Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-test      Run tests in Docker"
	@echo "  docker-up        Start all services"
	@echo "  docker-down      Stop all services"
	@echo "  docker-logs      View service logs"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-dev       Deploy to development environment"
	@echo "  deploy-prod      Deploy to production environment"
	@echo "  backup           Backup production data"
	@echo ""

# Installation
install: install-cpp install-python install-frontend

install-cpp:
	@echo "Installing C++ dependencies..."
	sudo apt-get update
	sudo apt-get install -y build-essential cmake libboost-all-dev libssl-dev \
		libcurl4-openssl-dev nlohmann-json3-dev libhiredis-dev postgresql-server-dev-all

install-python:
	@echo "Installing Python dependencies..."
	cd python && pip install -r requirements.txt

install-frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

# Building
build: build-cpp build-python build-frontend

build-cpp:
	@echo "Building C++ engine..."
	mkdir -p build
	cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$$(nproc)

build-python:
	@echo "Building Python services..."
	cd python && python -m py_compile **/*.py

build-frontend:
	@echo "Building frontend..."
	cd frontend && npm run build

# Testing
test: test-cpp test-python test-frontend

test-cpp:
	@echo "Running C++ tests..."
	cd build && make test

test-python:
	@echo "Running Python tests..."
	cd python && pytest -v --tb=short --cov=. --cov-report=xml

test-frontend:
	@echo "Running frontend tests..."
	cd frontend && npm run test:coverage

test-integration:
	@echo "Running integration tests..."
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

# Code Quality
lint: lint-cpp lint-python lint-frontend

lint-cpp:
	@echo "Linting C++ code..."
	find src include -name "*.cpp" -o -name "*.h" | xargs clang-format --dry-run --Werror

lint-python:
	@echo "Linting Python code..."
	cd python && flake8 . && mypy .

lint-frontend:
	@echo "Linting frontend code..."
	cd frontend && npm run lint

format: format-cpp format-python format-frontend

format-cpp:
	@echo "Formatting C++ code..."
	find src include -name "*.cpp" -o -name "*.h" | xargs clang-format -i

format-python:
	@echo "Formatting Python code..."
	cd python && black . && isort .

format-frontend:
	@echo "Formatting frontend code..."
	cd frontend && npm run format

# Performance Testing
benchmark:
	@echo "Running performance benchmarks..."
	cd build && ./performance_tests --benchmark_format=json > benchmark_results.json

# Docker Operations
docker-build:
	@echo "Building Docker images..."
	docker-compose build --parallel

docker-test:
	@echo "Running tests in Docker..."
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
	docker-compose -f docker-compose.test.yml down -v

docker-up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services starting... Use 'make docker-logs' to view logs"

docker-down:
	@echo "Stopping all services..."
	docker-compose down

docker-logs:
	@echo "Viewing service logs..."
	docker-compose logs -f

docker-clean:
	@echo "Cleaning Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f

# Health Checks
health:
	@echo "Checking service health..."
	@curl -f http://localhost:8080/health && echo "✅ C++ Engine: Healthy" || echo "❌ C++ Engine: Unhealthy"
	@curl -f http://localhost:5001/health && echo "✅ Price Predictor: Healthy" || echo "❌ Price Predictor: Unhealthy"
	@curl -f http://localhost:3000/health && echo "✅ Frontend: Healthy" || echo "❌ Frontend: Unhealthy"

# Database Operations
db-migrate:
	@echo "Running database migrations..."
	cd database && psql -h localhost -U forex_user -d forex_bot -f migrations/*.sql

db-backup:
	@echo "Creating database backup..."
	pg_dump -h localhost -U forex_user forex_bot > backup_$$(date +%Y%m%d_%H%M%S).sql

db-restore:
	@echo "Restoring database from backup..."
	@read -p "Enter backup file path: " backup_file; \
	psql -h localhost -U forex_user -d forex_bot < $$backup_file

# Monitoring
logs:
	@echo "Viewing application logs..."
	tail -f logs/*.log

monitor:
	@echo "Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3001"
	@echo "Prometheus: http://localhost:9090"

# Deployment
deploy-dev: test docker-build
	@echo "Deploying to development environment..."
	docker-compose -f docker-compose.dev.yml up -d

deploy-prod: test docker-build
	@echo "Deploying to production environment..."
	@read -p "Are you sure you want to deploy to production? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		docker-compose -f docker-compose.prod.yml up -d; \
	else \
		echo "Deployment cancelled"; \
	fi

# Security
security-scan:
	@echo "Running security scans..."
	docker run --rm -v $$(pwd):/workspace aquasec/trivy fs /workspace
	cd python && bandit -r .
	cd frontend && npm audit

# FXCM Testing
test-fxcm:
	@echo "🧪 Testing FXCM integration..."
	cd python/fxcm_service && python test_fxcm_integration.py

test-all: test test-fxcm
	@echo "🎉 All tests completed!"

# Multi-Broker Support
up-mt5:
	@echo "🚀 Starting services with MT5 support..."
	docker-compose --profile mt5 up -d

up-fxcm:
	@echo "🚀 Starting services with FXCM only..."
	docker-compose up -d

up-all-brokers:
	@echo "🚀 Starting services with all brokers..."
	docker-compose --profile mt5 up -d

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf python/**/__pycache__/
	rm -rf python/**/*.pyc
	rm -rf frontend/build/
	rm -rf frontend/node_modules/.cache/
	rm -rf logs/*.log

# Development Environment Setup
dev-setup: install
	@echo "Setting up development environment..."
	cp .env.example .env
	@echo "Please edit .env file with your configuration"
	@echo "Then run: make docker-up"

# Quick Start
quick-start: dev-setup docker-build docker-up
	@echo "🚀 Forex Scalping Bot is starting up!"
	@echo "📊 Dashboard: http://localhost:3000"
	@echo "🔧 API: http://localhost:8080"
	@echo "🤖 Gemini AI Service: http://localhost:5001"
	@echo "💱 FXCM Trading Service: http://localhost:5004"
	@echo "📈 Signal Processor: http://localhost:5006"
	@echo ""
	@echo "⚠️  Make sure to set GEMINI_API_KEY and FXCM_ACCESS_TOKEN in your .env file"
	@echo "💡 Optional: Enable MT5 with 'make up-mt5' (requires MT5 credentials)"
	@echo "Use 'make health' to check service status"
	@echo "Use 'make logs' to view application logs"
	@echo "Use 'make test-fxcm' to test FXCM integration"