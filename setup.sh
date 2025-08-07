#!/bin/bash

# Trading Bot Hybrid Architecture Setup Script
# This script sets up the complete MT5 Windows VM + Linux Bot + InfluxDB architecture

set -e

echo "🚀 Setting up Trading Bot Hybrid Architecture..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed."
}

# Create environment file
create_env_file() {
    print_step "Creating environment configuration..."
    
    if [ ! -f .env ]; then
        cp .env.example .env
        print_warning "Created .env file from template. Please update with your actual values:"
        print_warning "- GEMINI_API_KEY: Your Google Gemini API key"
        print_warning "- MT5_VM_IP: IP address of your Windows VM running MT5"
        echo ""
        read -p "Do you want to continue with default values for testing? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Please edit .env file and run this script again."
            exit 0
        fi
    else
        print_status ".env file already exists."
    fi
}

# Create necessary directories
create_directories() {
    print_step "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p config
    
    print_status "Directories created."
}

# Start the services
start_services() {
    print_step "Starting services with Docker Compose..."
    
    # Pull images first
    docker-compose pull
    
    # Build and start services
    docker-compose up -d --build
    
    print_status "Services started successfully!"
}

# Wait for services to be healthy
wait_for_services() {
    print_step "Waiting for services to be healthy..."
    
    # Wait for InfluxDB
    print_status "Waiting for InfluxDB..."
    timeout=60
    while ! curl -s http://localhost:8086/ping > /dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            print_error "InfluxDB failed to start within 60 seconds"
            exit 1
        fi
    done
    print_status "InfluxDB is healthy"
    
    # Wait for Trading Bot
    print_status "Waiting for Trading Bot..."
    timeout=60
    while ! curl -s http://localhost:8080/api/health > /dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            print_error "Trading Bot failed to start within 60 seconds"
            exit 1
        fi
    done
    print_status "Trading Bot is healthy"
    
    # Wait for Grafana
    print_status "Waiting for Grafana..."
    timeout=60
    while ! curl -s http://localhost:3000/api/health > /dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [ $timeout -le 0 ]; then
            print_error "Grafana failed to start within 60 seconds"
            exit 1
        fi
    done
    print_status "Grafana is healthy"
}

# Display service information
display_info() {
    print_step "Setup completed successfully! 🎉"
    echo ""
    echo "📊 Service URLs:"
    echo "  • Trading Bot API: http://localhost:8080"
    echo "  • InfluxDB UI: http://localhost:8086"
    echo "  • Grafana Dashboard: http://localhost:3000"
    echo "  • Redis: localhost:6379"
    echo ""
    echo "🔐 Default Credentials:"
    echo "  • InfluxDB: admin / trading123!"
    echo "  • Grafana: admin / trading123!"
    echo "  • Redis: trading123!"
    echo ""
    echo "📡 API Endpoints:"
    echo "  • Health Check: GET http://localhost:8080/api/health"
    echo "  • Signal Receiver: POST http://localhost:8080/api/signals"
    echo "  • Trade Execution: POST http://localhost:8080/api/trade-execution"
    echo "  • Statistics: GET http://localhost:8080/api/statistics"
    echo ""
    echo "📁 Important Files:"
    echo "  • MT5 Expert Advisor: MT5_SignalSender.mq5"
    echo "  • Trading Bot: linux_trading_bot.py"
    echo "  • Configuration: .env"
    echo "  • Logs: ./logs/"
    echo ""
    echo "🚀 Next Steps:"
    echo "  1. Install MT5_SignalSender.mq5 in your Windows VM MT5"
    echo "  2. Update the Linux bot IP in MT5 EA settings"
    echo "  3. Configure your Gemini API key in .env"
    echo "  4. Monitor the system via Grafana dashboard"
    echo ""
    print_warning "Remember to update .env with your actual API keys and IP addresses!"
}

# Test the system
test_system() {
    print_step "Testing system connectivity..."
    
    # Test Trading Bot API
    if curl -s http://localhost:8080/api/health | grep -q "healthy"; then
        print_status "✅ Trading Bot API is responding"
    else
        print_error "❌ Trading Bot API is not responding"
    fi
    
    # Test InfluxDB
    if curl -s http://localhost:8086/ping | grep -q "OK"; then
        print_status "✅ InfluxDB is responding"
    else
        print_error "❌ InfluxDB is not responding"
    fi
    
    # Test Grafana
    if curl -s http://localhost:3000/api/health | grep -q "ok"; then
        print_status "✅ Grafana is responding"
    else
        print_error "❌ Grafana is not responding"
    fi
}

# Main execution
main() {
    echo "===========================================" 
    echo "   Trading Bot Hybrid Architecture Setup   "
    echo "==========================================="
    echo ""
    
    check_docker
    create_env_file
    create_directories
    start_services
    wait_for_services
    test_system
    display_info
}

# Run main function
main

# Optional: Show logs
echo ""
read -p "Do you want to view the logs? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose logs -f
fi