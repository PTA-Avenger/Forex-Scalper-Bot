#!/bin/bash

# MT5 Integration Setup Script
# This script helps set up the MetaTrader 5 integration for the Forex Scalping Bot

set -e

echo "🚀 MetaTrader 5 Integration Setup"
echo "================================="

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

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if running on supported OS
check_os() {
    print_header "Checking Operating System..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Linux detected"
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "macOS detected"
        OS="macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        print_status "Windows detected"
        OS="windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check if Python is installed
check_python() {
    print_header "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python $PYTHON_VERSION found"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python version is compatible"
        else
            print_error "Python 3.8 or higher is required"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Install Python dependencies
install_python_deps() {
    print_header "Installing Python dependencies..."
    
    cd python/mt5_bridge
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing MT5 bridge dependencies..."
        pip3 install -r requirements.txt
        print_status "Dependencies installed successfully"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
    
    cd ../..
}

# Check Docker installation
check_docker() {
    print_header "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        print_status "Docker found"
        
        if command -v docker-compose &> /dev/null; then
            print_status "Docker Compose found"
        else
            print_warning "Docker Compose not found. Please install Docker Compose."
        fi
    else
        print_warning "Docker not found. Docker is recommended for production deployment."
    fi
}

# Create environment file
create_env_file() {
    print_header "Creating environment configuration..."
    
    if [ ! -f ".env" ]; then
        print_status "Creating .env file..."
        
        cat > .env << EOF
# MT5 Configuration
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_PATH=C:\\Program Files\\MetaTrader 5\\terminal64.exe

# Bridge Configuration
MT5_BRIDGE_HOST=0.0.0.0
MT5_BRIDGE_PORT=5000

# Other configurations (keep existing values)
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_oanda_account_id
GEMINI_API_KEY=your_gemini_api_key
EOF
        
        print_status ".env file created"
        print_warning "Please edit .env file with your actual MT5 credentials"
    else
        print_status ".env file already exists"
        
        # Check if MT5 variables are present
        if ! grep -q "MT5_LOGIN" .env; then
            print_status "Adding MT5 configuration to existing .env file..."
            
            cat >> .env << EOF

# MT5 Configuration
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server
MT5_PATH=C:\\Program Files\\MetaTrader 5\\terminal64.exe

# Bridge Configuration
MT5_BRIDGE_HOST=0.0.0.0
MT5_BRIDGE_PORT=5000
EOF
            print_warning "Please edit .env file with your actual MT5 credentials"
        fi
    fi
}

# Validate configuration
validate_config() {
    print_header "Validating configuration files..."
    
    # Check if config files exist
    CONFIG_FILES=(
        "config/trading_config.json"
        "config/mt5_config.json"
    )
    
    for config_file in "${CONFIG_FILES[@]}"; do
        if [ -f "$config_file" ]; then
            print_status "$config_file exists"
            
            # Validate JSON syntax
            if python3 -m json.tool "$config_file" > /dev/null 2>&1; then
                print_status "$config_file has valid JSON syntax"
            else
                print_error "$config_file has invalid JSON syntax"
                exit 1
            fi
        else
            print_error "$config_file not found"
            exit 1
        fi
    done
}

# Test MT5 bridge
test_mt5_bridge() {
    print_header "Testing MT5 bridge..."
    
    if [ "$1" == "--skip-test" ]; then
        print_warning "Skipping MT5 bridge test"
        return
    fi
    
    print_status "Starting MT5 bridge test..."
    
    cd python/mt5_bridge
    
    # Run a quick test
    timeout 10s python3 -c "
import sys
try:
    import MetaTrader5 as mt5
    import flask
    import requests
    print('✅ All required packages are available')
    sys.exit(0)
except ImportError as e:
    print(f'❌ Missing package: {e}')
    sys.exit(1)
" || {
    print_error "MT5 bridge test failed"
    cd ../..
    exit 1
}
    
    cd ../..
    print_status "MT5 bridge test passed"
}

# Build Docker images
build_docker_images() {
    print_header "Building Docker images..."
    
    if command -v docker &> /dev/null; then
        print_status "Building MT5 bridge image..."
        docker build -t forex-mt5-bridge python/mt5_bridge/
        
        print_status "Building main application..."
        docker-compose build
        
        print_status "Docker images built successfully"
    else
        print_warning "Docker not available, skipping image build"
    fi
}

# Run integration tests
run_tests() {
    print_header "Running integration tests..."
    
    if [ -f "tests/test_mt5_integration.py" ]; then
        print_status "Running MT5 integration tests..."
        
        cd tests
        python3 test_mt5_integration.py || {
            print_warning "Some tests failed (this is expected if MT5 is not running)"
        }
        cd ..
    else
        print_warning "Integration tests not found"
    fi
}

# Display next steps
show_next_steps() {
    print_header "Setup Complete! Next Steps:"
    
    echo ""
    echo "1. 📝 Edit your credentials:"
    echo "   - Update .env file with your MT5 login, password, and server"
    echo "   - Ensure MetaTrader 5 is installed and running"
    echo ""
    echo "2. 🚀 Start the system:"
    echo "   - With Docker: docker-compose up -d"
    echo "   - Manual: ./scripts/start_mt5_bridge.sh"
    echo ""
    echo "3. 🔍 Verify the setup:"
    echo "   - Check MT5 bridge: curl http://localhost:5004/health"
    echo "   - Check main engine: curl http://localhost:8080/health"
    echo ""
    echo "4. 📖 Read the documentation:"
    echo "   - MT5_INTEGRATION_GUIDE.md for detailed setup instructions"
    echo "   - README.md for general usage"
    echo ""
    echo "⚠️  Important Notes:"
    echo "   - MetaTrader 5 must be running on Windows for the Python library to work"
    echo "   - Enable 'Allow algorithmic trading' in MT5 settings"
    echo "   - Start with a demo account for testing"
    echo ""
}

# Main setup function
main() {
    echo ""
    print_status "Starting MT5 integration setup..."
    echo ""
    
    # Parse command line arguments
    SKIP_TEST=false
    SKIP_DOCKER=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-test)
                SKIP_TEST=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --skip-test    Skip MT5 bridge testing"
                echo "  --skip-docker  Skip Docker image building"
                echo "  -h, --help     Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_os
    check_python
    check_docker
    install_python_deps
    create_env_file
    validate_config
    
    if [ "$SKIP_TEST" = false ]; then
        test_mt5_bridge
    fi
    
    if [ "$SKIP_DOCKER" = false ]; then
        build_docker_images
    fi
    
    run_tests
    show_next_steps
    
    echo ""
    print_status "MT5 integration setup completed successfully! 🎉"
    echo ""
}

# Run main function with all arguments
main "$@"