#!/bin/bash

# MT5 Demo Account Setup Script
# Sets up demo accounts with top 3 recommended brokers for MT5 integration

set -e

echo "🚀 MT5 Demo Account Setup"
echo "========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

print_highlight() {
    echo -e "${CYAN}$1${NC}"
}

# Function to open URL in browser
open_url() {
    local url=$1
    if command -v xdg-open > /dev/null; then
        xdg-open "$url"
    elif command -v open > /dev/null; then
        open "$url"
    elif command -v start > /dev/null; then
        start "$url"
    else
        print_warning "Please manually open: $url"
    fi
}

# Function to wait for user input
wait_for_user() {
    echo ""
    read -p "Press Enter when you've completed the registration..."
    echo ""
}

# Function to collect demo account details
collect_demo_details() {
    local broker_name=$1
    echo ""
    print_header "📝 Enter your $broker_name demo account details:"
    
    read -p "Login/Account Number: " demo_login
    read -s -p "Password: " demo_password
    echo ""
    read -p "Server: " demo_server
    
    # Store in temporary variables
    eval "${broker_name}_LOGIN=\"$demo_login\""
    eval "${broker_name}_PASSWORD=\"$demo_password\""
    eval "${broker_name}_SERVER=\"$demo_server\""
    
    print_status "Demo account details saved for $broker_name"
}

# Function to setup IC Markets demo
setup_icmarkets_demo() {
    print_header "🏆 Setting up IC Markets Demo Account"
    print_highlight "✅ Unlimited demo account"
    print_highlight "✅ Raw spreads from 0.0 pips"
    print_highlight "✅ Perfect for scalping bots"
    print_highlight "✅ Multiple regulatory licenses"
    
    echo ""
    print_status "Opening IC Markets demo registration page..."
    
    # Open IC Markets demo registration
    open_url "https://www.icmarkets.com/en/open-trading-account/demo"
    
    echo ""
    print_warning "Please complete the following steps:"
    echo "1. Fill out the demo account form"
    echo "2. Select 'Raw Spread' account type (recommended for scalping)"
    echo "3. Choose MT5 platform"
    echo "4. Set demo balance (recommended: $100,000)"
    echo "5. Submit the form"
    echo "6. Check your email for login credentials"
    
    wait_for_user
    collect_demo_details "ICMARKETS"
}

# Function to setup XM Group demo
setup_xm_demo() {
    print_header "🌟 Setting up XM Group Demo Account"
    print_highlight "✅ Unlimited demo account"
    print_highlight "✅ $100,000 virtual balance"
    print_highlight "✅ Great for beginners"
    print_highlight "✅ Excellent support"
    
    echo ""
    print_status "Opening XM Group demo registration page..."
    
    # Open XM demo registration
    open_url "https://www.xm.com/register/demo"
    
    echo ""
    print_warning "Please complete the following steps:"
    echo "1. Fill out the demo account form"
    echo "2. Select 'Standard' account type"
    echo "3. Choose MT5 platform"
    echo "4. Set leverage to 1:100 or 1:500"
    echo "5. Submit the form"
    echo "6. Check your email for login credentials"
    
    wait_for_user
    collect_demo_details "XM"
}

# Function to setup Dukascopy demo
setup_dukascopy_demo() {
    print_header "🏦 Setting up Dukascopy Bank Demo Account"
    print_highlight "✅ Swiss bank regulation"
    print_highlight "✅ 14-day demo (renewable)"
    print_highlight "✅ ECN marketplace access"
    print_highlight "✅ Professional platform"
    
    echo ""
    print_status "Opening Dukascopy demo registration page..."
    
    # Open Dukascopy demo registration
    open_url "https://www.dukascopy.com/swiss/english/forex/demo-mt5-account/"
    
    echo ""
    print_warning "Please complete the following steps:"
    echo "1. Fill out the demo account form"
    echo "2. Select MT5 checkbox"
    echo "3. Choose your preferred currency (USD recommended)"
    echo "4. Set demo balance (recommended: $100,000)"
    echo "5. Submit the form"
    echo "6. Check your email for login credentials"
    
    wait_for_user
    collect_demo_details "DUKASCOPY"
}

# Function to update environment file
update_env_file() {
    print_header "📝 Updating Environment Configuration"
    
    # Backup existing .env if it exists
    if [ -f ".env" ]; then
        cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
        print_status "Backed up existing .env file"
    fi
    
    # Create or update .env file
    cat > .env << EOF
# MT5 Demo Account Configurations
# Generated on $(date)

# Primary Broker: IC Markets (Recommended for scalping)
MT5_LOGIN=${ICMARKETS_LOGIN:-your_icmarkets_login}
MT5_PASSWORD=${ICMARKETS_PASSWORD:-your_icmarkets_password}
MT5_SERVER=${ICMARKETS_SERVER:-your_icmarkets_server}

# Alternative Broker 1: XM Group
XM_LOGIN=${XM_LOGIN:-your_xm_login}
XM_PASSWORD=${XM_PASSWORD:-your_xm_password}
XM_SERVER=${XM_SERVER:-your_xm_server}

# Alternative Broker 2: Dukascopy Bank
DUKASCOPY_LOGIN=${DUKASCOPY_LOGIN:-your_dukascopy_login}
DUKASCOPY_PASSWORD=${DUKASCOPY_PASSWORD:-your_dukascopy_password}
DUKASCOPY_SERVER=${DUKASCOPY_SERVER:-your_dukascopy_server}

# MT5 Bridge Configuration
MT5_BRIDGE_HOST=0.0.0.0
MT5_BRIDGE_PORT=5000

# Other configurations
GEMINI_API_KEY=your_gemini_api_key
OANDA_API_KEY=your_oanda_api_key
OANDA_ACCOUNT_ID=your_oanda_account_id
EOF
    
    print_status ".env file updated with demo account configurations"
}

# Function to create broker-specific config files
create_broker_configs() {
    print_header "⚙️ Creating Broker-Specific Configuration Files"
    
    # IC Markets config
    cat > config/icmarkets_config.json << EOF
{
  "broker": "icmarkets",
  "mt5": {
    "login": "\${ICMARKETS_LOGIN}",
    "password": "\${ICMARKETS_PASSWORD}",
    "server": "\${ICMARKETS_SERVER}",
    "bridge_url": "http://localhost:5004",
    "magic_number": 12345
  },
  "symbols": [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF",
    "NZDUSD", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"
  ],
  "trading": {
    "account_type": "raw_spread",
    "min_lot_size": 0.01,
    "max_lot_size": 50.0,
    "max_spread": 2.0,
    "slippage": 3
  },
  "features": {
    "scalping_friendly": true,
    "ea_allowed": true,
    "hedging_allowed": true,
    "fifo_rule": false
  }
}
EOF

    # XM Group config
    cat > config/xm_config.json << EOF
{
  "broker": "xm",
  "mt5": {
    "login": "\${XM_LOGIN}",
    "password": "\${XM_PASSWORD}",
    "server": "\${XM_SERVER}",
    "bridge_url": "http://localhost:5004",
    "magic_number": 12346
  },
  "symbols": [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF",
    "NZDUSD", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"
  ],
  "trading": {
    "account_type": "standard",
    "min_lot_size": 0.01,
    "max_lot_size": 50.0,
    "max_spread": 3.0,
    "slippage": 3
  },
  "features": {
    "scalping_friendly": true,
    "ea_allowed": true,
    "hedging_allowed": true,
    "fifo_rule": false
  }
}
EOF

    # Dukascopy config
    cat > config/dukascopy_config.json << EOF
{
  "broker": "dukascopy",
  "mt5": {
    "login": "\${DUKASCOPY_LOGIN}",
    "password": "\${DUKASCOPY_PASSWORD}",
    "server": "\${DUKASCOPY_SERVER}",
    "bridge_url": "http://localhost:5004",
    "magic_number": 12347
  },
  "symbols": [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF",
    "NZDUSD", "USDCAD", "EURJPY", "GBPJPY", "AUDJPY"
  ],
  "trading": {
    "account_type": "ecn",
    "min_lot_size": 0.001,
    "max_lot_size": 100.0,
    "max_spread": 1.5,
    "slippage": 2
  },
  "features": {
    "scalping_friendly": true,
    "ea_allowed": true,
    "hedging_allowed": true,
    "fifo_rule": false
  }
}
EOF
    
    print_status "Created broker-specific configuration files"
}

# Function to create broker switching script
create_broker_switcher() {
    print_header "🔄 Creating Broker Switching Script"
    
    cat > scripts/switch_broker.sh << 'EOF'
#!/bin/bash

# Broker Switching Script
# Allows easy switching between demo brokers

BROKERS=("icmarkets" "xm" "dukascopy")

echo "🔄 MT5 Broker Switcher"
echo "======================"

echo "Available brokers:"
for i in "${!BROKERS[@]}"; do
    echo "$((i+1)). ${BROKERS[$i]}"
done

echo ""
read -p "Select broker (1-3): " choice

case $choice in
    1)
        SELECTED="icmarkets"
        LOGIN_VAR="ICMARKETS_LOGIN"
        PASSWORD_VAR="ICMARKETS_PASSWORD"
        SERVER_VAR="ICMARKETS_SERVER"
        ;;
    2)
        SELECTED="xm"
        LOGIN_VAR="XM_LOGIN"
        PASSWORD_VAR="XM_PASSWORD"
        SERVER_VAR="XM_SERVER"
        ;;
    3)
        SELECTED="dukascopy"
        LOGIN_VAR="DUKASCOPY_LOGIN"
        PASSWORD_VAR="DUKASCOPY_PASSWORD"
        SERVER_VAR="DUKASCOPY_SERVER"
        ;;
    *)
        echo "Invalid selection"
        exit 1
        ;;
esac

echo "Switching to $SELECTED broker..."

# Update main MT5 config to point to selected broker
sed -i.bak \
    -e "s/MT5_LOGIN=.*/MT5_LOGIN=\${$LOGIN_VAR}/" \
    -e "s/MT5_PASSWORD=.*/MT5_PASSWORD=\${$PASSWORD_VAR}/" \
    -e "s/MT5_SERVER=.*/MT5_SERVER=\${$SERVER_VAR}/" \
    .env

# Copy broker-specific config to main config
cp "config/${SELECTED}_config.json" "config/mt5_config.json"

echo "✅ Switched to $SELECTED broker"
echo "Please restart your MT5 bridge service"
EOF

    chmod +x scripts/switch_broker.sh
    print_status "Created broker switching script"
}

# Function to create testing script
create_test_script() {
    print_header "🧪 Creating Demo Account Testing Script"
    
    cat > scripts/test_demo_accounts.sh << 'EOF'
#!/bin/bash

# Demo Account Testing Script
# Tests connection to all configured demo accounts

echo "🧪 Testing Demo Account Connections"
echo "==================================="

source .env

test_broker() {
    local name=$1
    local login=$2
    local password=$3
    local server=$4
    
    echo ""
    echo "Testing $name..."
    echo "Login: $login"
    echo "Server: $server"
    
    # This is a placeholder - actual testing would require MT5 terminal
    if [ -n "$login" ] && [ -n "$password" ] && [ -n "$server" ]; then
        echo "✅ Configuration complete for $name"
    else
        echo "❌ Missing configuration for $name"
    fi
}

# Test all brokers
test_broker "IC Markets" "$ICMARKETS_LOGIN" "$ICMARKETS_PASSWORD" "$ICMARKETS_SERVER"
test_broker "XM Group" "$XM_LOGIN" "$XM_PASSWORD" "$XM_SERVER"
test_broker "Dukascopy" "$DUKASCOPY_LOGIN" "$DUKASCOPY_PASSWORD" "$DUKASCOPY_SERVER"

echo ""
echo "💡 To test actual connections, use MT5 terminal or run:"
echo "   python python/mt5_bridge/mt5_bridge.py"
EOF

    chmod +x scripts/test_demo_accounts.sh
    print_status "Created demo account testing script"
}

# Function to display next steps
show_next_steps() {
    print_header "🎉 Demo Account Setup Complete!"
    
    echo ""
    print_highlight "📋 What you now have:"
    echo "✅ Demo accounts with 3 top MT5 brokers"
    echo "✅ Environment configuration (.env)"
    echo "✅ Broker-specific config files"
    echo "✅ Broker switching script"
    echo "✅ Testing script"
    
    echo ""
    print_highlight "🚀 Next Steps:"
    echo ""
    echo "1. 📝 Verify your demo account details:"
    echo "   cat .env"
    echo ""
    echo "2. 🔄 Switch between brokers easily:"
    echo "   ./scripts/switch_broker.sh"
    echo ""
    echo "3. 🧪 Test your demo connections:"
    echo "   ./scripts/test_demo_accounts.sh"
    echo ""
    echo "4. 🚀 Start your MT5 bridge:"
    echo "   docker-compose up -d mt5-bridge"
    echo ""
    echo "5. 📊 Test your trading bot:"
    echo "   python tests/test_mt5_integration.py"
    echo ""
    print_highlight "💡 Pro Tips:"
    echo "• Start with IC Markets (best for scalping)"
    echo "• Test all brokers to compare execution"
    echo "• Keep demo accounts active by logging in regularly"
    echo "• Monitor spreads during different market sessions"
    echo ""
    print_highlight "📞 Support:"
    echo "• IC Markets: 24/7 live chat"
    echo "• XM Group: Multilingual support"
    echo "• Dukascopy: European business hours"
    echo ""
}

# Main setup function
main() {
    echo ""
    print_status "Starting demo account setup for top 3 MT5 brokers..."
    echo ""
    
    # Check if scripts directory exists
    mkdir -p scripts
    mkdir -p config
    
    # Setup demo accounts
    setup_icmarkets_demo
    setup_xm_demo
    setup_dukascopy_demo
    
    # Create configuration files
    update_env_file
    create_broker_configs
    create_broker_switcher
    create_test_script
    
    # Show completion message
    show_next_steps
    
    echo ""
    print_status "Demo account setup completed successfully! 🎉"
    echo ""
}

# Run main function
main "$@"