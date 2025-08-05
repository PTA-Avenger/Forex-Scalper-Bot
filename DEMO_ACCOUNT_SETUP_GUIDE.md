# 🚀 MT5 Demo Account Setup Guide

## Quick Start

Run the automated setup script:

```bash
chmod +x scripts/setup_demo_accounts.sh
./scripts/setup_demo_accounts.sh
```

This script will:
1. ✅ Open registration pages for all 3 brokers
2. ✅ Guide you through the signup process
3. ✅ Collect your demo credentials
4. ✅ Configure your environment automatically
5. ✅ Create broker switching tools

## 🏆 Top 3 MT5 Demo Brokers

### 1. IC Markets (Recommended for Scalping)
- **Demo Duration**: Unlimited ♾️
- **Virtual Balance**: Customizable ($10K - $1M)
- **Account Type**: Raw Spread (0.0 pip spreads)
- **Regulation**: ASIC, CySEC, FSA
- **Best For**: High-frequency scalping bots
- **Registration**: https://www.icmarkets.com/en/open-trading-account/demo

**Why IC Markets?**
- ✅ Perfect for your scalping bot
- ✅ Raw spreads from 0.0 pips
- ✅ Excellent execution speed
- ✅ No restrictions on EAs/scalping

### 2. XM Group (Best for Beginners)
- **Demo Duration**: Unlimited ♾️
- **Virtual Balance**: $100,000
- **Account Type**: Standard
- **Regulation**: CySEC, ASIC, IFSC
- **Best For**: Learning and testing
- **Registration**: https://www.xm.com/register/demo

**Why XM Group?**
- ✅ Excellent educational resources
- ✅ Multilingual support
- ✅ User-friendly platform
- ✅ Great for beginners

### 3. Dukascopy Bank (Swiss Quality)
- **Demo Duration**: 14 days (renewable)
- **Virtual Balance**: Customizable
- **Account Type**: ECN
- **Regulation**: FINMA (Switzerland)
- **Best For**: Professional trading
- **Registration**: https://www.dukascopy.com/swiss/english/forex/demo-mt5-account/

**Why Dukascopy?**
- ✅ Swiss bank regulation
- ✅ ECN marketplace access
- ✅ Institutional-grade platform
- ✅ Can get permanent demo

## 📋 Manual Setup Instructions

### Step 1: IC Markets Demo Account

1. **Visit**: https://www.icmarkets.com/en/open-trading-account/demo
2. **Fill out the form**:
   - Personal details
   - Account type: **Raw Spread** (recommended)
   - Platform: **MetaTrader 5**
   - Demo balance: **$100,000**
   - Leverage: **1:500**
3. **Submit** and check your email
4. **Save credentials**: Login, Password, Server

### Step 2: XM Group Demo Account

1. **Visit**: https://www.xm.com/register/demo
2. **Fill out the form**:
   - Personal details
   - Account type: **Standard**
   - Platform: **MT5**
   - Leverage: **1:100** or **1:500**
3. **Submit** and check your email
4. **Save credentials**: Login, Password, Server

### Step 3: Dukascopy Demo Account

1. **Visit**: https://www.dukascopy.com/swiss/english/forex/demo-mt5-account/
2. **Fill out the form**:
   - Personal details
   - Check **MT5** box
   - Currency: **USD**
   - Demo balance: **$100,000**
3. **Submit** and check your email
4. **Save credentials**: Login, Password, Server

## ⚙️ Configuration Files

After running the setup script, you'll have:

### Environment Configuration (`.env`)
```bash
# Primary Broker: IC Markets
MT5_LOGIN=your_icmarkets_login
MT5_PASSWORD=your_icmarkets_password
MT5_SERVER=your_icmarkets_server

# Alternative Brokers
XM_LOGIN=your_xm_login
XM_PASSWORD=your_xm_password
XM_SERVER=your_xm_server

DUKASCOPY_LOGIN=your_dukascopy_login
DUKASCOPY_PASSWORD=your_dukascopy_password
DUKASCOPY_SERVER=your_dukascopy_server
```

### Broker-Specific Configs
- `config/icmarkets_config.json` - IC Markets settings
- `config/xm_config.json` - XM Group settings  
- `config/dukascopy_config.json` - Dukascopy settings

## 🔄 Switching Between Brokers

Use the broker switcher script:

```bash
./scripts/switch_broker.sh
```

This allows you to easily switch between:
1. IC Markets (scalping optimized)
2. XM Group (beginner friendly)
3. Dukascopy (professional grade)

## 🧪 Testing Your Demo Accounts

### Quick Test
```bash
./scripts/test_demo_accounts.sh
```

### Full Integration Test
```bash
# Start MT5 bridge
docker-compose up -d mt5-bridge

# Test connection
curl http://localhost:5004/health

# Run integration tests
python tests/test_mt5_integration.py
```

## 📊 Broker Comparison

| Feature | IC Markets | XM Group | Dukascopy |
|---------|------------|----------|-----------|
| **Demo Duration** | Unlimited | Unlimited | 14 days |
| **Min Spreads** | 0.0 pips | 1.0 pips | 0.1 pips |
| **Execution** | ECN/STP | Market Maker | ECN |
| **Scalping** | ✅ Excellent | ✅ Good | ✅ Excellent |
| **EAs Allowed** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Support** | 24/7 | 24/5 | Business hours |
| **Regulation** | ASIC/CySEC | CySEC/ASIC | FINMA |

## 💡 Pro Tips

### For Scalping Bots
- **Start with IC Markets** - Best spreads and execution
- **Test during London/NY overlap** - Highest liquidity
- **Monitor latency** - Use VPS near broker servers
- **Check commission structures** - Raw spread accounts better for high frequency

### Account Management
- **Keep demos active** - Login at least once per week
- **Test all brokers** - Compare execution and spreads
- **Document performance** - Track which broker works best
- **Renew Dukascopy** - 14-day limit, but renewable

### Risk Management
- **Always start with demo** - Never test with real money
- **Verify settings** - Check lot sizes and leverage
- **Monitor drawdowns** - Set appropriate limits
- **Test edge cases** - Market open/close, news events

## 🚨 Common Issues & Solutions

### Issue: "Invalid Account" Error
**Solution**: 
- Verify login credentials
- Check server name (case sensitive)
- Ensure demo account is still active

### Issue: Connection Timeout
**Solution**:
- Check internet connection
- Try different server (some brokers have multiple)
- Contact broker support

### Issue: Orders Not Executing
**Solution**:
- Check market hours
- Verify symbol names (EURUSD vs EUR/USD)
- Check lot size limits
- Ensure sufficient virtual balance

### Issue: Demo Account Expired
**Solution**:
- **IC Markets/XM**: Unlimited, shouldn't expire
- **Dukascopy**: Create new demo or contact support

## 📞 Broker Support

### IC Markets
- **Live Chat**: 24/7 available
- **Phone**: Multiple international numbers
- **Email**: support@icmarkets.com
- **Languages**: English, Chinese, Spanish, Arabic

### XM Group  
- **Live Chat**: 24/5 multilingual
- **Phone**: +357 25 029 600
- **Email**: support@xm.com
- **Languages**: 30+ languages supported

### Dukascopy Bank
- **Phone**: +41 22 799 4888
- **Email**: info@dukascopy.com
- **Hours**: European business hours
- **Languages**: English, French, German, Russian

## 🔧 Advanced Configuration

### Custom Symbol Mapping
Edit `config/mt5_config.json`:
```json
{
  "symbol_mapping": {
    "EUR/USD": "EURUSD",
    "GBP/USD": "GBPUSD.pro",
    "USD/JPY": "USDJPY#"
  }
}
```

### Latency Optimization
```json
{
  "connection": {
    "retry_attempts": 3,
    "retry_delay": 1000,
    "heartbeat_interval": 30000,
    "market_data_timeout": 5000
  }
}
```

### Multiple Account Testing
```json
{
  "accounts": [
    {
      "name": "icmarkets_raw",
      "login": "${ICMARKETS_LOGIN}",
      "server": "${ICMARKETS_SERVER}",
      "priority": 1
    },
    {
      "name": "xm_standard", 
      "login": "${XM_LOGIN}",
      "server": "${XM_SERVER}",
      "priority": 2
    }
  ]
}
```

## 📈 Performance Monitoring

### Key Metrics to Track
- **Connection uptime** - Should be >99%
- **Order execution time** - Target <100ms
- **Spread variations** - Monitor during different sessions
- **Slippage** - Track actual vs requested prices

### Monitoring Scripts
```bash
# Check connection status
curl http://localhost:5004/health

# Monitor spreads
curl http://localhost:5004/market_data/EURUSD

# Check account info
curl http://localhost:5004/account_info
```

## 🎯 Next Steps

After setup:

1. **✅ Test basic connectivity** with all 3 brokers
2. **✅ Compare execution speeds** during different market sessions  
3. **✅ Run your scalping strategies** on each platform
4. **✅ Monitor performance metrics** for 1-2 weeks
5. **✅ Choose primary broker** based on results
6. **✅ Consider live account** once satisfied with demo performance

---

**🎉 You're now ready to test your MT5 Forex Scalping Bot with professional-grade demo accounts!**

**Need help?** Check the troubleshooting section or contact the respective broker support teams.