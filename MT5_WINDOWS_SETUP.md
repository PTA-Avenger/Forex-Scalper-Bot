# 🪟 MetaTrader 5 Setup Guide for Windows

Complete guide to set up MT5 on Windows for testing with your Forex Scalping Bot.

## 📋 Prerequisites

- **Windows 10/11** (64-bit recommended)
- **Python 3.11+** installed
- **Admin privileges** for installation
- **Internet connection** for downloads
- **Demo trading account** (recommended for testing)

## 🚀 Step-by-Step Setup

### **Step 1: Download MetaTrader 5**

#### Option A: Official MT5 (Recommended)
```
🔗 Direct Download: https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
```

#### Option B: Broker-Specific MT5
Choose a broker for better integration:

| Broker | Download Link | Benefits |
|--------|---------------|----------|
| **IC Markets** | https://www.icmarkets.com/global/en/trading-platforms/metatrader-5 | Low spreads, good API |
| **Pepperstone** | https://pepperstone.com/en/trading-platforms/metatrader-5 | Fast execution |
| **XM** | https://www.xm.com/mt5 | Good for beginners |
| **FXCM** | https://www.fxcm.com/platforms/metatrader-5/ | Professional features |

### **Step 2: Install MetaTrader 5**

1. **Download** the MT5 installer (mt5setup.exe)
2. **Run as Administrator** (right-click → "Run as administrator")
3. **Follow installation wizard**:
   - Accept license agreement
   - Choose installation directory (default: `C:\Program Files\MetaTrader 5`)
   - Wait for installation to complete
4. **Launch MT5** from desktop shortcut

### **Step 3: Create Demo Account**

1. **Open MT5** application
2. **File** → **Open an Account**
3. **Select your broker** or "MetaQuotes Demo"
4. **Fill account details**:
   - Choose "Demo Account"
   - Set balance (e.g., $10,000)
   - Select leverage (1:100 recommended)
   - Choose base currency (USD)
5. **Save credentials** (login, password, server)

### **Step 4: Enable Algorithmic Trading**

1. In MT5, go to **Tools** → **Options**
2. **Expert Advisors** tab
3. **Check these options**:
   - ✅ Allow algorithmic trading
   - ✅ Allow DLL imports
   - ✅ Allow imports of external experts
4. **Apply** and **OK**

### **Step 5: Install Python Dependencies**

Open **Command Prompt as Administrator** and run:

```cmd
# Install MT5 Python package
pip install MetaTrader5

# Install Windows-specific dependencies
pip install pywin32

# Install other required packages
pip install pandas numpy requests flask python-dotenv
```

### **Step 6: Configure Environment Variables**

Create a `.env` file in your project directory:

```env
# MT5 Configuration
MT5_LOGIN=your_demo_account_number
MT5_PASSWORD=your_demo_password
MT5_SERVER=your_demo_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# Example for IC Markets Demo
# MT5_LOGIN=12345678
# MT5_PASSWORD=DemoPass123
# MT5_SERVER=ICMarkets-Demo
```

### **Step 7: Test MT5 Connection**

Create a test file `test_mt5.py`:

```python
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

# Get credentials from environment
login = int(os.getenv('MT5_LOGIN', 0))
password = os.getenv('MT5_PASSWORD', '')
server = os.getenv('MT5_SERVER', '')

print(f"Testing MT5 connection...")
print(f"Login: {login}")
print(f"Server: {server}")

# Initialize MT5
if not mt5.initialize():
    print("❌ MT5 initialize failed")
    quit()

# Login
if not mt5.login(login, password, server):
    print("❌ MT5 login failed")
    print(f"Error: {mt5.last_error()}")
    mt5.shutdown()
    quit()

print("✅ MT5 connection successful!")

# Get account info
account_info = mt5.account_info()
if account_info:
    print(f"Account: {account_info.name}")
    print(f"Balance: ${account_info.balance:.2f}")
    print(f"Equity: ${account_info.equity:.2f}")
    print(f"Server: {account_info.server}")

# Get symbols
symbols = mt5.symbols_get()
print(f"Available symbols: {len(symbols)}")

# Test market data
if symbols:
    symbol = "EURUSD"
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"{symbol}: Bid={tick.bid}, Ask={tick.ask}")

mt5.shutdown()
print("✅ Test completed successfully!")
```

Run the test:
```cmd
python test_mt5.py
```

### **Step 8: Integration with Your Bot**

Update your bot's `.env` file:

```env
# Add MT5 configuration to your existing .env
MT5_LOGIN=your_demo_account_number
MT5_PASSWORD=your_demo_password
MT5_SERVER=your_demo_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# Keep existing FXCM settings for comparison
FXCM_ACCESS_TOKEN=demo_token_replace_later
FXCM_SERVER_TYPE=demo
```

### **Step 9: Start MT5 Bridge Service**

If running on Windows, you can run the MT5 bridge directly:

```cmd
# Navigate to your bot directory
cd path\to\your\forex-bot

# Install Python dependencies
pip install -r python\mt5_bridge\requirements.txt

# Run MT5 bridge
python python\mt5_bridge\mt5_client.py
```

## 🔧 Common Issues & Solutions

### **Issue 1: "MT5 initialize failed"**
**Solutions:**
- Ensure MT5 terminal is running
- Run Python script as Administrator
- Check MT5 path in environment variables
- Restart MT5 terminal

### **Issue 2: "Login failed"**
**Solutions:**
- Verify login credentials
- Check server name (case-sensitive)
- Ensure demo account is active
- Try different server if available

### **Issue 3: "DLL import error"**
**Solutions:**
- Enable "Allow DLL imports" in MT5 settings
- Run MT5 as Administrator
- Check Windows Defender/Antivirus settings

### **Issue 4: "MetaTrader5 module not found"**
**Solutions:**
```cmd
pip uninstall MetaTrader5
pip install MetaTrader5==5.0.45
```

### **Issue 5: "Connection timeout"**
**Solutions:**
- Check internet connection
- Verify broker server status
- Try different server endpoint
- Disable VPN if using one

## 📊 Popular Broker Settings

### **IC Markets Demo**
```env
MT5_LOGIN=your_demo_login
MT5_PASSWORD=your_demo_password
MT5_SERVER=ICMarkets-Demo
```

### **Pepperstone Demo**
```env
MT5_LOGIN=your_demo_login
MT5_PASSWORD=your_demo_password
MT5_SERVER=Pepperstone-Demo
```

### **XM Demo**
```env
MT5_LOGIN=your_demo_login
MT5_PASSWORD=your_demo_password
MT5_SERVER=XMGlobal-Demo
```

## 🧪 Testing Checklist

- [ ] MT5 installed and running
- [ ] Demo account created
- [ ] Algorithmic trading enabled
- [ ] Python packages installed
- [ ] Environment variables configured
- [ ] Connection test successful
- [ ] Market data accessible
- [ ] Account info retrieved

## 🚀 Next Steps

Once MT5 is working:

1. **Test the bot integration**:
   ```cmd
   python python\mt5_bridge\mt5_client.py
   ```

2. **Start the full bot** (from Linux server):
   ```bash
   # Enable MT5 profile
   make up-mt5
   ```

3. **Monitor both brokers**:
   - FXCM Service: http://localhost:5004
   - MT5 Bridge: http://localhost:5005

## 📈 Performance Comparison

| Feature | FXCM | MT5 |
|---------|------|-----|
| **Setup Complexity** | Medium | High |
| **API Speed** | 50-80ms | 10-30ms |
| **Spreads** | 0.1-0.8 pips | Varies by broker |
| **Execution** | REST API | Direct terminal |
| **Reliability** | High | Very High |
| **Broker Options** | Limited | Many |

## 🔐 Security Notes

- **Never use real money** for testing
- **Keep demo credentials secure**
- **Use VPS for production** (not personal Windows)
- **Regular password changes**
- **Monitor account activity**

## 📞 Support Resources

- **MT5 Documentation**: https://www.mql5.com/en/docs/integration/python_metatrader5
- **Python MT5 Package**: https://pypi.org/project/MetaTrader5/
- **MQL5 Community**: https://www.mql5.com/en/forum
- **Broker Support**: Contact your chosen broker

---

**🎉 You're now ready to test your forex bot with MetaTrader 5 on Windows!**

The MT5 integration provides direct terminal access with minimal latency, making it ideal for high-frequency trading strategies.