# Trading Bot Hybrid Architecture

A sophisticated trading system that combines MT5 on Windows VM for market data collection with a Linux-based AI bot for intelligent decision making, all monitored through comprehensive analytics.

## 🏗️ Architecture Overview

```
┌─────────────────┐    HTTP/JSON    ┌─────────────────┐    InfluxDB    ┌─────────────────┐
│   MT5 Windows   │ ──────────────> │   Linux Bot     │ ────────────> │   InfluxDB      │
│      VM         │    Signals      │   (AI Engine)   │   Analytics   │   (Database)    │
└─────────────────┘                 └─────────────────┘               └─────────────────┘
        │                                    │                                │
        │                                    │                                │
        v                                    v                                v
┌─────────────────┐                 ┌─────────────────┐               ┌─────────────────┐
│   Broker API    │                 │   Gemini AI     │               │    Grafana      │
│   (Market Data) │                 │   (Decisions)   │               │  (Dashboard)    │
└─────────────────┘                 └─────────────────┘               └─────────────────┘
```

## 🔧 Components

### 1. **MT5 Windows VM**
- **Purpose**: Market data collection and signal generation
- **File**: `MT5_SignalSender.mq5`
- **Features**:
  - Real-time market data monitoring
  - Technical indicator calculations (RSI, MA, Bollinger Bands, ATR)
  - HTTP signal transmission to Linux bot
  - Trade execution reporting

### 2. **Linux Trading Bot**
- **Purpose**: AI-powered decision making and execution
- **File**: `linux_trading_bot.py`
- **Features**:
  - RESTful API for signal reception
  - Gemini AI integration for trade analysis
  - Risk management and position sizing
  - Comprehensive logging and audit trails

### 3. **InfluxDB Database**
- **Purpose**: Time-series data storage and analytics
- **Features**:
  - Market signal storage
  - AI decision tracking
  - Trade execution logging
  - Performance analytics

### 4. **Grafana Dashboard**
- **Purpose**: Real-time monitoring and visualization
- **Features**:
  - Live trading metrics
  - AI confidence tracking
  - Risk assessment visualization
  - Historical performance analysis

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Windows VM with MT5 installed
- Gemini AI API key
- Network connectivity between VMs

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd trading-bot-hybrid
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your actual values:
   # - GEMINI_API_KEY
   # - MT5_VM_IP
   ```

3. **Install MT5 Expert Advisor**
   - Copy `MT5_SignalSender.mq5` to your MT5 `Experts` folder
   - Compile in MetaEditor
   - Attach to chart with proper settings

4. **Start Services**
   ```bash
   docker-compose up -d
   ```

## 📊 Service URLs

- **Trading Bot API**: http://localhost:8080
- **InfluxDB UI**: http://localhost:8086
- **Grafana Dashboard**: http://localhost:3000
- **Redis**: localhost:6379

## 🔐 Default Credentials

- **InfluxDB**: admin / trading123!
- **Grafana**: admin / trading123!
- **Redis**: trading123!

## 📡 API Endpoints

### Trading Bot API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/signals` | Receive market signals from MT5 |
| POST | `/api/trade-execution` | Receive trade execution updates |
| GET | `/api/statistics` | Get trading statistics |

### Example Signal Payload
```json
{
  "signal_type": "market_data",
  "symbol": "EURUSD",
  "timestamp": "2024-01-01T12:00:00Z",
  "bid": 1.10250,
  "ask": 1.10255,
  "spread": 0.00005,
  "volume": 1000,
  "ohlc": {
    "open": 1.10245,
    "high": 1.10260,
    "low": 1.10240,
    "close": 1.10250
  },
  "indicators": {
    "rsi": 65.5,
    "ma_fast": 1.10248,
    "ma_slow": 1.10235,
    "bollinger_upper": 1.10270,
    "bollinger_lower": 1.10220,
    "atr": 0.00015
  },
  "source": "MT5_Windows_VM",
  "version": "1.0"
}
```

## 🤖 AI Decision Making

The system uses Google's Gemini AI to analyze market signals and make trading decisions:

### Input Analysis
- Technical indicators (RSI, Moving Averages, Bollinger Bands, ATR)
- Price action and momentum
- Market volatility and liquidity
- Risk factors and spread analysis

### Decision Output
```json
{
  "action": "BUY|SELL|HOLD",
  "confidence": 0.85,
  "reasoning": "Strong bullish momentum with RSI oversold recovery...",
  "risk_level": "MEDIUM",
  "stop_loss": 1.10200,
  "take_profit": 1.10350,
  "position_size": 0.01,
  "time_horizon": "SHORT"
}
```

## 📈 Monitoring and Analytics

### InfluxDB Measurements

1. **market_signals**: Raw market data from MT5
2. **ai_decisions**: AI analysis results
3. **trade_executions**: Simulated/actual trade executions
4. **mt5_executions**: Real MT5 trade confirmations

### Grafana Dashboards

- **Real-time Signals**: Live market data visualization
- **AI Performance**: Decision accuracy and confidence tracking
- **Risk Metrics**: Risk assessment and position monitoring
- **Trade History**: Execution history and performance analysis

## ⚙️ Configuration

### MT5 Expert Advisor Settings

```mql5
input string   LinuxBotURL = "http://192.168.1.100:8080";
input string   APIEndpoint = "/api/signals";
input int      SignalInterval = 60;  // seconds
input double   MinPriceChange = 0.0010;
input bool     EnableSignals = true;
input string   TradingSymbols = "EURUSD,GBPUSD,USDJPY";
```

### Linux Bot Configuration

```python
# Environment Variables
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-token
GEMINI_API_KEY=your-api-key
BOT_PORT=8080
MIN_CONFIDENCE_THRESHOLD=0.7
```

## 🔒 Security Considerations

- API key management through environment variables
- Network isolation using Docker networks
- Authentication tokens for InfluxDB access
- Encrypted communication between components
- Input validation and sanitization

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check service status
   docker-compose ps
   
   # View logs
   docker-compose logs trading-bot
   ```

2. **MT5 Signal Not Received**
   - Verify network connectivity between VMs
   - Check MT5 EA logs in MetaTrader
   - Validate Linux bot IP in MT5 settings

3. **AI API Errors**
   - Verify Gemini API key
   - Check API rate limits
   - Monitor bot logs for error details

4. **Database Connection Issues**
   - Verify InfluxDB is running
   - Check connection credentials
   - Validate network connectivity

### Log Locations

- **Trading Bot**: `./logs/trading_bot.log`
- **Docker Logs**: `docker-compose logs [service]`
- **MT5 Logs**: MT5 Experts tab

## 🔄 Data Flow

1. **Signal Generation**: MT5 monitors market and generates signals
2. **Signal Transmission**: HTTP POST to Linux bot API
3. **AI Analysis**: Gemini AI processes signal data
4. **Decision Storage**: Results stored in InfluxDB
5. **Execution Logic**: High-confidence decisions trigger actions
6. **Monitoring**: Grafana visualizes all metrics

## 📝 Development

### Adding New Indicators

1. **MT5 Side**: Update `CalculateIndicators()` function
2. **Linux Side**: Update signal processing logic
3. **Database**: Add new fields to InfluxDB schema
4. **Dashboard**: Update Grafana visualizations

### Extending AI Logic

1. **Prompt Engineering**: Modify AI prompts in `create_ai_prompt()`
2. **Decision Logic**: Update `execute_decision()` for new actions
3. **Risk Management**: Enhance risk assessment algorithms

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📞 Support

For issues and questions:
- Create GitHub issues
- Check troubleshooting section
- Review logs for error details

---

**⚠️ Disclaimer**: This system is for educational purposes. Always test thoroughly before using with real money. Trading involves risk of loss.