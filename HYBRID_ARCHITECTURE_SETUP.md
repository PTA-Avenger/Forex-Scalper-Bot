# Hybrid Architecture Integration Guide

## Overview

This guide explains how to integrate the hybrid MT5 Windows VM + Linux Bot architecture with your existing trading bot system. The integration adds MT5 signal reception capabilities to your current price predictor and MT5 bridge services.

## Architecture Flow

```
MT5 (Windows VM) → MT5 Bridge Service → Price Predictor Service → Gemini AI → Database
```

1. **MT5 Windows VM**: Runs `WindowsVM_SignalSender.mq5` Expert Advisor
2. **MT5 Bridge Service**: Receives signals and forwards to price predictor
3. **Price Predictor Service**: Processes signals with Gemini AI
4. **Database**: Stores signals, decisions, and results (Redis + InfluxDB)

## What Was Updated

### 1. Price Predictor Service (`/python/price_predictor/app.py`)

**Added new endpoint**: `/mt5-signals`
- Receives market signals from MT5 Windows VM
- Processes signals with existing Gemini AI integration
- Stores results in Redis and InfluxDB
- Returns AI decision back to caller

**New Functions**:
- `handle_mt5_signal()`: Main signal reception endpoint
- `process_mt5_signal_with_ai()`: AI processing using existing Gemini predictor
- `store_signal_in_influxdb()`: Data persistence

### 2. MT5 Bridge Service (`/python/mt5_bridge/mt5_bridge.py`)

**Added new endpoint**: `/windows-vm-signals`
- Receives signals from Windows VM
- Forwards signals to price predictor service
- Acts as a routing layer

**New Functions**:
- `handle_windows_vm_signal()`: Signal reception from Windows VM
- `forward_signal_to_predictor()`: Forward to price predictor service

### 3. Docker Compose Configuration

**Updated MT5 Bridge**:
- Added `PRICE_PREDICTOR_URL` environment variable
- Fixed port conflicts
- Added dependency on price predictor service
- Removed profile restriction (now runs by default)

## Setup Instructions

### 1. Deploy Updated Services

```bash
# Start your existing services with the updates
docker-compose up -d

# Check that services are running
docker-compose ps
```

### 2. Install MT5 Expert Advisor

1. Copy `WindowsVM_SignalSender.mq5` to your Windows VM
2. Place it in: `<MT5_DATA_FOLDER>\MQL5\Experts\`
3. Open MetaEditor and compile the EA
4. Attach to chart with these settings:
   - **LinuxBotIP**: Your Linux server IP (e.g., `192.168.1.100`)
   - **BridgePort**: `5005` (MT5 bridge port)
   - **TradingSymbols**: `EURUSD,GBPUSD,USDJPY`
   - **SignalInterval**: `30` seconds
   - **EnableSignals**: `true`

### 3. Configure Network Access

Ensure your Windows VM can reach the Linux services:

```bash
# Test connectivity from Windows VM
curl http://192.168.1.100:5005/health
```

### 4. Monitor the System

**Check logs**:
```bash
# MT5 Bridge logs
docker-compose logs mt5-bridge

# Price Predictor logs  
docker-compose logs price-predictor
```

**API Endpoints**:
- MT5 Bridge Health: `http://localhost:5005/health`
- Price Predictor Health: `http://localhost:5001/health`
- Signal Stats: `http://localhost:5001/cache-stats`

## Data Flow

### Signal Reception Flow
1. MT5 EA sends HTTP POST to `http://<linux-ip>:5005/windows-vm-signals`
2. MT5 Bridge receives signal and forwards to `http://price-predictor:5001/mt5-signals`
3. Price Predictor processes with Gemini AI
4. Results stored in Redis and InfluxDB
5. Response sent back through the chain

### Signal Format

**MT5 to Bridge**:
```json
{
  "signal_type": "market_data",
  "symbol": "EURUSD",
  "timestamp": "2024-01-01 12:00:00",
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
    "atr": 0.00015
  },
  "source": "MT5_Windows_VM"
}
```

**AI Response**:
```json
{
  "status": "success",
  "ai_decision": {
    "action": "BUY",
    "confidence": 0.75,
    "reasoning": "Strong bullish momentum with RSI recovery...",
    "risk_level": "MEDIUM"
  }
}
```

## Integration with Existing Features

### Gemini AI Integration
- Uses your existing `GeminiPredictor` class
- Leverages current model configuration
- Maintains existing caching mechanisms

### Database Integration  
- Signals stored in Redis with existing patterns
- InfluxDB integration ready for time-series data
- Compatible with existing monitoring setup

### Service Discovery
- Uses existing Docker network (`forex-network`)
- Environment-based service URLs
- Health check integration

## Troubleshooting

### Common Issues

1. **Connection Refused**
   ```bash
   # Check if services are running
   docker-compose ps
   
   # Restart MT5 bridge
   docker-compose restart mt5-bridge
   ```

2. **MT5 EA Not Sending Signals**
   - Check MT5 Expert tab for errors
   - Verify network connectivity from Windows VM
   - Ensure WebRequest is allowed in MT5 settings

3. **Gemini API Errors**
   - Verify `GEMINI_API_KEY` in environment
   - Check API rate limits
   - Review price predictor logs

### Monitoring Commands

```bash
# View signal statistics
curl http://localhost:5001/cache-stats

# Check MT5 bridge health
curl http://localhost:5005/health

# Monitor logs in real-time
docker-compose logs -f mt5-bridge price-predictor
```

## Configuration Options

### Environment Variables

Add to your `.env` file:
```bash
# MT5 Windows VM Integration
MT5_WINDOWS_VM_IP=192.168.1.101
PRICE_PREDICTOR_URL=http://price-predictor:5001

# Signal Processing
MIN_CONFIDENCE_THRESHOLD=0.7
SIGNAL_PROCESSING_TIMEOUT=30
```

### MT5 EA Settings

Customize the Expert Advisor parameters:
- `SignalInterval`: How often to send signals (seconds)
- `MinPriceChange`: Minimum price movement to trigger signal
- `TradingSymbols`: Comma-separated list of symbols to monitor

## Next Steps

1. **Test the Integration**: Use the test signal script to verify connectivity
2. **Monitor Performance**: Watch logs and signal statistics
3. **Scale Up**: Add more symbols or reduce signal intervals as needed
4. **Integrate with C++ Engine**: Connect AI decisions to your existing trading engine

## Benefits of This Approach

✅ **Preserves Existing Architecture**: No disruption to current services  
✅ **Leverages Current AI**: Uses your existing Gemini integration  
✅ **Scalable**: Easy to add more MT5 instances or signals  
✅ **Monitorable**: Full integration with existing logging and monitoring  
✅ **Flexible**: Can route signals to different services as needed  

The hybrid architecture extends your current system without breaking existing functionality, giving you the best of both worlds: MT5's market data capabilities and your sophisticated AI trading logic.