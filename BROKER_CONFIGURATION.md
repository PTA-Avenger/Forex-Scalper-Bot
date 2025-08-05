# Multi-Broker Configuration Guide

Your Forex Scalping Bot now supports multiple broker integrations after resolving the merge conflict.

## 🏗️ Current Architecture

### Primary Broker: FXCM
- **Service**: `fxcm-service` (Port 5004)
- **Status**: Always enabled
- **Features**: Professional spreads, high API limits, real-time streaming
- **Configuration**: Requires `FXCM_ACCESS_TOKEN` and `FXCM_SERVER_TYPE`

### Alternative Broker: MT5 Bridge
- **Service**: `mt5-bridge` (Port 5005)
- **Status**: Optional (enabled with `--profile mt5`)
- **Features**: MetaTrader 5 integration, multiple broker support
- **Configuration**: Requires MT5 credentials and terminal path

## 🚀 Deployment Options

### Option 1: FXCM Only (Default)
```bash
# Start with FXCM as primary broker
make up
# or
docker-compose up -d
```

**Services Started:**
- C++ Trading Engine (8080)
- FXCM Service (5004)
- Gemini AI Service (5001)
- Signal Processor (5006)
- React Dashboard (3000)
- Supporting services (Redis, PostgreSQL, etc.)

### Option 2: FXCM + MT5 (Dual Broker)
```bash
# Start with both brokers
make up-mt5
# or
docker-compose --profile mt5 up -d
```

**Additional Service:**
- MT5 Bridge (5005)

### Option 3: All Brokers
```bash
# Future-proof for additional brokers
make up-all-brokers
```

## 🔧 Configuration

### Required Environment Variables

#### FXCM Configuration (Always Required)
```env
FXCM_ACCESS_TOKEN=your_fxcm_access_token_here
FXCM_SERVER_TYPE=demo  # or 'real'
```

#### MT5 Configuration (Optional)
```env
MT5_LOGIN=your_mt5_login_number
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
MT5_PATH=/path/to/mt5/terminal
```

#### AI Configuration
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-pro
```

## 📊 Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| C++ Engine | 8080 | Main trading engine API |
| React Dashboard | 3000 | Web interface |
| Gemini AI | 5001 | Price prediction & analysis |
| Sentiment Analyzer | 5002 | Market sentiment |
| **FXCM Service** | **5004** | **Primary broker** |
| **MT5 Bridge** | **5005** | **Alternative broker** |
| Signal Processor | 5006 | Trading signals |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache |
| InfluxDB | 8086 | Time-series data |
| Prometheus | 9090 | Monitoring |
| Grafana | 3001 | Dashboards |

## 🧪 Testing

### Test FXCM Integration
```bash
make test-fxcm
```

### Test All Services
```bash
make test-all
```

### Health Checks
```bash
# Check all services
make health

# Check specific services
curl http://localhost:5004/health  # FXCM
curl http://localhost:5005/health  # MT5 (if enabled)
```

## 🔄 Broker Selection Logic

The C++ engine can be configured to use different brokers:

### Environment Variables for Broker Selection
```env
# Primary broker (default: fxcm)
PRIMARY_BROKER=fxcm
PRIMARY_BROKER_URL=http://fxcm-service:5003

# Optional backup broker
BACKUP_BROKER=mt5
BACKUP_BROKER_URL=http://mt5-bridge:5000

# Load balancing ratio (optional)
BROKER_SPLIT_RATIO=80  # 80% FXCM, 20% MT5
```

### C++ Code Example
```cpp
// Dynamic broker selection
std::unique_ptr<BrokerInterface> createBroker(const std::string& type) {
    if (type == "fxcm") {
        return std::make_unique<FXCMBroker>("http://fxcm-service:5003");
    } else if (type == "mt5") {
        return std::make_unique<MT5Broker>("http://mt5-bridge:5000");
    }
    return nullptr;
}
```

## 🛠️ Troubleshooting

### Port Conflicts Resolved
- **Signal Processor**: Moved from 5003 → 5006 to avoid FXCM conflict
- **MT5 Bridge**: Uses 5005 to avoid conflicts
- **FXCM Service**: Uses 5004 (external) → 5003 (internal)

### Common Issues

#### FXCM Service Won't Start
```bash
# Check FXCM credentials
docker-compose logs fxcm-service

# Verify environment variables
docker-compose exec fxcm-service env | grep FXCM
```

#### MT5 Service Won't Start
```bash
# Check MT5 credentials
docker-compose logs mt5-bridge

# Verify MT5 terminal path
docker-compose exec mt5-bridge ls -la $MT5_PATH
```

#### Port Already in Use
```bash
# Check what's using the port
netstat -tulpn | grep :5004

# Stop conflicting services
docker-compose down
```

## 📈 Performance Comparison

| Broker | Spreads | Speed | API Limits | Complexity |
|--------|---------|-------|------------|------------|
| **FXCM** | 0.1-0.8 pips | 50-80ms | 300 req/min | Low |
| **MT5** | Varies by broker | 30-100ms | Varies | Medium |

## 🔐 Security Notes

### FXCM Security
- API tokens can be regenerated
- Demo vs Real server separation
- Rate limiting built-in

### MT5 Security
- Direct terminal connection
- Encrypted credentials
- Local file system access required

## 🎯 Recommendations

### For Most Users
1. **Start with FXCM only** (`make up`)
2. **Test thoroughly** with demo account
3. **Add MT5 later** if needed for specific brokers

### For Advanced Users
1. **Use both brokers** for redundancy
2. **Implement broker selection logic** in C++ engine
3. **Monitor performance** of both connections

### Production Deployment
1. **Use real FXCM account** with proper credentials
2. **Configure monitoring** for both brokers
3. **Set up failover logic** between brokers

## 📚 Additional Resources

- [FXCM Migration Guide](FXCM_MIGRATION_GUIDE.md)
- [Architecture Overview](ARCHITECTURE.md)
- [API Documentation](docs/API.md)

---

**The merge conflict has been successfully resolved with a multi-broker architecture that supports both FXCM (primary) and MT5 (optional) integrations.** 🎉