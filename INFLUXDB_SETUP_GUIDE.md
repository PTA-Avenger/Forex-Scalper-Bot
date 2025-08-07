# InfluxDB Setup Guide for Trading Bot System

## Overview

This guide will help you set up InfluxDB 2.x for your trading bot system to store time-series market data, AI decisions, and trading metrics.

## 🚀 Quick Start

### 1. Update Environment Variables

Add these to your `.env` file:

```bash
# InfluxDB Configuration
INFLUXDB_PASSWORD=trading_password_123
INFLUXDB_TOKEN=forex-super-secret-token-12345
```

### 2. Start InfluxDB

```bash
# Start InfluxDB and related services
docker-compose up -d influxdb

# Check if InfluxDB is running
docker-compose ps influxdb
```

### 3. Access InfluxDB UI

Open your browser and go to: **http://localhost:8086**

**Default Credentials:**
- **Username**: `admin`
- **Password**: `trading_password_123` (or your custom password)
- **Organization**: `forex-trading-org`
- **Bucket**: `market-data`

## 📊 Configuration Details

### Docker Compose Configuration

Your InfluxDB service is configured with:

```yaml
influxdb:
  image: influxdb:2.7
  environment:
    - DOCKER_INFLUXDB_INIT_MODE=setup
    - DOCKER_INFLUXDB_INIT_USERNAME=admin
    - DOCKER_INFLUXDB_INIT_PASSWORD=trading_password_123
    - DOCKER_INFLUXDB_INIT_ORG=forex-trading-org
    - DOCKER_INFLUXDB_INIT_BUCKET=market-data
    - DOCKER_INFLUXDB_INIT_RETENTION=30d
    - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=forex-super-secret-token-12345
  ports:
    - "8086:8086"
```

### Environment Variables

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `INFLUXDB_PASSWORD` | `trading_password_123` | Admin user password |
| `INFLUXDB_TOKEN` | `forex-super-secret-token-12345` | API authentication token |

## 📈 Data Structure

### Measurements (Tables)

1. **market_signals** - Raw market data from MT5
2. **ai_decisions** - AI analysis results
3. **trade_executions** - Trade execution records
4. **mt5_executions** - MT5 trade confirmations

### market_signals Schema

**Tags:**
- `symbol` (string): Trading pair (e.g., EURUSD)
- `source` (string): Data source (e.g., MT5)

**Fields:**
- `bid` (float): Bid price
- `ask` (float): Ask price
- `spread` (float): Bid-ask spread
- `volume` (integer): Trading volume
- `open` (float): Open price
- `high` (float): High price
- `low` (float): Low price
- `close` (float): Close price
- `indicator_rsi` (float): RSI indicator
- `indicator_ma_fast` (float): Fast moving average
- `indicator_ma_slow` (float): Slow moving average
- `indicator_atr` (float): ATR indicator

### ai_decisions Schema

**Tags:**
- `symbol` (string): Trading pair
- `action` (string): AI decision (BUY/SELL/HOLD)
- `risk_level` (string): Risk assessment (LOW/MEDIUM/HIGH)

**Fields:**
- `confidence` (float): Decision confidence (0.0-1.0)
- `reasoning` (string): AI reasoning text

## 🔧 Setup Steps

### Step 1: Start InfluxDB

```bash
# Start InfluxDB service
docker-compose up -d influxdb

# Wait for it to be ready (about 30-60 seconds)
docker-compose logs -f influxdb
```

### Step 2: Verify Installation

```bash
# Check health
curl http://localhost:8086/ping

# Should return: OK
```

### Step 3: Access Web UI

1. Open browser to http://localhost:8086
2. Login with:
   - **Username**: `admin`
   - **Password**: `trading_password_123`
3. You should see the InfluxDB dashboard

### Step 4: Verify Bucket Creation

In the InfluxDB UI:
1. Go to **Data** → **Buckets**
2. You should see the `market-data` bucket
3. Retention policy should be set to 30 days

### Step 5: Test Data Integration

Start your price predictor service:

```bash
# Start the price predictor with InfluxDB integration
docker-compose up -d price-predictor

# Check logs
docker-compose logs price-predictor
```

## 📊 Querying Data

### Using InfluxDB UI

1. Go to **Data Explorer** in the UI
2. Select your bucket: `market-data`
3. Choose measurement: `market_signals` or `ai_decisions`
4. Build your query and run

### Sample Queries

**Get recent market signals:**
```flux
from(bucket: "market-data")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "market_signals")
  |> filter(fn: (r) => r["symbol"] == "EURUSD")
```

**Get AI decisions with high confidence:**
```flux
from(bucket: "market-data")
  |> range(start: -24h)
  |> filter(fn: (r) => r["_measurement"] == "ai_decisions")
  |> filter(fn: (r) => r["_field"] == "confidence")
  |> filter(fn: (r) => r["_value"] >= 0.7)
```

**Average bid prices by symbol:**
```flux
from(bucket: "market-data")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "market_signals")
  |> filter(fn: (r) => r["_field"] == "bid")
  |> group(columns: ["symbol"])
  |> mean()
```

## 🔌 API Integration

### Python Client Example

```python
from influxdb_client import InfluxDBClient, Point
from datetime import datetime

# Connection
client = InfluxDBClient(
    url="http://localhost:8086",
    token="forex-super-secret-token-12345",
    org="forex-trading-org"
)

# Write data
write_api = client.write_api()
point = Point("market_signals") \
    .tag("symbol", "EURUSD") \
    .tag("source", "MT5") \
    .field("bid", 1.1025) \
    .field("ask", 1.1027) \
    .time(datetime.now())

write_api.write(bucket="market-data", record=point)

# Query data
query_api = client.query_api()
query = '''
from(bucket: "market-data")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_measurement"] == "market_signals")
'''
result = query_api.query(query)
```

### REST API Example

```bash
# Write data point
curl -X POST "http://localhost:8086/api/v2/write?org=forex-trading-org&bucket=market-data" \
  -H "Authorization: Token forex-super-secret-token-12345" \
  -H "Content-Type: text/plain" \
  --data-raw "market_signals,symbol=EURUSD,source=MT5 bid=1.1025,ask=1.1027"

# Query data
curl -X POST "http://localhost:8086/api/v2/query?org=forex-trading-org" \
  -H "Authorization: Token forex-super-secret-token-12345" \
  -H "Content-Type: application/vnd.flux" \
  --data 'from(bucket:"market-data") |> range(start: -1h)'
```

## 📊 Monitoring and Maintenance

### Health Checks

```bash
# Check InfluxDB health
curl http://localhost:8086/health

# Check service status
docker-compose ps influxdb

# View logs
docker-compose logs influxdb
```

### Data Retention

Your data is automatically retained for 30 days. To modify:

1. Go to InfluxDB UI → **Data** → **Buckets**
2. Edit the `market-data` bucket
3. Change retention policy as needed

### Backup

```bash
# Create backup
docker exec forex-influxdb influx backup /tmp/backup

# Copy backup from container
docker cp forex-influxdb:/tmp/backup ./influxdb-backup
```

## 🔧 Troubleshooting

### Common Issues

1. **InfluxDB won't start**
   ```bash
   # Check logs
   docker-compose logs influxdb
   
   # Remove volumes and restart
   docker-compose down -v
   docker-compose up -d influxdb
   ```

2. **Can't connect to InfluxDB**
   ```bash
   # Check if port is accessible
   curl http://localhost:8086/ping
   
   # Verify container is running
   docker-compose ps influxdb
   ```

3. **Authentication errors**
   - Verify your token in `.env` file
   - Check organization and bucket names
   - Ensure token has proper permissions

4. **Price predictor can't write to InfluxDB**
   ```bash
   # Check price predictor logs
   docker-compose logs price-predictor
   
   # Verify InfluxDB environment variables
   docker-compose exec price-predictor env | grep INFLUX
   ```

### Performance Tuning

For high-frequency trading data:

1. **Increase memory limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 2G
   ```

2. **Adjust retention policy** for your data volume needs

3. **Use batch writes** in your applications

## 🎯 Integration with Trading System

### Current Integration Points

1. **Price Predictor Service**: Automatically stores MT5 signals and AI decisions
2. **MT5 Bridge**: Can be extended to store additional trade data
3. **Future Grafana Integration**: Ready for visualization dashboards

### Data Flow

```
MT5 Windows VM → Price Predictor → InfluxDB
                      ↓
               AI Decision Storage
                      ↓
                 Redis Cache ← → Grafana Dashboard
```

## 🚀 Next Steps

1. **Test the Setup**: Send some MT5 signals and verify data appears in InfluxDB
2. **Create Dashboards**: Set up Grafana for visualization
3. **Monitoring**: Set up alerts for data ingestion issues
4. **Scaling**: Consider clustering for high-volume trading

Your InfluxDB setup is now ready to handle time-series data for your trading bot system!