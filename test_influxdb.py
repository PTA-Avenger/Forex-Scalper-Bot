#!/usr/bin/env python3
"""
Test script to verify InfluxDB setup and integration
"""

import os
import sys
from datetime import datetime
import json

def test_influxdb_connection():
    """Test basic InfluxDB connectivity"""
    try:
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS
        
        # InfluxDB connection settings
        influx_url = os.getenv('INFLUXDB_URL', 'http://localhost:8086')
        influx_token = os.getenv('INFLUXDB_TOKEN', 'forex-super-secret-token-12345')
        influx_org = os.getenv('INFLUXDB_ORG', 'forex-trading-org')
        influx_bucket = os.getenv('INFLUXDB_BUCKET', 'market-data')
        
        print("🔍 Testing InfluxDB Connection...")
        print(f"   URL: {influx_url}")
        print(f"   Org: {influx_org}")
        print(f"   Bucket: {influx_bucket}")
        print()
        
        # Create client
        client = InfluxDBClient(url=influx_url, token=influx_token, org=influx_org)
        
        # Test connection
        health = client.health()
        print(f"✅ InfluxDB Health: {health.status}")
        print(f"   Version: {health.version}")
        print()
        
        return client, influx_org, influx_bucket
        
    except ImportError:
        print("❌ InfluxDB client not installed. Run: pip install influxdb-client")
        return None, None, None
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return None, None, None

def test_write_sample_data(client, org, bucket):
    """Test writing sample trading data"""
    try:
        print("📝 Testing Data Write...")
        
        write_api = client.write_api(write_options=SYNCHRONOUS)
        
        # Create sample market signal
        signal_point = (
            Point("market_signals")
            .tag("symbol", "EURUSD")
            .tag("source", "TEST")
            .field("bid", 1.10250)
            .field("ask", 1.10255)
            .field("spread", 0.00005)
            .field("volume", 1000)
            .field("open", 1.10245)
            .field("high", 1.10260)
            .field("low", 1.10240)
            .field("close", 1.10250)
            .field("indicator_rsi", 65.5)
            .field("indicator_ma_fast", 1.10248)
            .field("indicator_ma_slow", 1.10235)
            .time(datetime.now())
        )
        
        # Create sample AI decision
        ai_point = (
            Point("ai_decisions")
            .tag("symbol", "EURUSD")
            .tag("action", "BUY")
            .tag("risk_level", "MEDIUM")
            .field("confidence", 0.75)
            .field("reasoning", "Test AI decision - strong bullish momentum")
            .time(datetime.now())
        )
        
        # Write points
        write_api.write(bucket=bucket, org=org, record=[signal_point, ai_point])
        print("✅ Sample data written successfully")
        print("   - Market signal for EURUSD")
        print("   - AI decision (BUY, confidence: 0.75)")
        print()
        
    except Exception as e:
        print(f"❌ Write test failed: {str(e)}")

def test_query_data(client, org, bucket):
    """Test querying data from InfluxDB"""
    try:
        print("📊 Testing Data Query...")
        
        query_api = client.query_api()
        
        # Query recent market signals
        query = f'''
        from(bucket: "{bucket}")
          |> range(start: -1h)
          |> filter(fn: (r) => r["_measurement"] == "market_signals")
          |> filter(fn: (r) => r["symbol"] == "EURUSD")
          |> limit(n: 5)
        '''
        
        result = query_api.query(query, org=org)
        
        signal_count = 0
        for table in result:
            for record in table.records:
                signal_count += 1
                if signal_count == 1:  # Show first record details
                    print(f"✅ Found market signals:")
                    print(f"   Symbol: {record.values.get('symbol')}")
                    print(f"   Field: {record.get_field()}")
                    print(f"   Value: {record.get_value()}")
                    print(f"   Time: {record.get_time()}")
        
        if signal_count == 0:
            print("⚠️  No market signals found (this is normal for a fresh setup)")
        else:
            print(f"   Total records found: {signal_count}")
        print()
        
        # Query AI decisions
        ai_query = f'''
        from(bucket: "{bucket}")
          |> range(start: -1h)
          |> filter(fn: (r) => r["_measurement"] == "ai_decisions")
          |> limit(n: 5)
        '''
        
        ai_result = query_api.query(ai_query, org=org)
        
        ai_count = 0
        for table in ai_result:
            for record in table.records:
                ai_count += 1
                if ai_count == 1:  # Show first record details
                    print(f"✅ Found AI decisions:")
                    print(f"   Symbol: {record.values.get('symbol')}")
                    print(f"   Action: {record.values.get('action')}")
                    print(f"   Field: {record.get_field()}")
                    print(f"   Value: {record.get_value()}")
        
        if ai_count == 0:
            print("⚠️  No AI decisions found (this is normal for a fresh setup)")
        else:
            print(f"   Total AI decision records: {ai_count}")
        print()
        
    except Exception as e:
        print(f"❌ Query test failed: {str(e)}")

def test_price_predictor_endpoint():
    """Test the price predictor MT5 signals endpoint"""
    try:
        import requests
        
        print("🧪 Testing Price Predictor Integration...")
        
        # Sample MT5 signal data
        test_signal = {
            "signal_type": "market_data",
            "symbol": "EURUSD",
            "timestamp": datetime.now().isoformat(),
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
            "source": "TEST_SCRIPT"
        }
        
        # Send to price predictor
        response = requests.post(
            'http://localhost:5001/mt5-signals',
            json=test_signal,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Price Predictor endpoint working")
            print(f"   AI Decision: {result.get('ai_decision', {}).get('action', 'N/A')}")
            print(f"   Confidence: {result.get('ai_decision', {}).get('confidence', 'N/A')}")
        else:
            print(f"❌ Price Predictor endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
        
        print()
        
    except ImportError:
        print("⚠️  Requests library not available, skipping endpoint test")
    except requests.exceptions.ConnectionError:
        print("⚠️  Price Predictor service not running, skipping endpoint test")
    except Exception as e:
        print(f"❌ Endpoint test failed: {str(e)}")

def main():
    """Main test function"""
    print("=" * 60)
    print("           InfluxDB Setup Test")
    print("=" * 60)
    print()
    
    # Test 1: Connection
    client, org, bucket = test_influxdb_connection()
    
    if not client:
        print("❌ Cannot continue without InfluxDB connection")
        sys.exit(1)
    
    # Test 2: Write data
    test_write_sample_data(client, org, bucket)
    
    # Test 3: Query data
    test_query_data(client, org, bucket)
    
    # Test 4: Price predictor integration
    test_price_predictor_endpoint()
    
    # Cleanup
    client.close()
    
    print("=" * 60)
    print("🎉 InfluxDB Test Complete!")
    print()
    print("Next steps:")
    print("1. Start your MT5 EA to send real signals")
    print("2. Monitor data in InfluxDB UI: http://localhost:8086")
    print("3. Set up Grafana dashboards for visualization")
    print("=" * 60)

if __name__ == '__main__':
    main()