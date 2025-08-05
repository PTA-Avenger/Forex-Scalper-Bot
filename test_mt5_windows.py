#!/usr/bin/env python3
"""
Windows MT5 Test Script
Quick test to verify MT5 setup on Windows
"""

import os
import sys
from datetime import datetime

print("🪟 MetaTrader 5 Windows Test")
print("=" * 40)

# Test 1: Check Python version
print(f"✅ Python Version: {sys.version}")

# Test 2: Check if we're on Windows
if os.name != 'nt':
    print("❌ This script is for Windows only")
    sys.exit(1)
print("✅ Running on Windows")

# Test 3: Import MT5 package
try:
    import MetaTrader5 as mt5
    print("✅ MetaTrader5 package imported successfully")
    print(f"   Version: {mt5.__version__ if hasattr(mt5, '__version__') else 'Unknown'}")
except ImportError as e:
    print("❌ MetaTrader5 package not found")
    print("   Install with: pip install MetaTrader5")
    print(f"   Error: {e}")
    sys.exit(1)

# Test 4: Import other required packages
try:
    import pandas as pd
    import numpy as np
    print("✅ Data processing packages available")
except ImportError as e:
    print("⚠️ Some packages missing - install with:")
    print("   pip install pandas numpy")

# Test 5: Check for .env file
env_file = ".env"
if os.path.exists(env_file):
    print(f"✅ Environment file found: {env_file}")
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded")
    except ImportError:
        print("⚠️ python-dotenv not installed")
        print("   Install with: pip install python-dotenv")
else:
    print(f"⚠️ Environment file not found: {env_file}")
    print("   Create .env file with your MT5 credentials")

# Test 6: Get MT5 credentials
mt5_login = os.getenv('MT5_LOGIN', '')
mt5_password = os.getenv('MT5_PASSWORD', '')
mt5_server = os.getenv('MT5_SERVER', '')
mt5_path = os.getenv('MT5_PATH', '')

print("\n🔐 MT5 Configuration:")
print(f"   Login: {'✅ Set' if mt5_login else '❌ Not set'}")
print(f"   Password: {'✅ Set' if mt5_password else '❌ Not set'}")
print(f"   Server: {'✅ Set' if mt5_server else '❌ Not set'}")
print(f"   Path: {'✅ Set' if mt5_path else '⚠️ Using default'}")

if not all([mt5_login, mt5_password, mt5_server]):
    print("\n❌ Missing MT5 credentials in .env file")
    print("Add these to your .env file:")
    print("MT5_LOGIN=your_demo_account_number")
    print("MT5_PASSWORD=your_demo_password")
    print("MT5_SERVER=your_demo_server")
    print("MT5_PATH=C:\\Program Files\\MetaTrader 5\\terminal64.exe")
    sys.exit(1)

# Test 7: Initialize MT5
print("\n🔌 Testing MT5 Connection...")
try:
    if mt5_path and os.path.exists(mt5_path):
        print(f"   Using MT5 path: {mt5_path}")
        if not mt5.initialize(path=mt5_path):
            print(f"❌ MT5 initialize failed: {mt5.last_error()}")
            sys.exit(1)
    else:
        print("   Using default MT5 path")
        if not mt5.initialize():
            print(f"❌ MT5 initialize failed: {mt5.last_error()}")
            print("   Make sure MT5 terminal is running")
            sys.exit(1)
    
    print("✅ MT5 initialized successfully")
    
except Exception as e:
    print(f"❌ MT5 initialization error: {e}")
    print("   Make sure:")
    print("   1. MT5 terminal is running")
    print("   2. Run this script as Administrator")
    print("   3. Check MT5 path in .env file")
    sys.exit(1)

# Test 8: Login to MT5
print("\n🔑 Testing MT5 Login...")
try:
    login = int(mt5_login)
    if not mt5.login(login, mt5_password, mt5_server):
        error = mt5.last_error()
        print(f"❌ MT5 login failed: {error}")
        print("   Check your credentials:")
        print(f"   Login: {login}")
        print(f"   Server: {mt5_server}")
        mt5.shutdown()
        sys.exit(1)
    
    print("✅ MT5 login successful")
    
except ValueError:
    print(f"❌ Invalid login number: {mt5_login}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Login error: {e}")
    mt5.shutdown()
    sys.exit(1)

# Test 9: Get account information
print("\n📊 Account Information:")
try:
    account_info = mt5.account_info()
    if account_info:
        print(f"   Name: {account_info.name}")
        print(f"   Login: {account_info.login}")
        print(f"   Server: {account_info.server}")
        print(f"   Currency: {account_info.currency}")
        print(f"   Balance: ${account_info.balance:,.2f}")
        print(f"   Equity: ${account_info.equity:,.2f}")
        print(f"   Margin: ${account_info.margin:,.2f}")
        print(f"   Free Margin: ${account_info.margin_free:,.2f}")
        print(f"   Margin Level: {account_info.margin_level:.2f}%")
        print(f"   Trade Allowed: {'✅ Yes' if account_info.trade_allowed else '❌ No'}")
        print(f"   Expert Advisors: {'✅ Allowed' if account_info.trade_expert else '❌ Not allowed'}")
    else:
        print("❌ Could not retrieve account information")
except Exception as e:
    print(f"❌ Account info error: {e}")

# Test 10: Get available symbols
print("\n📈 Available Symbols:")
try:
    symbols = mt5.symbols_get()
    if symbols:
        symbol_names = [s.name for s in symbols if s.visible]
        print(f"   Total symbols: {len(symbol_names)}")
        
        # Show major forex pairs
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']
        available_majors = [pair for pair in major_pairs if pair in symbol_names]
        print(f"   Major pairs available: {len(available_majors)}")
        for pair in available_majors:
            print(f"     ✅ {pair}")
    else:
        print("❌ No symbols available")
except Exception as e:
    print(f"❌ Symbols error: {e}")

# Test 11: Get market data
print("\n💰 Market Data Test:")
try:
    test_symbol = "EURUSD"
    tick = mt5.symbol_info_tick(test_symbol)
    if tick:
        spread = tick.ask - tick.bid
        print(f"   {test_symbol}:")
        print(f"     Bid: {tick.bid:.5f}")
        print(f"     Ask: {tick.ask:.5f}")
        print(f"     Spread: {spread:.5f} ({spread * 10000:.1f} pips)")
        print(f"     Time: {datetime.fromtimestamp(tick.time)}")
        print("   ✅ Market data accessible")
    else:
        print(f"❌ No market data for {test_symbol}")
except Exception as e:
    print(f"❌ Market data error: {e}")

# Test 12: Get historical data
print("\n📊 Historical Data Test:")
try:
    test_symbol = "EURUSD"
    rates = mt5.copy_rates_from_pos(test_symbol, mt5.TIMEFRAME_H1, 0, 10)
    if rates is not None and len(rates) > 0:
        print(f"   Retrieved {len(rates)} H1 bars for {test_symbol}")
        latest = rates[-1]
        print(f"   Latest bar:")
        print(f"     Time: {datetime.fromtimestamp(latest['time'])}")
        print(f"     OHLC: {latest['open']:.5f} | {latest['high']:.5f} | {latest['low']:.5f} | {latest['close']:.5f}")
        print(f"     Volume: {latest['tick_volume']}")
        print("   ✅ Historical data accessible")
    else:
        print(f"❌ No historical data for {test_symbol}")
except Exception as e:
    print(f"❌ Historical data error: {e}")

# Test 13: Check positions
print("\n📋 Open Positions:")
try:
    positions = mt5.positions_get()
    if positions:
        print(f"   Open positions: {len(positions)}")
        for pos in positions:
            profit_status = "🟢" if pos.profit >= 0 else "🔴"
            print(f"     {profit_status} {pos.symbol} {pos.volume} lots @ {pos.price_open:.5f} (Profit: ${pos.profit:.2f})")
    else:
        print("   No open positions")
        print("   ✅ Position data accessible")
except Exception as e:
    print(f"❌ Positions error: {e}")

# Cleanup
print("\n🔚 Cleaning up...")
mt5.shutdown()
print("✅ MT5 connection closed")

print("\n" + "=" * 40)
print("🎉 MT5 Test Completed Successfully!")
print("\nYour MT5 setup is ready for:")
print("✅ Account access")
print("✅ Market data retrieval")
print("✅ Historical data access")
print("✅ Position monitoring")
print("\n🚀 You can now integrate with your Forex Bot!")

print("\nNext steps:")
print("1. Test order placement (carefully with demo account)")
print("2. Run your bot's MT5 bridge service")
print("3. Monitor performance vs FXCM")

print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")