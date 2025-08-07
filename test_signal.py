#!/usr/bin/env python3
"""
Test script to simulate MT5 signals for testing the trading bot system
"""

import asyncio
import json
import random
import time
from datetime import datetime, timezone
from typing import Dict, Any

import aiohttp


class MT5SignalSimulator:
    def __init__(self, bot_url: str = "http://localhost:8080"):
        self.bot_url = bot_url
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD"]
        self.current_prices = {
            "EURUSD": 1.10250,
            "GBPUSD": 1.27500,
            "USDJPY": 149.50,
            "USDCHF": 0.89750,
            "AUDUSD": 0.67250
        }

    def generate_realistic_signal(self, symbol: str) -> Dict[str, Any]:
        """Generate a realistic market signal"""
        
        # Get current price and simulate small movements
        current_price = self.current_prices[symbol]
        
        # Simulate price movement (±0.1% max)
        price_change = random.uniform(-0.001, 0.001)
        new_price = current_price * (1 + price_change)
        self.current_prices[symbol] = new_price
        
        # Calculate bid/ask spread (typical forex spreads)
        spread_pips = random.uniform(0.5, 2.0)  # 0.5 to 2.0 pips
        if symbol == "USDJPY":
            spread = spread_pips * 0.01  # JPY pairs
        else:
            spread = spread_pips * 0.00001  # Other pairs
        
        bid = new_price - spread / 2
        ask = new_price + spread / 2
        
        # Generate OHLC data
        high = new_price + random.uniform(0, spread * 2)
        low = new_price - random.uniform(0, spread * 2)
        open_price = new_price + random.uniform(-spread, spread)
        
        # Generate technical indicators
        rsi = random.uniform(20, 80)
        ma_fast = new_price * random.uniform(0.999, 1.001)
        ma_slow = new_price * random.uniform(0.998, 1.002)
        bb_upper = new_price * 1.002
        bb_lower = new_price * 0.998
        atr = spread * random.uniform(5, 15)
        
        return {
            "signal_type": "market_data",
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bid": round(bid, 5),
            "ask": round(ask, 5),
            "spread": round(spread, 5),
            "volume": random.randint(100, 10000),
            "ohlc": {
                "open": round(open_price, 5),
                "high": round(high, 5),
                "low": round(low, 5),
                "close": round(new_price, 5)
            },
            "indicators": {
                "rsi": round(rsi, 2),
                "ma_fast": round(ma_fast, 5),
                "ma_slow": round(ma_slow, 5),
                "bollinger_upper": round(bb_upper, 5),
                "bollinger_lower": round(bb_lower, 5),
                "atr": round(atr, 5)
            },
            "source": "MT5_Signal_Simulator",
            "version": "1.0"
        }

    async def send_signal(self, session: aiohttp.ClientSession, signal: Dict[str, Any]) -> bool:
        """Send signal to trading bot"""
        try:
            url = f"{self.bot_url}/api/signals"
            async with session.post(url, json=signal) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Signal sent for {signal['symbol']}: {result.get('ai_decision', {}).get('action', 'N/A')}")
                    return True
                else:
                    print(f"❌ Failed to send signal for {signal['symbol']}: {response.status}")
                    return False
        except Exception as e:
            print(f"❌ Error sending signal for {signal['symbol']}: {str(e)}")
            return False

    async def simulate_trading_session(self, duration_minutes: int = 10, signals_per_minute: int = 2):
        """Simulate a trading session with multiple signals"""
        print(f"🚀 Starting {duration_minutes}-minute trading simulation...")
        print(f"📊 Sending {signals_per_minute} signals per minute")
        print(f"💱 Symbols: {', '.join(self.symbols)}")
        print("-" * 50)
        
        async with aiohttp.ClientSession() as session:
            end_time = time.time() + (duration_minutes * 60)
            signal_count = 0
            successful_signals = 0
            
            while time.time() < end_time:
                # Select random symbol
                symbol = random.choice(self.symbols)
                
                # Generate and send signal
                signal = self.generate_realistic_signal(symbol)
                if await self.send_signal(session, signal):
                    successful_signals += 1
                
                signal_count += 1
                
                # Wait before next signal
                wait_time = 60 / signals_per_minute
                await asyncio.sleep(wait_time)
                
                # Progress update every 10 signals
                if signal_count % 10 == 0:
                    elapsed = (time.time() - (end_time - duration_minutes * 60)) / 60
                    print(f"📈 Progress: {signal_count} signals sent in {elapsed:.1f} minutes")
        
        print("-" * 50)
        print(f"✅ Simulation completed!")
        print(f"📊 Total signals: {signal_count}")
        print(f"✅ Successful: {successful_signals}")
        print(f"❌ Failed: {signal_count - successful_signals}")
        print(f"📈 Success rate: {(successful_signals/signal_count)*100:.1f}%")

    async def send_single_test_signal(self, symbol: str = "EURUSD"):
        """Send a single test signal"""
        print(f"🧪 Sending test signal for {symbol}...")
        
        signal = self.generate_realistic_signal(symbol)
        
        async with aiohttp.ClientSession() as session:
            success = await self.send_signal(session, signal)
            
        if success:
            print("✅ Test signal sent successfully!")
        else:
            print("❌ Test signal failed!")
        
        return success

    async def check_bot_health(self):
        """Check if the trading bot is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.bot_url}/api/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print("✅ Trading Bot is healthy!")
                        print(f"📊 Status: {health_data.get('status', 'Unknown')}")
                        print(f"🕐 Timestamp: {health_data.get('timestamp', 'Unknown')}")
                        return True
                    else:
                        print(f"❌ Trading Bot health check failed: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Cannot connect to trading bot: {str(e)}")
            return False


async def main():
    """Main function with interactive menu"""
    simulator = MT5SignalSimulator()
    
    print("🤖 MT5 Signal Simulator")
    print("=" * 30)
    
    # Check bot health first
    if not await simulator.check_bot_health():
        print("\n❌ Trading bot is not available. Please start the bot first:")
        print("   docker-compose up -d")
        return
    
    while True:
        print("\n📋 Options:")
        print("1. Send single test signal")
        print("2. Run short simulation (2 minutes)")
        print("3. Run medium simulation (10 minutes)")
        print("4. Run long simulation (30 minutes)")
        print("5. Custom simulation")
        print("6. Check bot health")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            symbol = input("Enter symbol (default: EURUSD): ").strip() or "EURUSD"
            await simulator.send_single_test_signal(symbol.upper())
        elif choice == "2":
            await simulator.simulate_trading_session(2, 3)
        elif choice == "3":
            await simulator.simulate_trading_session(10, 2)
        elif choice == "4":
            await simulator.simulate_trading_session(30, 1)
        elif choice == "5":
            try:
                duration = int(input("Enter duration in minutes: "))
                signals_per_min = int(input("Enter signals per minute: "))
                await simulator.simulate_trading_session(duration, signals_per_min)
            except ValueError:
                print("❌ Invalid input. Please enter numbers only.")
        elif choice == "6":
            await simulator.check_bot_health()
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    asyncio.run(main())