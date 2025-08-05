#!/usr/bin/env python3
"""
MetaTrader 5 Bridge Client for Windows
Handles all MT5 API interactions for the Forex Scalping Bot
"""

import os
import sys
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal
import pandas as pd
import numpy as np

# Windows-specific imports
try:
    import MetaTrader5 as mt5
    import win32api
    import win32con
    MT5_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MT5 not available: {e}")
    MT5_AVAILABLE = False

import requests
import redis
from cachetools import TTLCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MT5Config:
    """MT5 Configuration"""
    login: int
    password: str
    server: str
    path: str = None
    timeout: int = 10000
    portable: bool = False

@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    side: str  # buy/sell
    volume: float  # lot size
    order_type: str = "market"
    price: Optional[float] = None
    sl: Optional[float] = None  # stop loss
    tp: Optional[float] = None  # take profit
    deviation: int = 20
    magic: int = 234000
    comment: str = "ForexBot"

@dataclass
class OrderResult:
    """Order result structure"""
    order_id: int
    retcode: int
    deal: int
    volume: float
    price: float
    comment: str
    request_id: int

@dataclass
class Position:
    """Position structure"""
    ticket: int
    symbol: str
    type: int  # 0=buy, 1=sell
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    comment: str
    time: datetime

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    bid: float
    ask: float
    spread: int
    volume: int
    time: datetime

class MT5Client:
    """MetaTrader 5 Client for Windows"""
    
    def __init__(self, config: MT5Config):
        self.config = config
        self.connected = False
        self.account_info = {}
        self.symbols_info = {}
        
        # Cache for market data
        self.price_cache = TTLCache(maxsize=1000, ttl=1)  # 1 second cache
        
        logger.info(f"Initialized MT5 client for server: {self.config.server}")
        
        if not MT5_AVAILABLE:
            logger.error("MT5 not available - install MetaTrader5 package")
            return
    
    def connect(self) -> bool:
        """Connect to MT5 terminal"""
        if not MT5_AVAILABLE:
            logger.error("MT5 package not available")
            return False
            
        try:
            # Initialize MT5 connection
            if self.config.path:
                if not mt5.initialize(path=self.config.path):
                    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize():
                    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            
            # Login to trading account
            if not mt5.login(self.config.login, self.config.password, self.config.server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            self.connected = True
            self.account_info = self._get_account_info()
            
            logger.info(f"✅ Connected to MT5: {self.account_info.get('name', 'Unknown')}")
            logger.info(f"Account: {self.config.login}, Server: {self.config.server}")
            logger.info(f"Balance: ${self.account_info.get('balance', 0):.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    def _get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.connected:
            return {}
            
        try:
            account_info = mt5.account_info()
            if account_info is None:
                return {}
                
            return {
                'login': account_info.login,
                'trade_mode': account_info.trade_mode,
                'name': account_info.name,
                'server': account_info.server,
                'currency': account_info.currency,
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'profit': account_info.profit,
                'trade_allowed': account_info.trade_allowed,
                'trade_expert': account_info.trade_expert
            }
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return {}
            
        self.account_info = self._get_account_info()
        return self.account_info
    
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place a trading order"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return OrderResult(0, -1, 0, 0, 0, "Not connected", 0)
            
        try:
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if order.side.lower() == 'buy' else mt5.ORDER_TYPE_SELL
            
            if order.order_type.lower() == 'market':
                action = mt5.TRADE_ACTION_DEAL
                type_time = mt5.ORDER_TIME_GTC
                price = 0  # Market price
            else:
                action = mt5.TRADE_ACTION_PENDING
                type_time = mt5.ORDER_TIME_GTC
                price = order.price or 0
            
            request = {
                "action": action,
                "symbol": order.symbol,
                "volume": order.volume,
                "type": order_type,
                "price": price,
                "deviation": order.deviation,
                "magic": order.magic,
                "comment": order.comment,
                "type_time": type_time,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if specified
            if order.sl:
                request["sl"] = order.sl
            if order.tp:
                request["tp"] = order.tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logger.error("Order send failed - no result")
                return OrderResult(0, -1, 0, 0, 0, "No result", 0)
            
            logger.info(f"Order result: {result.retcode} - {result.comment}")
            
            return OrderResult(
                order_id=result.order,
                retcode=result.retcode,
                deal=result.deal,
                volume=result.volume,
                price=result.price,
                comment=result.comment,
                request_id=result.request_id
            )
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResult(0, -1, 0, 0, 0, f"Error: {e}", 0)
    
    def close_position(self, ticket: int) -> bool:
        """Close a position by ticket"""
        if not self.connected:
            logger.error("Not connected to MT5")
            return False
            
        try:
            # Get position info
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error(f"Position {ticket} not found")
                return False
            
            position = position[0]
            
            # Prepare close request
            close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": ticket,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close by bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Position {ticket} closed successfully")
                return True
            else:
                logger.error(f"Failed to close position {ticket}: {result.comment if result else 'No result'}")
                return False
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        if not self.connected:
            return []
            
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append(Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    profit=pos.profit,
                    swap=pos.swap,
                    comment=pos.comment,
                    time=datetime.fromtimestamp(pos.time)
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol"""
        if not self.connected:
            return None
            
        # Check cache first
        if symbol in self.price_cache:
            return self.price_cache[symbol]
            
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}")
                return None
            
            market_data = MarketData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                spread=int((tick.ask - tick.bid) / mt5.symbol_info(symbol).point),
                volume=tick.volume,
                time=datetime.fromtimestamp(tick.time)
            )
            
            # Cache the result
            self.price_cache[symbol] = market_data
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = "1H", count: int = 100) -> List[Dict]:
        """Get historical price data"""
        if not self.connected:
            return []
            
        try:
            # Map timeframe string to MT5 constant
            timeframe_map = {
                "1M": mt5.TIMEFRAME_M1,
                "5M": mt5.TIMEFRAME_M5,
                "15M": mt5.TIMEFRAME_M15,
                "30M": mt5.TIMEFRAME_M30,
                "1H": mt5.TIMEFRAME_H1,
                "4H": mt5.TIMEFRAME_H4,
                "1D": mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            if rates is None:
                logger.warning(f"No historical data for {symbol}")
                return []
            
            # Convert to list of dictionaries
            result = []
            for rate in rates:
                result.append({
                    "timestamp": datetime.fromtimestamp(rate['time']).isoformat(),
                    "open": rate['open'],
                    "high": rate['high'],
                    "low": rate['low'],
                    "close": rate['close'],
                    "volume": rate['tick_volume']
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        if not self.connected:
            return []
            
        try:
            symbols = mt5.symbols_get()
            if symbols is None:
                return []
            
            return [symbol.name for symbol in symbols if symbol.visible]
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return []

def create_mt5_client() -> MT5Client:
    """Factory function to create MT5 client"""
    config = MT5Config(
        login=int(os.getenv('MT5_LOGIN', 0)),
        password=os.getenv('MT5_PASSWORD', ''),
        server=os.getenv('MT5_SERVER', ''),
        path=os.getenv('MT5_PATH', None)
    )
    
    return MT5Client(config)

if __name__ == "__main__":
    # Test the client
    print("🧪 Testing MT5 Client...")
    
    if not MT5_AVAILABLE:
        print("❌ MT5 not available - install MetaTrader5 package")
        print("   pip install MetaTrader5")
        sys.exit(1)
    
    client = create_mt5_client()
    
    if client.connect():
        print("✅ Connection successful")
        
        # Test account info
        account = client.get_account_info()
        print(f"Account: {account.get('name', 'Unknown')}")
        print(f"Balance: ${account.get('balance', 0):.2f}")
        print(f"Server: {account.get('server', 'Unknown')}")
        
        # Test symbols
        symbols = client.get_symbols()
        print(f"Available symbols: {len(symbols)}")
        if symbols:
            print(f"First 5: {symbols[:5]}")
        
        # Test market data
        if 'EURUSD' in symbols:
            market_data = client.get_market_data('EURUSD')
            if market_data:
                print(f"EURUSD: {market_data.bid}/{market_data.ask} (spread: {market_data.spread})")
        
        # Test positions
        positions = client.get_open_positions()
        print(f"Open positions: {len(positions)}")
        
        client.disconnect()
        print("✅ Test completed successfully")
    else:
        print("❌ Connection failed")
        print("Make sure:")
        print("1. MT5 terminal is running")
        print("2. Login credentials are correct")
        print("3. Server allows API connections")