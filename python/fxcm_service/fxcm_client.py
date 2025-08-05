#!/usr/bin/env python3
"""
FXCM Trading Client
Handles all FXCM API interactions for the Forex Scalping Bot
"""

import os
import sys
import time
import json
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from decimal import Decimal

import pandas as pd
import numpy as np
# import fxcmpy  # Using REST API instead
import requests
from websocket import WebSocketApp
import redis
from cachetools import TTLCache
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FXCMConfig:
    """FXCM configuration settings"""
    access_token: str
    server_type: str = "demo"  # "demo" or "real"
    log_level: str = "info"
    log_file: str = "fxcm_client.log"
    
    # Rate limiting
    max_requests_per_minute: int = 300
    max_orders_per_minute: int = 100
    
    # WebSocket settings
    reconnect_attempts: int = 5
    heartbeat_interval: int = 30
    
    # Cache settings
    price_cache_ttl: int = 5  # seconds
    account_cache_ttl: int = 10  # seconds

@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    side: str  # "buy" or "sell"
    amount: float
    order_type: str = "market"  # "market", "limit", "stop"
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: str = "GTC"  # "GTC", "IOC", "FOK"
    comment: str = "Gemini AI Bot"

@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    executed_price: Optional[float] = None
    executed_amount: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = None

@dataclass
class Position:
    """Position information"""
    position_id: str
    symbol: str
    side: str
    amount: float
    open_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime

class FXCMClient:
    """Main FXCM trading client"""
    
    def __init__(self, config: FXCMConfig):
        """Initialize FXCM client"""
        self.config = config
        self.connection = None
        self.ws_connection = None
        self.is_connected = False
        self.is_streaming = False
        
        # Caches
        self.price_cache = TTLCache(maxsize=1000, ttl=config.price_cache_ttl)
        self.account_cache = TTLCache(maxsize=100, ttl=config.account_cache_ttl)
        
        # Redis for pub/sub
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD'),
            decode_responses=True
        )
        
        # Event handlers
        self.order_handlers = []
        self.price_handlers = []
        self.position_handlers = []
        
        # Threading
        self.ws_thread = None
        self.heartbeat_thread = None
        
        # Initialize connection
        self._connect()
    
    def _connect(self) -> bool:
        """Connect to FXCM API"""
        try:
            logger.info("Connecting to FXCM API...")
            
            # Initialize FXCM connection
            self.connection = fxcmpy.fxcmpy(
                access_token=self.config.access_token,
                log_level=self.config.log_level,
                server=self.config.server_type,
                log_file=self.config.log_file
            )
            
            # Test connection
            if self.connection.is_connected():
                self.is_connected = True
                logger.info("Successfully connected to FXCM")
                
                # Get account info
                account_info = self.get_account_info()
                logger.info(f"Account Balance: {account_info.get('balance', 'N/A')}")
                logger.info(f"Account Currency: {account_info.get('currency', 'N/A')}")
                
                # Start WebSocket connection for real-time data
                self._start_websocket()
                
                return True
            else:
                logger.error("Failed to connect to FXCM")
                return False
                
        except Exception as e:
            logger.error(f"FXCM connection error: {e}")
            self.is_connected = False
            return False
    
    def _start_websocket(self):
        """Start WebSocket connection for real-time data"""
        try:
            logger.info("Starting FXCM WebSocket connection...")
            
            # Start WebSocket in separate thread
            self.ws_thread = threading.Thread(target=self._websocket_worker, daemon=True)
            self.ws_thread.start()
            
            # Start heartbeat thread
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_worker, daemon=True)
            self.heartbeat_thread.start()
            
            self.is_streaming = True
            logger.info("WebSocket connection started")
            
        except Exception as e:
            logger.error(f"WebSocket startup error: {e}")
    
    def _websocket_worker(self):
        """WebSocket worker thread"""
        while self.is_connected:
            try:
                # Subscribe to price updates for major pairs
                major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD']
                
                for symbol in major_pairs:
                    if self.connection and self.connection.is_connected():
                        # Subscribe to real-time prices
                        self.connection.subscribe_market_data(symbol, self._on_price_update)
                
                time.sleep(1)  # Prevent tight loop
                
            except Exception as e:
                logger.error(f"WebSocket worker error: {e}")
                time.sleep(5)  # Wait before retry
    
    def _heartbeat_worker(self):
        """Heartbeat worker to maintain connection"""
        while self.is_connected:
            try:
                if self.connection and self.connection.is_connected():
                    # Send heartbeat by checking account status
                    self.get_account_info()
                    logger.debug("Heartbeat sent")
                else:
                    logger.warning("Connection lost, attempting reconnect...")
                    self._reconnect()
                
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                time.sleep(self.config.heartbeat_interval)
    
    def _reconnect(self):
        """Reconnect to FXCM"""
        for attempt in range(self.config.reconnect_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{self.config.reconnect_attempts}")
            
            try:
                if self.connection:
                    self.connection.close()
                
                time.sleep(5)  # Wait before reconnect
                
                if self._connect():
                    logger.info("Reconnection successful")
                    return True
                    
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        logger.error("All reconnection attempts failed")
        self.is_connected = False
        return False
    
    def _on_price_update(self, data):
        """Handle real-time price updates"""
        try:
            symbol = data.get('Symbol', '')
            bid = float(data.get('Bid', 0))
            ask = float(data.get('Ask', 0))
            
            if symbol and bid > 0 and ask > 0:
                market_data = MarketData(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    spread=ask - bid,
                    timestamp=datetime.now()
                )
                
                # Update cache
                self.price_cache[symbol] = market_data
                
                # Publish to Redis
                self.redis_client.publish(
                    f"fxcm:price:{symbol}",
                    json.dumps(asdict(market_data), default=str)
                )
                
                # Call registered handlers
                for handler in self.price_handlers:
                    try:
                        handler(market_data)
                    except Exception as e:
                        logger.error(f"Price handler error: {e}")
                        
        except Exception as e:
            logger.error(f"Price update error: {e}")
    
    @sleep_and_retry
    @limits(calls=300, period=60)  # Rate limiting
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            if not self.is_connected:
                raise Exception("Not connected to FXCM")
            
            # Check cache first
            cache_key = "account_info"
            if cache_key in self.account_cache:
                return self.account_cache[cache_key]
            
            # Get from FXCM
            account_info = self.connection.get_accounts()
            
            if not account_info.empty:
                account_data = {
                    'account_id': account_info.iloc[0]['accountId'],
                    'balance': float(account_info.iloc[0]['balance']),
                    'equity': float(account_info.iloc[0]['equity']),
                    'used_margin': float(account_info.iloc[0]['usableMargin']),
                    'free_margin': float(account_info.iloc[0]['usableMargin']),
                    'currency': account_info.iloc[0]['currency'],
                    'timestamp': datetime.now()
                }
                
                # Cache result
                self.account_cache[cache_key] = account_data
                return account_data
            else:
                raise Exception("No account information available")
                
        except Exception as e:
            logger.error(f"Get account info error: {e}")
            return {}
    
    @sleep_and_retry
    @limits(calls=100, period=60)  # Rate limiting for orders
    def place_order(self, order_request: OrderRequest) -> OrderResult:
        """Place a trading order"""
        try:
            if not self.is_connected:
                raise Exception("Not connected to FXCM")
            
            logger.info(f"Placing {order_request.side} order for {order_request.amount} {order_request.symbol}")
            
            # Prepare order parameters
            order_params = {
                'symbol': order_request.symbol,
                'is_buy': order_request.side.lower() == 'buy',
                'amount': order_request.amount,
                'time_in_force': order_request.time_in_force,
                'order_type': order_request.order_type.upper()
            }
            
            # Add conditional parameters
            if order_request.price:
                order_params['rate'] = order_request.price
            
            if order_request.stop_loss:
                order_params['stop'] = order_request.stop_loss
                
            if order_request.take_profit:
                order_params['limit'] = order_request.take_profit
            
            # Execute order
            if order_request.order_type.lower() == 'market':
                result = self.connection.create_market_buy_order(**order_params) if order_request.side.lower() == 'buy' else self.connection.create_market_sell_order(**order_params)
            else:
                result = self.connection.create_entry_order(**order_params)
            
            if result and not result.empty:
                order_result = OrderResult(
                    success=True,
                    order_id=str(result.iloc[0]['orderId']),
                    trade_id=str(result.iloc[0].get('tradeId', '')),
                    executed_price=float(result.iloc[0].get('open', 0)),
                    executed_amount=float(result.iloc[0].get('amountK', 0)) * 1000,
                    timestamp=datetime.now()
                )
                
                # Publish order update
                self.redis_client.publish(
                    "fxcm:order:placed",
                    json.dumps(asdict(order_result), default=str)
                )
                
                # Send webhook notification
                self._send_webhook("order_placed", asdict(order_result))
                
                logger.info(f"Order placed successfully: {order_result.order_id}")
                return order_result
            else:
                raise Exception("Order execution failed - no result returned")
                
        except Exception as e:
            error_msg = f"Order placement error: {e}"
            logger.error(error_msg)
            
            order_result = OrderResult(
                success=False,
                error_message=error_msg,
                timestamp=datetime.now()
            )
            
            # Send webhook notification for failed order
            self._send_webhook("order_failed", asdict(order_result))
            
            return order_result
    
    def close_position(self, position_id: str) -> OrderResult:
        """Close a specific position"""
        try:
            if not self.is_connected:
                raise Exception("Not connected to FXCM")
            
            logger.info(f"Closing position: {position_id}")
            
            # Get position details
            positions = self.get_open_positions()
            target_position = None
            
            for position in positions:
                if position.position_id == position_id:
                    target_position = position
                    break
            
            if not target_position:
                raise Exception(f"Position {position_id} not found")
            
            # Close position
            result = self.connection.close_trade(trade_id=position_id, amount=target_position.amount)
            
            if result and not result.empty:
                order_result = OrderResult(
                    success=True,
                    order_id=str(result.iloc[0]['orderId']),
                    trade_id=position_id,
                    executed_price=float(result.iloc[0].get('close', 0)),
                    executed_amount=target_position.amount,
                    timestamp=datetime.now()
                )
                
                # Publish position close
                self.redis_client.publish(
                    "fxcm:position:closed",
                    json.dumps(asdict(order_result), default=str)
                )
                
                # Send webhook notification
                self._send_webhook("position_closed", asdict(order_result))
                
                logger.info(f"Position closed successfully: {position_id}")
                return order_result
            else:
                raise Exception("Position close failed - no result returned")
                
        except Exception as e:
            error_msg = f"Position close error: {e}"
            logger.error(error_msg)
            
            return OrderResult(
                success=False,
                error_message=error_msg,
                timestamp=datetime.now()
            )
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        try:
            if not self.is_connected:
                return []
            
            positions_df = self.connection.get_open_positions()
            positions = []
            
            if not positions_df.empty:
                for _, row in positions_df.iterrows():
                    position = Position(
                        position_id=str(row['tradeId']),
                        symbol=row['currency'],
                        side='buy' if row['isBuy'] else 'sell',
                        amount=float(row['amountK']) * 1000,
                        open_price=float(row['open']),
                        current_price=float(row['close']),
                        unrealized_pnl=float(row['grossPL']),
                        realized_pnl=0.0,  # FXCM doesn't provide this in open positions
                        timestamp=datetime.now()
                    )
                    positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol"""
        try:
            # Check cache first
            if symbol in self.price_cache:
                return self.price_cache[symbol]
            
            if not self.is_connected:
                return None
            
            # Get from FXCM
            prices = self.connection.get_prices(symbol)
            
            if not prices.empty:
                latest = prices.iloc[-1]
                market_data = MarketData(
                    symbol=symbol,
                    bid=float(latest['bidclose']),
                    ask=float(latest['askclose']),
                    spread=float(latest['askclose']) - float(latest['bidclose']),
                    timestamp=datetime.now()
                )
                
                # Update cache
                self.price_cache[symbol] = market_data
                return market_data
            
            return None
            
        except Exception as e:
            logger.error(f"Get market data error for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
        """Get historical OHLCV data"""
        try:
            if not self.is_connected:
                return pd.DataFrame()
            
            logger.info(f"Getting historical data for {symbol}, timeframe: {timeframe}, periods: {periods}")
            
            # Convert timeframe to FXCM format
            fxcm_timeframe = self._convert_timeframe(timeframe)
            
            # Get historical data
            data = self.connection.get_candles(symbol, period=fxcm_timeframe, number=periods)
            
            if not data.empty:
                # Rename columns to standard format
                data = data.rename(columns={
                    'bidopen': 'open',
                    'bidhigh': 'high',
                    'bidlow': 'low',
                    'bidclose': 'close'
                })
                
                # Add volume (FXCM doesn't provide volume, so we'll use tick count as proxy)
                data['volume'] = data.get('tickqty', 1000)  # Default volume if not available
                
                # Reset index to have timestamp as column
                data = data.reset_index()
                data['timestamp'] = data['date']
                
                return data[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Get historical data error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert standard timeframe to FXCM format"""
        timeframe_map = {
            '1m': 'm1',
            '5m': 'm5',
            '15m': 'm15',
            '30m': 'm30',
            '1h': 'H1',
            '4h': 'H4',
            '1d': 'D1',
            '1w': 'W1',
            '1M': 'M1'
        }
        return timeframe_map.get(timeframe, 'H1')
    
    def _send_webhook(self, event_type: str, data: Dict[str, Any]):
        """Send webhook notification"""
        try:
            webhook_url = os.getenv('WEBHOOK_URL')
            if not webhook_url:
                return
            
            webhook_data = {
                'source': 'fxcm',
                'event_type': event_type,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            response = requests.post(
                webhook_url,
                json=webhook_data,
                timeout=5,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.debug(f"Webhook sent successfully: {event_type}")
            else:
                logger.warning(f"Webhook failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Webhook error: {e}")
    
    def add_order_handler(self, handler):
        """Add order event handler"""
        self.order_handlers.append(handler)
    
    def add_price_handler(self, handler):
        """Add price update handler"""
        self.price_handlers.append(handler)
    
    def add_position_handler(self, handler):
        """Add position update handler"""
        self.position_handlers.append(handler)
    
    def disconnect(self):
        """Disconnect from FXCM"""
        try:
            logger.info("Disconnecting from FXCM...")
            
            self.is_connected = False
            self.is_streaming = False
            
            if self.connection:
                self.connection.close()
            
            # Wait for threads to finish
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=5)
            
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=5)
            
            logger.info("Disconnected from FXCM")
            
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.disconnect()

# Factory function for easy initialization
def create_fxcm_client(access_token: str, server_type: str = "demo") -> FXCMClient:
    """Create and initialize FXCM client"""
    config = FXCMConfig(
        access_token=access_token,
        server_type=server_type
    )
    return FXCMClient(config)

if __name__ == "__main__":
    # Test the FXCM client
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    access_token = os.getenv('FXCM_ACCESS_TOKEN')
    if not access_token:
        print("Please set FXCM_ACCESS_TOKEN environment variable")
        sys.exit(1)
    
    # Create client
    client = create_fxcm_client(access_token, "demo")
    
    if client.is_connected:
        print("✅ FXCM client connected successfully")
        
        # Test account info
        account_info = client.get_account_info()
        print(f"Account Balance: {account_info.get('balance', 'N/A')}")
        
        # Test market data
        market_data = client.get_market_data("EUR/USD")
        if market_data:
            print(f"EUR/USD: Bid={market_data.bid}, Ask={market_data.ask}, Spread={market_data.spread:.5f}")
        
        # Test historical data
        historical_data = client.get_historical_data("EUR/USD", "1h", 10)
        print(f"Historical data points: {len(historical_data)}")
        
        # Keep running for a bit to test real-time data
        print("Testing real-time data for 30 seconds...")
        time.sleep(30)
        
        client.disconnect()
        print("✅ Test completed")
    else:
        print("❌ Failed to connect to FXCM")
        sys.exit(1)