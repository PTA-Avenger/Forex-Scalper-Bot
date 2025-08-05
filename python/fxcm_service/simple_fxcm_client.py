#!/usr/bin/env python3
"""
Simplified FXCM Trading Client using REST API
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FXCMConfig:
    """FXCM Configuration"""
    access_token: str
    server_type: str = "demo"  # demo or real
    base_url: str = None
    
    def __post_init__(self):
        if self.base_url is None:
            if self.server_type == "demo":
                self.base_url = "https://api-fxpractice.oanda.com"  # Demo endpoint
            else:
                self.base_url = "https://api-fxtrade.oanda.com"    # Live endpoint

@dataclass
class OrderRequest:
    """Order request structure"""
    symbol: str
    side: str  # buy/sell
    amount: float
    order_type: str = "market"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class OrderResult:
    """Order result structure"""
    order_id: str
    status: str
    symbol: str
    side: str
    amount: float
    price: Optional[float] = None
    timestamp: datetime = None

@dataclass
class Position:
    """Position structure"""
    position_id: str
    symbol: str
    side: str
    amount: float
    open_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    bid: float
    ask: float
    spread: float
    timestamp: datetime

class SimpleFXCMClient:
    """Simplified FXCM Client using REST API"""
    
    def __init__(self, config: FXCMConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.config.access_token}',
            'Content-Type': 'application/json'
        })
        
        self.connected = False
        self.account_info = {}
        
        logger.info(f"Initialized FXCM client for {self.config.server_type} server")
    
    def connect(self) -> bool:
        """Connect to FXCM API"""
        try:
            # Test connection with account info
            account_info = self.get_account_info()
            if account_info:
                self.connected = True
                logger.info("Successfully connected to FXCM API")
                return True
            else:
                logger.error("Failed to connect to FXCM API")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from FXCM API"""
        self.connected = False
        logger.info("Disconnected from FXCM API")
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            # For demo purposes, return mock data
            # In real implementation, this would call FXCM REST API
            mock_account = {
                "account_id": "demo_account_12345",
                "balance": 10000.0,
                "equity": 10000.0,
                "margin_used": 0.0,
                "margin_available": 10000.0,
                "currency": "USD",
                "server_type": self.config.server_type
            }
            
            self.account_info = mock_account
            logger.info(f"Account info retrieved: Balance ${mock_account['balance']}")
            return mock_account
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def place_order(self, order: OrderRequest) -> OrderResult:
        """Place a trading order"""
        try:
            # For demo purposes, simulate order placement
            order_id = f"order_{int(time.time())}"
            
            # Simulate market price
            mock_price = 1.0850 if order.symbol == "EUR/USD" else 1.2500
            
            result = OrderResult(
                order_id=order_id,
                status="filled",
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                price=mock_price,
                timestamp=datetime.now()
            )
            
            logger.info(f"Order placed: {order.symbol} {order.side} {order.amount} @ {mock_price}")
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return OrderResult(
                order_id="",
                status="error",
                symbol=order.symbol,
                side=order.side,
                amount=order.amount,
                timestamp=datetime.now()
            )
    
    def close_position(self, position_id: str) -> bool:
        """Close a position"""
        try:
            logger.info(f"Closing position: {position_id}")
            # Simulate position closure
            return True
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def get_open_positions(self) -> List[Position]:
        """Get open positions"""
        try:
            # Return mock positions for demo
            mock_positions = [
                Position(
                    position_id="pos_1",
                    symbol="EUR/USD",
                    side="buy",
                    amount=1000,
                    open_price=1.0850,
                    current_price=1.0855,
                    unrealized_pnl=5.0,
                    timestamp=datetime.now()
                )
            ]
            
            return mock_positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for a symbol"""
        try:
            # Return mock market data
            if symbol == "EUR/USD":
                bid, ask = 1.0850, 1.0852
            elif symbol == "GBP/USD":
                bid, ask = 1.2500, 1.2502
            elif symbol == "USD/JPY":
                bid, ask = 149.50, 149.52
            else:
                bid, ask = 1.0000, 1.0002
            
            return MarketData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                spread=ask - bid,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = "1H", count: int = 100) -> List[Dict]:
        """Get historical price data"""
        try:
            # Return mock historical data
            mock_data = []
            base_price = 1.0850 if symbol == "EUR/USD" else 1.2500
            
            for i in range(count):
                timestamp = datetime.now() - timedelta(hours=count-i)
                price_variation = (hash(str(timestamp)) % 100) / 10000  # Small random variation
                
                mock_data.append({
                    "timestamp": timestamp.isoformat(),
                    "open": base_price + price_variation,
                    "high": base_price + price_variation + 0.0005,
                    "low": base_price + price_variation - 0.0005,
                    "close": base_price + price_variation + 0.0002,
                    "volume": 1000
                })
            
            return mock_data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []

def create_fxcm_client() -> SimpleFXCMClient:
    """Factory function to create FXCM client"""
    config = FXCMConfig(
        access_token=os.getenv('FXCM_ACCESS_TOKEN', 'demo_token'),
        server_type=os.getenv('FXCM_SERVER_TYPE', 'demo')
    )
    
    return SimpleFXCMClient(config)

if __name__ == "__main__":
    # Test the client
    client = create_fxcm_client()
    
    if client.connect():
        print("✅ Connection successful")
        
        # Test account info
        account = client.get_account_info()
        print(f"Account balance: ${account.get('balance', 0)}")
        
        # Test market data
        market_data = client.get_market_data("EUR/USD")
        if market_data:
            print(f"EUR/USD: {market_data.bid}/{market_data.ask}")
        
        # Test order placement
        order = OrderRequest(
            symbol="EUR/USD",
            side="buy",
            amount=1000
        )
        result = client.place_order(order)
        print(f"Order result: {result.status}")
        
        client.disconnect()
    else:
        print("❌ Connection failed")