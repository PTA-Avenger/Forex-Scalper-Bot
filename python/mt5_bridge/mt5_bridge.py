#!/usr/bin/env python3
"""
MetaTrader 5 Python Bridge Service

This service provides a bridge between the C++ trading engine and MetaTrader 5
using the MetaTrader5 Python library. It handles authentication, market data,
and order execution through MT5.
"""

import json
import logging
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import signal
import os

try:
    import MetaTrader5 as mt5
    import pandas as pd
    import numpy as np
    from flask import Flask, request, jsonify
    import requests
except ImportError as e:
    print(f"Required dependencies not installed: {e}")
    print("Please install: pip install MetaTrader5 pandas numpy flask requests")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mt5_bridge.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('MT5Bridge')

class MT5Bridge:
    """MetaTrader 5 Bridge Service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.account_info = None
        self.symbols_info = {}
        self.market_data_thread = None
        self.should_stop = threading.Event()
        
        # Flask app for HTTP API
        self.app = Flask(__name__)
        self.setup_routes()
        
        # WebSocket callback URL for real-time updates
        self.callback_url = config.get('callback_url', 'http://localhost:8080/mt5/callback')
        
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({
                'status': 'healthy',
                'connected': self.connected,
                'mt5_version': mt5.version() if self.connected else None
            })
        
        @self.app.route('/connect', methods=['POST'])
        def connect():
            data = request.get_json()
            result = self.connect_to_mt5(
                login=data.get('login'),
                password=data.get('password'),
                server=data.get('server')
            )
            return jsonify({'success': result, 'connected': self.connected})
        
        @self.app.route('/disconnect', methods=['POST'])
        def disconnect():
            result = self.disconnect_from_mt5()
            return jsonify({'success': result, 'connected': self.connected})
        
        @self.app.route('/account_info', methods=['GET'])
        def get_account_info():
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            account_info = mt5.account_info()
            if account_info is None:
                return jsonify({'error': 'Failed to get account info'}), 500
            
            return jsonify(account_info._asdict())
        
        @self.app.route('/symbols', methods=['GET'])
        def get_symbols():
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            symbols = mt5.symbols_get()
            if symbols is None:
                return jsonify({'error': 'Failed to get symbols'}), 500
            
            return jsonify([symbol._asdict() for symbol in symbols])
        
        @self.app.route('/symbol_info/<symbol>', methods=['GET'])
        def get_symbol_info(symbol):
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return jsonify({'error': f'Symbol {symbol} not found'}), 404
            
            return jsonify(symbol_info._asdict())
        
        @self.app.route('/market_data/<symbol>', methods=['GET'])
        def get_market_data(symbol):
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return jsonify({'error': f'No tick data for {symbol}'}), 404
            
            return jsonify({
                'symbol': symbol,
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'time': tick.time,
                'flags': tick.flags,
                'volume_real': tick.volume_real
            })
        
        @self.app.route('/historical_data/<symbol>', methods=['GET'])
        def get_historical_data(symbol):
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            timeframe = request.args.get('timeframe', 'M1')
            count = int(request.args.get('count', 1000))
            
            # Convert timeframe string to MT5 constant
            tf_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
            
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            if rates is None:
                return jsonify({'error': f'No historical data for {symbol}'}), 404
            
            # Convert to list of dictionaries
            data = []
            for rate in rates:
                data.append({
                    'time': int(rate['time']),
                    'open': float(rate['open']),
                    'high': float(rate['high']),
                    'low': float(rate['low']),
                    'close': float(rate['close']),
                    'tick_volume': int(rate['tick_volume']),
                    'spread': int(rate['spread']),
                    'real_volume': int(rate['real_volume'])
                })
            
            return jsonify(data)
        
        @self.app.route('/place_order', methods=['POST'])
        def place_order():
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            data = request.get_json()
            result = self.place_order_mt5(data)
            return jsonify(result)
        
        @self.app.route('/modify_order', methods=['POST'])
        def modify_order():
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            data = request.get_json()
            result = self.modify_order_mt5(data)
            return jsonify(result)
        
        @self.app.route('/cancel_order/<order_id>', methods=['DELETE'])
        def cancel_order(order_id):
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            result = self.cancel_order_mt5(int(order_id))
            return jsonify(result)
        
        @self.app.route('/positions', methods=['GET'])
        def get_positions():
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            positions = mt5.positions_get()
            if positions is None:
                return jsonify([])
            
            return jsonify([pos._asdict() for pos in positions])
        
        @self.app.route('/orders', methods=['GET'])
        def get_orders():
            if not self.connected:
                return jsonify({'error': 'Not connected to MT5'}), 400
            
            orders = mt5.orders_get()
            if orders is None:
                return jsonify([])
            
            return jsonify([order._asdict() for order in orders])
    
    def connect_to_mt5(self, login: int, password: str, server: str) -> bool:
        """Connect to MetaTrader 5"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login to account
            if not mt5.login(login, password=password, server=server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            self.connected = True
            self.account_info = mt5.account_info()
            
            logger.info(f"Connected to MT5 - Account: {login}, Server: {server}")
            logger.info(f"Account Info: {self.account_info}")
            
            # Start market data streaming
            self.start_market_data_stream()
            
            return True
            
        except Exception as e:
            logger.error(f"Exception during MT5 connection: {e}")
            return False
    
    def disconnect_from_mt5(self) -> bool:
        """Disconnect from MetaTrader 5"""
        try:
            self.should_stop.set()
            
            if self.market_data_thread and self.market_data_thread.is_alive():
                self.market_data_thread.join(timeout=5)
            
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
            return True
            
        except Exception as e:
            logger.error(f"Exception during MT5 disconnection: {e}")
            return False
    
    def place_order_mt5(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place order in MT5"""
        try:
            symbol = order_data['symbol']
            order_type = order_data['order_type']  # 'buy' or 'sell'
            volume = float(order_data['volume'])
            price = float(order_data.get('price', 0))
            sl = float(order_data.get('sl', 0))
            tp = float(order_data.get('tp', 0))
            comment = order_data.get('comment', 'Forex Bot Order')
            
            # Prepare order request
            if order_type.lower() == 'buy':
                order_type_mt5 = mt5.ORDER_TYPE_BUY
                if price == 0:
                    price = mt5.symbol_info_tick(symbol).ask
            else:
                order_type_mt5 = mt5.ORDER_TYPE_SELL
                if price == 0:
                    price = mt5.symbol_info_tick(symbol).bid
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type_mt5,
                "price": price,
                "sl": sl,
                "tp": tp,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.retcode} - {result.comment}")
                return {
                    'success': False,
                    'error': f"Order failed: {result.retcode} - {result.comment}",
                    'retcode': result.retcode
                }
            
            logger.info(f"Order placed successfully: {result.order}")
            return {
                'success': True,
                'order_id': str(result.order),
                'retcode': result.retcode,
                'deal': result.deal,
                'volume': result.volume,
                'price': result.price,
                'comment': result.comment
            }
            
        except Exception as e:
            logger.error(f"Exception placing order: {e}")
            return {'success': False, 'error': str(e)}
    
    def modify_order_mt5(self, modify_data: Dict[str, Any]) -> Dict[str, Any]:
        """Modify order in MT5"""
        try:
            order_id = int(modify_data['order_id'])
            price = float(modify_data.get('price', 0))
            sl = float(modify_data.get('sl', 0))
            tp = float(modify_data.get('tp', 0))
            
            request = {
                "action": mt5.TRADE_ACTION_MODIFY,
                "order": order_id,
                "price": price,
                "sl": sl,
                "tp": tp,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order modification failed: {result.retcode}")
                return {
                    'success': False,
                    'error': f"Modification failed: {result.retcode}",
                    'retcode': result.retcode
                }
            
            return {'success': True, 'retcode': result.retcode}
            
        except Exception as e:
            logger.error(f"Exception modifying order: {e}")
            return {'success': False, 'error': str(e)}
    
    def cancel_order_mt5(self, order_id: int) -> Dict[str, Any]:
        """Cancel order in MT5"""
        try:
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order_id,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order cancellation failed: {result.retcode}")
                return {
                    'success': False,
                    'error': f"Cancellation failed: {result.retcode}",
                    'retcode': result.retcode
                }
            
            return {'success': True, 'retcode': result.retcode}
            
        except Exception as e:
            logger.error(f"Exception cancelling order: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_market_data_stream(self):
        """Start market data streaming thread"""
        if self.market_data_thread and self.market_data_thread.is_alive():
            return
        
        self.should_stop.clear()
        self.market_data_thread = threading.Thread(target=self._market_data_worker)
        self.market_data_thread.daemon = True
        self.market_data_thread.start()
        logger.info("Market data streaming started")
    
    def _market_data_worker(self):
        """Market data streaming worker"""
        symbols = self.config.get('symbols', ['EURUSD', 'GBPUSD', 'USDJPY'])
        
        while not self.should_stop.is_set():
            try:
                for symbol in symbols:
                    if self.should_stop.is_set():
                        break
                    
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None:
                        # Send tick data to C++ engine via callback
                        tick_data = {
                            'symbol': symbol,
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'last': tick.last,
                            'volume': tick.volume,
                            'time': tick.time,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        
                        # Send to callback URL
                        try:
                            requests.post(
                                self.callback_url,
                                json=tick_data,
                                timeout=1
                            )
                        except requests.RequestException:
                            pass  # Ignore callback failures
                
                time.sleep(0.1)  # 100ms interval
                
            except Exception as e:
                logger.error(f"Market data streaming error: {e}")
                time.sleep(1)
        
        logger.info("Market data streaming stopped")
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the Flask application"""
        logger.info(f"Starting MT5 Bridge on {host}:{port}")
        self.app.run(host=host, port=port, debug=False, threaded=True)
    
    def shutdown(self):
        """Shutdown the bridge"""
        logger.info("Shutting down MT5 Bridge")
        self.disconnect_from_mt5()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    if 'bridge' in globals():
        bridge.shutdown()
    sys.exit(0)

def main():
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config_file = os.environ.get('MT5_CONFIG', '/workspace/config/mt5_config.json')
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Config file {config_file} not found, using defaults")
        config = {
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF'],
            'callback_url': 'http://localhost:8080/mt5/callback'
        }
    
    # Create and run bridge
    global bridge
    bridge = MT5Bridge(config)
    
    try:
        bridge.run(
            host=os.environ.get('MT5_BRIDGE_HOST', '0.0.0.0'),
            port=int(os.environ.get('MT5_BRIDGE_PORT', 5000))
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        bridge.shutdown()

if __name__ == '__main__':
    main()