#!/usr/bin/env python3
"""
FXCM Service Flask Application
REST API interface for FXCM trading operations
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import asyncio

from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxcm_client import FXCMClient, FXCMConfig, OrderRequest, create_fxcm_client

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
FXCM_ACCESS_TOKEN = os.getenv('FXCM_ACCESS_TOKEN')
FXCM_SERVER_TYPE = os.getenv('FXCM_SERVER_TYPE', 'demo')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')

if not FXCM_ACCESS_TOKEN:
    logger.error("FXCM_ACCESS_TOKEN environment variable is required")
    raise ValueError("FXCM_ACCESS_TOKEN is required")

# Initialize services
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# Initialize FXCM client
fxcm_client = create_fxcm_client(FXCM_ACCESS_TOKEN, FXCM_SERVER_TYPE)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

def run_async(coro):
    """Run async function in thread pool"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        redis_client.ping()
        
        # Test FXCM connection
        if not fxcm_client.is_connected:
            return jsonify({
                'status': 'unhealthy',
                'error': 'FXCM not connected',
                'timestamp': datetime.now().isoformat(),
                'services': {
                    'redis': 'connected',
                    'fxcm': 'disconnected'
                }
            }), 503
        
        # Get account info to verify connection
        account_info = fxcm_client.get_account_info()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'redis': 'connected',
                'fxcm': 'connected'
            },
            'account': {
                'balance': account_info.get('balance', 0),
                'currency': account_info.get('currency', 'USD'),
                'server_type': FXCM_SERVER_TYPE
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/account', methods=['GET'])
def get_account():
    """Get account information"""
    try:
        account_info = fxcm_client.get_account_info()
        
        if not account_info:
            return jsonify({'error': 'Failed to get account information'}), 500
        
        return jsonify({
            'success': True,
            'account': account_info,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Get account error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/positions', methods=['GET'])
def get_positions():
    """Get all open positions"""
    try:
        positions = fxcm_client.get_open_positions()
        
        positions_data = []
        for position in positions:
            positions_data.append({
                'position_id': position.position_id,
                'symbol': position.symbol,
                'side': position.side,
                'amount': position.amount,
                'open_price': position.open_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'timestamp': position.timestamp.isoformat()
            })
        
        return jsonify({
            'success': True,
            'positions': positions_data,
            'count': len(positions_data),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/orders', methods=['POST'])
def place_order():
    """Place a new trading order"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['symbol', 'side', 'amount']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create order request
        order_request = OrderRequest(
            symbol=data['symbol'],
            side=data['side'].lower(),
            amount=float(data['amount']),
            order_type=data.get('order_type', 'market').lower(),
            price=float(data['price']) if data.get('price') else None,
            stop_loss=float(data['stop_loss']) if data.get('stop_loss') else None,
            take_profit=float(data['take_profit']) if data.get('take_profit') else None,
            time_in_force=data.get('time_in_force', 'GTC'),
            comment=data.get('comment', 'Gemini AI Bot')
        )
        
        # Place order
        result = fxcm_client.place_order(order_request)
        
        response_data = {
            'success': result.success,
            'timestamp': datetime.now().isoformat()
        }
        
        if result.success:
            response_data.update({
                'order_id': result.order_id,
                'trade_id': result.trade_id,
                'executed_price': result.executed_price,
                'executed_amount': result.executed_amount
            })
            return jsonify(response_data), 200
        else:
            response_data['error'] = result.error_message
            return jsonify(response_data), 400
            
    except Exception as e:
        logger.error(f"Place order error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/positions/<position_id>/close', methods=['POST'])
def close_position(position_id):
    """Close a specific position"""
    try:
        result = fxcm_client.close_position(position_id)
        
        response_data = {
            'success': result.success,
            'timestamp': datetime.now().isoformat()
        }
        
        if result.success:
            response_data.update({
                'order_id': result.order_id,
                'trade_id': result.trade_id,
                'executed_price': result.executed_price,
                'executed_amount': result.executed_amount
            })
            return jsonify(response_data), 200
        else:
            response_data['error'] = result.error_message
            return jsonify(response_data), 400
            
    except Exception as e:
        logger.error(f"Close position error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/market-data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    """Get current market data for a symbol"""
    try:
        market_data = fxcm_client.get_market_data(symbol)
        
        if not market_data:
            return jsonify({'error': f'No market data available for {symbol}'}), 404
        
        return jsonify({
            'success': True,
            'symbol': market_data.symbol,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'spread': market_data.spread,
            'timestamp': market_data.timestamp.isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Get market data error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/historical-data/<symbol>', methods=['GET'])
def get_historical_data(symbol):
    """Get historical OHLCV data for a symbol"""
    try:
        timeframe = request.args.get('timeframe', '1h')
        periods = int(request.args.get('periods', 100))
        
        if periods > 5000:  # Limit to prevent excessive data
            periods = 5000
        
        historical_data = fxcm_client.get_historical_data(symbol, timeframe, periods)
        
        if historical_data.empty:
            return jsonify({'error': f'No historical data available for {symbol}'}), 404
        
        # Convert DataFrame to list of dictionaries
        data_list = []
        for _, row in historical_data.iterrows():
            data_list.append({
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'timeframe': timeframe,
            'periods': len(data_list),
            'data': data_list,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Get historical data error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/symbols', methods=['GET'])
def get_available_symbols():
    """Get list of available trading symbols"""
    try:
        # FXCM major forex pairs
        symbols = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
            'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP',
            'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/CHF',
            'GBP/CHF', 'CHF/JPY', 'EUR/AUD', 'GBP/AUD',
            'AUD/CAD', 'AUD/CHF', 'CAD/CHF', 'CAD/JPY',
            'EUR/CAD', 'EUR/NZD', 'GBP/CAD', 'GBP/NZD',
            'NZD/CAD', 'NZD/CHF', 'NZD/JPY'
        ]
        
        # Get current prices for major pairs
        symbol_data = []
        major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD']
        
        for symbol in major_pairs:
            try:
                market_data = fxcm_client.get_market_data(symbol)
                if market_data:
                    symbol_data.append({
                        'symbol': symbol,
                        'bid': market_data.bid,
                        'ask': market_data.ask,
                        'spread': market_data.spread,
                        'timestamp': market_data.timestamp.isoformat()
                    })
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
        
        return jsonify({
            'success': True,
            'available_symbols': symbols,
            'current_prices': symbol_data,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Get symbols error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook/test', methods=['POST'])
def test_webhook():
    """Test webhook endpoint"""
    try:
        data = request.get_json()
        logger.info(f"Webhook test received: {data}")
        
        # Store in Redis for testing
        redis_client.setex(
            "fxcm:webhook:test",
            300,  # 5 minutes
            json.dumps(data, default=str)
        )
        
        return jsonify({
            'success': True,
            'message': 'Webhook received successfully',
            'data': data,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Webhook test error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cache-stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        price_cache_info = {
            'size': len(fxcm_client.price_cache),
            'maxsize': fxcm_client.price_cache.maxsize,
            'ttl': fxcm_client.price_cache.ttl,
            'hits': getattr(fxcm_client.price_cache, 'hits', 0),
            'misses': getattr(fxcm_client.price_cache, 'misses', 0)
        }
        
        account_cache_info = {
            'size': len(fxcm_client.account_cache),
            'maxsize': fxcm_client.account_cache.maxsize,
            'ttl': fxcm_client.account_cache.ttl,
            'hits': getattr(fxcm_client.account_cache, 'hits', 0),
            'misses': getattr(fxcm_client.account_cache, 'misses', 0)
        }
        
        return jsonify({
            'success': True,
            'cache_stats': {
                'price_cache': price_cache_info,
                'account_cache': account_cache_info
            },
            'connection_status': {
                'fxcm_connected': fxcm_client.is_connected,
                'fxcm_streaming': fxcm_client.is_streaming
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/connection/status', methods=['GET'])
def get_connection_status():
    """Get detailed connection status"""
    try:
        return jsonify({
            'success': True,
            'connection': {
                'fxcm_connected': fxcm_client.is_connected,
                'fxcm_streaming': fxcm_client.is_streaming,
                'server_type': FXCM_SERVER_TYPE,
                'reconnect_attempts': fxcm_client.config.reconnect_attempts,
                'heartbeat_interval': fxcm_client.config.heartbeat_interval
            },
            'rate_limits': {
                'max_requests_per_minute': fxcm_client.config.max_requests_per_minute,
                'max_orders_per_minute': fxcm_client.config.max_orders_per_minute
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Connection status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/connection/reconnect', methods=['POST'])
def reconnect():
    """Force reconnection to FXCM"""
    try:
        logger.info("Manual reconnection requested")
        
        # Disconnect and reconnect
        fxcm_client.disconnect()
        
        # Reinitialize
        global fxcm_client
        fxcm_client = create_fxcm_client(FXCM_ACCESS_TOKEN, FXCM_SERVER_TYPE)
        
        if fxcm_client.is_connected:
            return jsonify({
                'success': True,
                'message': 'Reconnection successful',
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Reconnection failed',
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        logger.error(f"Reconnection error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'timestamp': datetime.now().isoformat()
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == '__main__':
    # Check if running in production
    port = int(os.getenv('PORT', 5003))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting FXCM Service on port {port}")
    logger.info(f"FXCM Server Type: {FXCM_SERVER_TYPE}")
    logger.info(f"Debug Mode: {debug}")
    
    if fxcm_client.is_connected:
        logger.info("✅ FXCM client connected successfully")
        app.run(host='0.0.0.0', port=port, debug=debug)
    else:
        logger.error("❌ Failed to connect to FXCM - exiting")
        sys.exit(1)