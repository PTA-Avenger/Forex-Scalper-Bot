from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import redis
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List, Optional, Tuple

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_model import LSTMPredictor
from models.xgboost_model import XGBoostPredictor
from models.ensemble_model import EnsemblePredictor
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from config.settings import Config
from common.database import DatabaseManager
from common.redis_client import RedisManager

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/price_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global components
config = Config()
redis_manager = RedisManager(config)
db_manager = DatabaseManager(config)
preprocessor = DataPreprocessor(config)
feature_engineer = FeatureEngineer(config)

# Initialize models
lstm_model = None
xgboost_model = None
ensemble_model = None

def initialize_models():
    """Initialize ML models"""
    global lstm_model, xgboost_model, ensemble_model
    
    try:
        logger.info("Initializing ML models...")
        
        # Initialize individual models
        lstm_model = LSTMPredictor(config)
        xgboost_model = XGBoostPredictor(config)
        
        # Load pre-trained models if they exist
        if os.path.exists(config.LSTM_MODEL_PATH):
            lstm_model.load_model(config.LSTM_MODEL_PATH)
            logger.info("Loaded pre-trained LSTM model")
        
        if os.path.exists(config.XGBOOST_MODEL_PATH):
            xgboost_model.load_model(config.XGBOOST_MODEL_PATH)
            logger.info("Loaded pre-trained XGBoost model")
        
        # Initialize ensemble model
        ensemble_model = EnsemblePredictor([lstm_model, xgboost_model], config)
        
        logger.info("ML models initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        return False

def get_historical_data(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    """Get historical market data for a symbol"""
    try:
        # Try to get from cache first
        cache_key = f"historical_data:{symbol}:{timeframe}:{limit}"
        cached_data = redis_manager.get(cache_key)
        
        if cached_data:
            logger.debug(f"Retrieved historical data from cache for {symbol}")
            return pd.read_json(cached_data)
        
        # Get from database
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM market_data 
        WHERE symbol = %s AND timeframe = %s 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        
        data = db_manager.fetch_all(query, (symbol, timeframe, limit))
        
        if not data:
            logger.warning(f"No historical data found for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Cache the result
        redis_manager.setex(cache_key, 300, df.to_json())  # 5 minute cache
        
        logger.debug(f"Retrieved {len(df)} historical data points for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {str(e)}")
        return pd.DataFrame()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check model status
        models_ready = all([
            lstm_model is not None,
            xgboost_model is not None,
            ensemble_model is not None
        ])
        
        # Check Redis connection
        redis_status = redis_manager.ping()
        
        # Check database connection
        db_status = db_manager.test_connection()
        
        status = {
            'status': 'healthy' if all([models_ready, redis_status, db_status]) else 'unhealthy',
            'service': 'price_predictor',
            'models_ready': models_ready,
            'redis_connected': redis_status,
            'database_connected': db_status,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return jsonify(status), 200 if status['status'] == 'healthy' else 503
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict_price():
    """Generate price predictions"""
    try:
        data = request.json
        
        # Validate input
        required_fields = ['symbol']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        symbol = data['symbol']
        timeframe = data.get('timeframe', '1m')
        horizon = data.get('horizon', 1)
        model_type = data.get('model', 'ensemble')  # 'lstm', 'xgboost', 'ensemble'
        
        logger.info(f"Prediction request: {symbol}, {timeframe}, horizon={horizon}, model={model_type}")
        
        # Get historical data
        historical_data = get_historical_data(symbol, timeframe, limit=2000)
        
        if historical_data.empty:
            return jsonify({'error': f'No historical data available for {symbol}'}), 404
        
        # Preprocess data
        processed_data = preprocessor.process(historical_data)
        
        if processed_data is None or len(processed_data) < 100:
            return jsonify({'error': 'Insufficient data for prediction'}), 400
        
        # Generate predictions based on model type
        predictions = {}
        confidence_scores = {}
        
        if model_type in ['lstm', 'ensemble']:
            try:
                lstm_pred = lstm_model.predict(processed_data, horizon)
                predictions['lstm'] = lstm_pred.tolist() if isinstance(lstm_pred, np.ndarray) else [lstm_pred]
                confidence_scores['lstm'] = lstm_model.get_confidence() if hasattr(lstm_model, 'get_confidence') else 0.5
            except Exception as e:
                logger.error(f"LSTM prediction error: {str(e)}")
                predictions['lstm'] = [0.0] * horizon
                confidence_scores['lstm'] = 0.0
        
        if model_type in ['xgboost', 'ensemble']:
            try:
                xgb_pred = xgboost_model.predict(processed_data, horizon)
                predictions['xgboost'] = xgb_pred.tolist() if isinstance(xgb_pred, np.ndarray) else [xgb_pred]
                confidence_scores['xgboost'] = xgboost_model.get_confidence() if hasattr(xgboost_model, 'get_confidence') else 0.5
            except Exception as e:
                logger.error(f"XGBoost prediction error: {str(e)}")
                predictions['xgboost'] = [0.0] * horizon
                confidence_scores['xgboost'] = 0.0
        
        if model_type == 'ensemble':
            try:
                ensemble_pred = ensemble_model.predict(processed_data, horizon)
                predictions['ensemble'] = ensemble_pred.tolist() if isinstance(ensemble_pred, np.ndarray) else [ensemble_pred]
                confidence_scores['ensemble'] = ensemble_model.get_confidence()
            except Exception as e:
                logger.error(f"Ensemble prediction error: {str(e)}")
                predictions['ensemble'] = [0.0] * horizon
                confidence_scores['ensemble'] = 0.0
        
        # Calculate prediction metadata
        current_price = float(historical_data['close'].iloc[-1])
        
        # Use ensemble prediction as primary if available, otherwise use requested model
        primary_prediction = predictions.get('ensemble', predictions.get(model_type, [current_price]))
        primary_confidence = confidence_scores.get('ensemble', confidence_scores.get(model_type, 0.0))
        
        # Calculate price change and direction
        if len(primary_prediction) > 0:
            predicted_price = primary_prediction[0]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            direction = 'bullish' if price_change > 0 else 'bearish' if price_change < 0 else 'neutral'
        else:
            predicted_price = current_price
            price_change = 0.0
            price_change_pct = 0.0
            direction = 'neutral'
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'horizon': horizon,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'direction': direction,
            'confidence': primary_confidence,
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'model_used': model_type,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Cache the result
        cache_key = f"prediction:{symbol}:{timeframe}:{horizon}:{model_type}"
        redis_manager.setex(cache_key, 60, json.dumps(result))  # 1 minute cache
        
        logger.info(f"Prediction completed for {symbol}: {direction} ({price_change_pct:.2f}%)")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Generate predictions for multiple symbols"""
    try:
        data = request.json
        symbols = data.get('symbols', [])
        timeframe = data.get('timeframe', '1m')
        horizon = data.get('horizon', 1)
        model_type = data.get('model', 'ensemble')
        
        if not symbols:
            return jsonify({'error': 'No symbols provided'}), 400
        
        results = {}
        
        for symbol in symbols:
            try:
                # Make individual prediction request
                pred_data = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'horizon': horizon,
                    'model': model_type
                }
                
                # Simulate internal request
                with app.test_request_context('/predict', json=pred_data, method='POST'):
                    response = predict_price()
                    if response[1] == 200:  # Success
                        results[symbol] = response[0].get_json()
                    else:
                        results[symbol] = {'error': 'Prediction failed'}
                        
            except Exception as e:
                logger.error(f"Batch prediction error for {symbol}: {str(e)}")
                results[symbol] = {'error': str(e)}
        
        return jsonify({
            'results': results,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_models():
    """Trigger model retraining"""
    try:
        data = request.json
        symbol = data.get('symbol', 'ALL')
        model_type = data.get('model', 'ensemble')
        
        # Add retraining task to queue
        training_task = {
            'symbol': symbol,
            'model_type': model_type,
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'queued'
        }
        
        # Add to Redis queue for background processing
        redis_manager.lpush('training_queue', json.dumps(training_task))
        
        logger.info(f"Retraining queued for {symbol} ({model_type})")
        
        return jsonify({
            'message': 'Retraining queued successfully',
            'symbol': symbol,
            'model_type': model_type,
            'task_id': f"train_{symbol}_{int(datetime.utcnow().timestamp())}"
        })
        
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    try:
        info = {
            'lstm': {
                'loaded': lstm_model is not None,
                'architecture': lstm_model.get_architecture() if lstm_model and hasattr(lstm_model, 'get_architecture') else None,
                'last_trained': lstm_model.get_last_trained() if lstm_model and hasattr(lstm_model, 'get_last_trained') else None
            },
            'xgboost': {
                'loaded': xgboost_model is not None,
                'feature_importance': xgboost_model.get_feature_importance() if xgboost_model and hasattr(xgboost_model, 'get_feature_importance') else None,
                'last_trained': xgboost_model.get_last_trained() if xgboost_model and hasattr(xgboost_model, 'get_last_trained') else None
            },
            'ensemble': {
                'loaded': ensemble_model is not None,
                'weights': ensemble_model.get_weights() if ensemble_model and hasattr(ensemble_model, 'get_weights') else None
            }
        }
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Get service metrics for monitoring"""
    try:
        # Get prediction statistics from Redis
        stats = redis_manager.hgetall('prediction_stats') or {}
        
        metrics = {
            'predictions_made': int(stats.get('predictions_made', 0)),
            'cache_hits': int(stats.get('cache_hits', 0)),
            'cache_misses': int(stats.get('cache_misses', 0)),
            'errors': int(stats.get('errors', 0)),
            'avg_response_time': float(stats.get('avg_response_time', 0)),
            'uptime': (datetime.utcnow() - datetime.fromisoformat(stats.get('start_time', datetime.utcnow().isoformat()))).total_seconds() if 'start_time' in stats else 0
        }
        
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize models on startup
    if not initialize_models():
        logger.error("Failed to initialize models. Exiting.")
        sys.exit(1)
    
    # Record start time
    redis_manager.hset('prediction_stats', 'start_time', datetime.utcnow().isoformat())
    
    logger.info("Price Predictor Service starting...")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=config.PRICE_PREDICTOR_PORT,
        debug=config.DEBUG,
        threaded=True
    )