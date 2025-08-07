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
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_predictor import GeminiPredictor, MarketData, PredictionResult
from sentiment_analyzer.gemini_sentiment import GeminiSentimentAnalyzer, NewsItem, SocialPost, SentimentResult
from model_config import ModelManager, AVAILABLE_MODELS, print_available_models

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

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')

if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable is required")
    raise ValueError("GEMINI_API_KEY is required")

# Initialize services
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True
)

# Initialize Gemini models
gemini_predictor = GeminiPredictor(GEMINI_API_KEY)
sentiment_analyzer = GeminiSentimentAnalyzer(GEMINI_API_KEY)

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
        
        # Test Gemini connection (simple model info call)
        model_info = gemini_predictor.get_model_info()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'redis': 'connected',
                'gemini_predictor': 'connected',
                'sentiment_analyzer': 'connected'
            },
            'model_info': model_info
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict', methods=['POST'])
def predict_price():
    """Generate price prediction using Gemini"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['symbol', 'timeframe', 'ohlcv_data']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        symbol = data['symbol']
        timeframe = data['timeframe']
        ohlcv_data = data['ohlcv_data']
        
        # Convert OHLCV data to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in df.columns:
                return jsonify({'error': f'Missing OHLCV column: {col}'}), 400
        
        # Calculate technical indicators
        indicators = gemini_predictor.calculate_technical_indicators(df)
        
        # Get sentiment data if provided
        sentiment_score = data.get('sentiment_score')
        news_summary = data.get('news_summary')
        
        # Create market data object
        market_data = MarketData(
            symbol=symbol,
            timeframe=timeframe,
            ohlcv=df,
            indicators=indicators,
            sentiment_score=sentiment_score,
            news_summary=news_summary
        )
        
        # Generate prediction using Gemini
        prediction = executor.submit(
            run_async,
            gemini_predictor.predict_price_movement(market_data)
        ).result(timeout=60)  # 60 second timeout
        
        # Format response
        response = {
            'symbol': prediction.symbol,
            'direction': prediction.direction,
            'confidence': prediction.confidence,
            'target_price': prediction.target_price,
            'stop_loss': prediction.stop_loss,
            'time_horizon': prediction.time_horizon,
            'reasoning': prediction.reasoning,
            'risk_level': prediction.risk_level,
            'timestamp': prediction.timestamp.isoformat(),
            'technical_indicators': indicators,
            'model_info': {
                'provider': 'Google Gemini',
                'model': gemini_predictor.model_name
            }
        }
        
        # Cache the prediction
        cache_key = f"prediction:{symbol}:{timeframe}:{int(datetime.now().timestamp() // 300)}"
        redis_client.setex(cache_key, 300, json.dumps(response))  # 5 minute cache
        
        logger.info(f"Generated prediction for {symbol}: {prediction.direction} ({prediction.confidence:.2f})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        return jsonify({
            'error': 'Prediction generation failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    """Analyze market sentiment using Gemini"""
    try:
        data = request.get_json()
        
        # Validate input
        if 'symbol' not in data:
            return jsonify({'error': 'Missing required field: symbol'}), 400
        
        symbol = data['symbol']
        
        # Parse news items
        news_items = []
        for item_data in data.get('news', []):
            news_items.append(NewsItem(
                title=item_data.get('title', ''),
                content=item_data.get('content', ''),
                source=item_data.get('source', 'Unknown'),
                url=item_data.get('url', ''),
                published_at=datetime.fromisoformat(item_data.get('published_at', datetime.now().isoformat())),
                symbol=symbol
            ))
        
        # Parse social posts
        social_posts = []
        for post_data in data.get('social', []):
            social_posts.append(SocialPost(
                content=post_data.get('content', ''),
                platform=post_data.get('platform', 'Unknown'),
                author=post_data.get('author', 'Anonymous'),
                engagement=post_data.get('engagement', 0),
                posted_at=datetime.fromisoformat(post_data.get('posted_at', datetime.now().isoformat())),
                symbol=symbol
            ))
        
        # Generate sentiment analysis using Gemini
        sentiment = executor.submit(
            run_async,
            sentiment_analyzer.analyze_sentiment(symbol, news_items, social_posts)
        ).result(timeout=60)  # 60 second timeout
        
        # Format response
        response = {
            'symbol': sentiment.symbol,
            'overall_sentiment': sentiment.overall_sentiment,
            'confidence': sentiment.confidence,
            'key_themes': sentiment.key_themes,
            'news_sentiment': sentiment.news_sentiment,
            'social_sentiment': sentiment.social_sentiment,
            'risk_factors': sentiment.risk_factors,
            'opportunities': sentiment.opportunities,
            'summary': sentiment.summary,
            'timestamp': sentiment.timestamp.isoformat(),
            'model_info': {
                'provider': 'Google Gemini',
                'model': sentiment_analyzer.model_name
            }
        }
        
        # Cache the sentiment
        cache_key = f"sentiment:{symbol}:{int(datetime.now().timestamp() // 600)}"
        redis_client.setex(cache_key, 600, json.dumps(response))  # 10 minute cache
        
        logger.info(f"Generated sentiment for {symbol}: {sentiment.overall_sentiment:.2f} ({sentiment.confidence:.2f})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error generating sentiment: {e}")
        return jsonify({
            'error': 'Sentiment analysis failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/multi-timeframe', methods=['POST'])
def multi_timeframe_analysis():
    """Analyze multiple timeframes for comprehensive view"""
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['symbol', 'timeframes']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        symbol = data['symbol']
        timeframes = data['timeframes']
        
        # Prepare market data for each timeframe
        market_data_dict = {}
        for tf_data in timeframes:
            timeframe = tf_data['timeframe']
            ohlcv_data = tf_data['ohlcv_data']
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data)
            
            # Calculate indicators
            indicators = gemini_predictor.calculate_technical_indicators(df)
            
            market_data_dict[timeframe] = MarketData(
                symbol=symbol,
                timeframe=timeframe,
                ohlcv=df,
                indicators=indicators,
                sentiment_score=data.get('sentiment_score'),
                news_summary=data.get('news_summary')
            )
        
        # Analyze multiple timeframes
        predictions = executor.submit(
            run_async,
            gemini_predictor.analyze_multiple_timeframes(symbol, market_data_dict)
        ).result(timeout=120)  # 2 minute timeout for multiple timeframes
        
        # Format response
        response = {
            'symbol': symbol,
            'timeframes': {},
            'overall_consensus': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process each timeframe prediction
        consensus_scores = []
        for timeframe, prediction in predictions.items():
            response['timeframes'][timeframe] = {
                'direction': prediction.direction,
                'confidence': prediction.confidence,
                'target_price': prediction.target_price,
                'stop_loss': prediction.stop_loss,
                'reasoning': prediction.reasoning,
                'risk_level': prediction.risk_level
            }
            
            # Add to consensus calculation
            direction_score = 1 if prediction.direction == 'BUY' else (-1 if prediction.direction == 'SELL' else 0)
            consensus_scores.append(direction_score * prediction.confidence)
        
        # Calculate overall consensus
        if consensus_scores:
            avg_consensus = np.mean(consensus_scores)
            if avg_consensus > 0.3:
                consensus_direction = 'BUY'
            elif avg_consensus < -0.3:
                consensus_direction = 'SELL'
            else:
                consensus_direction = 'HOLD'
            
            response['overall_consensus'] = {
                'direction': consensus_direction,
                'strength': abs(avg_consensus),
                'agreement': len([s for s in consensus_scores if (s > 0) == (avg_consensus > 0)]) / len(consensus_scores)
            }
        
        logger.info(f"Generated multi-timeframe analysis for {symbol}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in multi-timeframe analysis: {e}")
        return jsonify({
            'error': 'Multi-timeframe analysis failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the AI models"""
    try:
        predictor_info = gemini_predictor.get_model_info()
        sentiment_info = sentiment_analyzer.get_model_info()
        
        return jsonify({
            'prediction_model': predictor_info,
            'sentiment_model': sentiment_info,
            'deployment_info': {
                'version': '2.0.0',
                'deployment_type': 'Gemini API',
                'hardware_requirements': 'None (Cloud-based)',
                'latency': 'Low (API-based)',
                'scalability': 'High'
            },
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/available', methods=['GET'])
def get_available_models():
    """Get list of all available models"""
    try:
        models_info = {}
        for model_name, config in AVAILABLE_MODELS.items():
            models_info[model_name] = {
                'display_name': config.display_name,
                'description': config.description,
                'multimodal': config.multimodal,
                'context_window': config.context_window,
                'max_tokens': config.max_tokens,
                'rate_limit_rpm': config.rate_limit_rpm,
                'recommended_temperature': config.recommended_temperature
            }
        
        return jsonify({
            'available_models': models_info,
            'current_prediction_model': gemini_predictor.model_name,
            'current_sentiment_model': sentiment_analyzer.model_name,
            'recommendations': gemini_predictor.model_manager.get_recommended_models_for_trading(),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting available models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/switch', methods=['POST'])
def switch_model():
    """Switch to a different model"""
    try:
        data = request.get_json()
        
        if 'model_name' not in data:
            return jsonify({'error': 'Missing required field: model_name'}), 400
        
        model_name = data['model_name']
        service = data.get('service', 'prediction')  # 'prediction' or 'sentiment'
        
        if model_name not in AVAILABLE_MODELS:
            return jsonify({
                'error': f'Model {model_name} not available',
                'available_models': list(AVAILABLE_MODELS.keys())
            }), 400
        
        success = False
        old_model = None
        
        if service == 'prediction':
            old_model = gemini_predictor.model_name
            success = gemini_predictor.switch_model(model_name)
        elif service == 'sentiment':
            old_model = sentiment_analyzer.model_name
            success = sentiment_analyzer.switch_model(model_name)
        else:
            return jsonify({'error': 'Invalid service. Must be "prediction" or "sentiment"'}), 400
        
        if success:
            return jsonify({
                'message': f'Successfully switched {service} model',
                'old_model': old_model,
                'new_model': model_name,
                'service': service,
                'timestamp': datetime.now().isoformat()
            }), 200
        else:
            return jsonify({
                'error': f'Failed to switch {service} model to {model_name}',
                'current_model': old_model
            }), 500
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/test', methods=['POST'])
def test_model():
    """Test a specific model with sample data"""
    try:
        data = request.get_json()
        
        if 'model_name' not in data:
            return jsonify({'error': 'Missing required field: model_name'}), 400
        
        model_name = data['model_name']
        test_type = data.get('test_type', 'prediction')
        
        if model_name not in AVAILABLE_MODELS:
            return jsonify({
                'error': f'Model {model_name} not available',
                'available_models': list(AVAILABLE_MODELS.keys())
            }), 400
        
        # Create temporary predictor with the test model
        test_predictor = GeminiPredictor(GEMINI_API_KEY, model_name)
        
        # Create simple test data
        test_df = pd.DataFrame({
            'open': [1.0950, 1.0955, 1.0960],
            'high': [1.0965, 1.0970, 1.0975],
            'low': [1.0945, 1.0950, 1.0955],
            'close': [1.0960, 1.0965, 1.0970],
            'volume': [1000, 1100, 1200]
        })
        
        # Calculate indicators
        indicators = test_predictor.calculate_technical_indicators(test_df)
        
        # Create market data
        market_data = MarketData(
            symbol="EUR/USD",
            timeframe="1h",
            ohlcv=test_df,
            indicators=indicators,
            sentiment_score=0.2,
            news_summary="Test market analysis"
        )
        
        # Generate test prediction
        start_time = datetime.now()
        prediction = executor.submit(
            run_async,
            test_predictor.predict_price_movement(market_data)
        ).result(timeout=30)
        end_time = datetime.now()
        
        response_time = (end_time - start_time).total_seconds()
        
        return jsonify({
            'model_name': model_name,
            'test_type': test_type,
            'test_result': 'success',
            'response_time_seconds': response_time,
            'prediction': {
                'direction': prediction.direction,
                'confidence': prediction.confidence,
                'reasoning': prediction.reasoning[:200] + "..." if len(prediction.reasoning) > 200 else prediction.reasoning
            },
            'model_info': test_predictor.get_model_info(),
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return jsonify({
            'model_name': model_name,
            'test_result': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/mt5-signals', methods=['POST'])
def handle_mt5_signal():
    """Handle incoming signals from MT5 Windows VM"""
    try:
        signal_data = request.get_json()
        
        if not signal_data:
            return jsonify({'error': 'No signal data provided'}), 400
        
        # Validate required fields
        required_fields = ['symbol', 'bid', 'ask', 'timestamp']
        for field in required_fields:
            if field not in signal_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        logger.info(f"Received MT5 signal for {signal_data.get('symbol')}")
        
        # Store signal in Redis for processing
        signal_key = f"mt5_signal:{signal_data['symbol']}:{int(datetime.now().timestamp())}"
        redis_client.setex(signal_key, 3600, json.dumps(signal_data))  # Store for 1 hour
        
        # Process signal with AI
        ai_decision = process_mt5_signal_with_ai(signal_data)
        
        # Store AI decision in InfluxDB (if available)
        store_signal_in_influxdb(signal_data, ai_decision)
        
        response = {
            'status': 'success',
            'signal_received': True,
            'ai_decision': ai_decision,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error handling MT5 signal: {e}")
        return jsonify({'error': str(e)}), 500

def process_mt5_signal_with_ai(signal_data):
    """Process MT5 signal data with Gemini AI"""
    try:
        # Create DataFrame from signal data
        signal_df = pd.DataFrame([{
            'open': signal_data.get('ohlc', {}).get('open', signal_data.get('bid')),
            'high': signal_data.get('ohlc', {}).get('high', signal_data.get('ask')),
            'low': signal_data.get('ohlc', {}).get('low', signal_data.get('bid')),
            'close': signal_data.get('ohlc', {}).get('close', signal_data.get('ask')),
            'volume': signal_data.get('volume', 1000)
        }])
        
        # Calculate technical indicators if not provided
        indicators = signal_data.get('indicators', {})
        if not indicators:
            indicators = gemini_predictor.calculate_technical_indicators(signal_df)
        
        # Create MarketData object
        market_data = MarketData(
            symbol=signal_data['symbol'],
            timeframe="M5",  # Default to 5-minute timeframe
            ohlcv=signal_df,
            indicators=indicators,
            sentiment_score=0.0,  # Default neutral sentiment
            news_summary="MT5 signal analysis"
        )
        
        # Generate prediction
        prediction = executor.submit(
            run_async,
            gemini_predictor.predict_price_movement(market_data)
        ).result(timeout=30)
        
        return {
            'action': prediction.direction,
            'confidence': prediction.confidence,
            'reasoning': prediction.reasoning,
            'risk_level': 'MEDIUM',  # Default risk level
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing MT5 signal with AI: {e}")
        return {
            'action': 'HOLD',
            'confidence': 0.0,
            'reasoning': f'AI processing error: {str(e)}',
            'risk_level': 'HIGH',
            'timestamp': datetime.now().isoformat()
        }

def store_signal_in_influxdb(signal_data, ai_decision):
    """Store signal and AI decision in InfluxDB"""
    try:
        # This would integrate with your existing InfluxDB setup
        # For now, we'll store in Redis as a fallback
        influx_data = {
            'signal': signal_data,
            'ai_decision': ai_decision,
            'timestamp': datetime.now().isoformat()
        }
        
        influx_key = f"influx_data:{signal_data['symbol']}:{int(datetime.now().timestamp())}"
        redis_client.setex(influx_key, 86400, json.dumps(influx_data))  # Store for 24 hours
        
        logger.info(f"Stored signal data for {signal_data['symbol']} in Redis (InfluxDB fallback)")
        
    except Exception as e:
        logger.error(f"Error storing signal in InfluxDB: {e}")

@app.route('/cache-stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        # Get cache keys
        prediction_keys = redis_client.keys('prediction:*')
        sentiment_keys = redis_client.keys('sentiment:*')
        mt5_signal_keys = redis_client.keys('mt5_signal:*')
        
        stats = {
            'prediction_cache': {
                'entries': len(prediction_keys),
                'keys': prediction_keys[:10]  # Show first 10
            },
            'sentiment_cache': {
                'entries': len(sentiment_keys),
                'keys': sentiment_keys[:10]  # Show first 10
            },
            'mt5_signals': {
                'entries': len(mt5_signal_keys),
                'keys': mt5_signal_keys[:10]  # Show first 10
            },
            'redis_info': {
                'connected': True,
                'memory_usage': redis_client.info().get('used_memory_human', 'Unknown')
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Gemini-powered Price Predictor Service")
    logger.info(f"Using Gemini model: {gemini_predictor.model_name}")
    logger.info(f"Redis connected: {REDIS_HOST}:{REDIS_PORT}")
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )