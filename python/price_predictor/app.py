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

@app.route('/cache-stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        # Get cache keys
        prediction_keys = redis_client.keys('prediction:*')
        sentiment_keys = redis_client.keys('sentiment:*')
        
        stats = {
            'prediction_cache': {
                'entries': len(prediction_keys),
                'keys': prediction_keys[:10]  # Show first 10
            },
            'sentiment_cache': {
                'entries': len(sentiment_keys),
                'keys': sentiment_keys[:10]  # Show first 10
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