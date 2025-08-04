#!/usr/bin/env python3
"""
Gemini Model Tester CLI
Test different Gemini models for forex trading analysis
"""

import os
import sys
import asyncio
import argparse
import pandas as pd
from datetime import datetime
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from price_predictor.gemini_predictor import GeminiPredictor, MarketData
from price_predictor.model_config import AVAILABLE_MODELS, print_available_models
from sentiment_analyzer.gemini_sentiment import GeminiSentimentAnalyzer, NewsItem, SocialPost

def create_test_data():
    """Create sample test data for model testing"""
    # Sample OHLCV data
    test_df = pd.DataFrame({
        'open': [1.0950, 1.0955, 1.0960, 1.0965, 1.0970],
        'high': [1.0965, 1.0970, 1.0975, 1.0980, 1.0985],
        'low': [1.0945, 1.0950, 1.0955, 1.0960, 1.0965],
        'close': [1.0960, 1.0965, 1.0970, 1.0975, 1.0980],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Sample news items
    news_items = [
        NewsItem(
            title="EUR/USD Technical Analysis: Bulls Eye 1.10 Resistance",
            content="The EUR/USD pair has been showing strong bullish momentum as it approaches the key 1.10 resistance level. Technical indicators suggest continued upward pressure.",
            source="ForexLive",
            url="https://example.com/news1",
            published_at=datetime.now()
        ),
        NewsItem(
            title="ECB Policy Decision Supports Euro Strength",
            content="The European Central Bank's latest policy decision has provided additional support for the Euro against major currencies, with markets expecting continued hawkish stance.",
            source="Reuters",
            url="https://example.com/news2",
            published_at=datetime.now()
        )
    ]
    
    # Sample social posts
    social_posts = [
        SocialPost(
            content="EUR/USD breaking above key resistance! Technical setup looking very bullish for the week ahead. #forex #EURUSD",
            platform="Twitter",
            author="ForexTrader123",
            engagement=45,
            posted_at=datetime.now()
        ),
        SocialPost(
            content="Watching EUR/USD closely. The fundamentals are aligning with the technicals for a potential breakout.",
            platform="Reddit",
            author="MarketAnalyst",
            engagement=23,
            posted_at=datetime.now()
        )
    ]
    
    return test_df, news_items, social_posts

async def test_prediction_model(api_key: str, model_name: str, test_df: pd.DataFrame) -> dict:
    """Test a prediction model with sample data"""
    try:
        print(f"🔮 Testing prediction model: {model_name}")
        
        # Initialize predictor
        predictor = GeminiPredictor(api_key, model_name)
        
        # Calculate technical indicators
        indicators = predictor.calculate_technical_indicators(test_df)
        
        # Create market data
        market_data = MarketData(
            symbol="EUR/USD",
            timeframe="1h",
            ohlcv=test_df,
            indicators=indicators,
            sentiment_score=0.3,
            news_summary="Positive market sentiment with bullish technical indicators"
        )
        
        # Generate prediction
        start_time = time.time()
        prediction = await predictor.predict_price_movement(market_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        return {
            'success': True,
            'model_name': model_name,
            'response_time': response_time,
            'prediction': {
                'direction': prediction.direction,
                'confidence': prediction.confidence,
                'target_price': prediction.target_price,
                'stop_loss': prediction.stop_loss,
                'risk_level': prediction.risk_level,
                'reasoning': prediction.reasoning[:200] + "..." if len(prediction.reasoning) > 200 else prediction.reasoning
            },
            'model_info': predictor.get_model_info()
        }
        
    except Exception as e:
        return {
            'success': False,
            'model_name': model_name,
            'error': str(e)
        }

async def test_sentiment_model(api_key: str, model_name: str, news_items: list, social_posts: list) -> dict:
    """Test a sentiment model with sample data"""
    try:
        print(f"😊 Testing sentiment model: {model_name}")
        
        # Initialize sentiment analyzer
        analyzer = GeminiSentimentAnalyzer(api_key, model_name)
        
        # Generate sentiment analysis
        start_time = time.time()
        sentiment = await analyzer.analyze_sentiment("EUR/USD", news_items, social_posts)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        return {
            'success': True,
            'model_name': model_name,
            'response_time': response_time,
            'sentiment': {
                'overall_sentiment': sentiment.overall_sentiment,
                'confidence': sentiment.confidence,
                'news_sentiment': sentiment.news_sentiment,
                'social_sentiment': sentiment.social_sentiment,
                'key_themes': sentiment.key_themes[:3],  # Top 3 themes
                'summary': sentiment.summary[:200] + "..." if len(sentiment.summary) > 200 else sentiment.summary
            },
            'model_info': analyzer.get_model_info()
        }
        
    except Exception as e:
        return {
            'success': False,
            'model_name': model_name,
            'error': str(e)
        }

async def benchmark_models(api_key: str, models_to_test: list = None):
    """Benchmark multiple models"""
    if models_to_test is None:
        models_to_test = list(AVAILABLE_MODELS.keys())
    
    print("🏁 Starting Model Benchmark")
    print("=" * 50)
    
    # Create test data
    test_df, news_items, social_posts = create_test_data()
    
    results = {
        'prediction_results': [],
        'sentiment_results': []
    }
    
    for model_name in models_to_test:
        if model_name not in AVAILABLE_MODELS:
            print(f"❌ Model {model_name} not found in available models")
            continue
        
        print(f"\n🤖 Testing {model_name}")
        print("-" * 30)
        
        # Test prediction
        pred_result = await test_prediction_model(api_key, model_name, test_df)
        results['prediction_results'].append(pred_result)
        
        if pred_result['success']:
            print(f"✅ Prediction: {pred_result['response_time']:.2f}s - {pred_result['prediction']['direction']} ({pred_result['prediction']['confidence']:.2f})")
        else:
            print(f"❌ Prediction failed: {pred_result['error']}")
        
        # Test sentiment
        sent_result = await test_sentiment_model(api_key, model_name, news_items, social_posts)
        results['sentiment_results'].append(sent_result)
        
        if sent_result['success']:
            print(f"✅ Sentiment: {sent_result['response_time']:.2f}s - {sent_result['sentiment']['overall_sentiment']:.2f} ({sent_result['sentiment']['confidence']:.2f})")
        else:
            print(f"❌ Sentiment failed: {sent_result['error']}")
        
        # Small delay between models to respect rate limits
        await asyncio.sleep(2)
    
    return results

def print_benchmark_summary(results: dict):
    """Print a summary of benchmark results"""
    print("\n📊 BENCHMARK SUMMARY")
    print("=" * 50)
    
    # Prediction results
    print("\n🔮 PREDICTION MODELS:")
    pred_results = [r for r in results['prediction_results'] if r['success']]
    if pred_results:
        # Sort by response time
        pred_results.sort(key=lambda x: x['response_time'])
        
        print(f"{'Model':<20} {'Time (s)':<10} {'Direction':<8} {'Confidence':<10}")
        print("-" * 50)
        for result in pred_results:
            print(f"{result['model_name']:<20} {result['response_time']:<10.2f} {result['prediction']['direction']:<8} {result['prediction']['confidence']:<10.2f}")
    
    # Sentiment results
    print("\n😊 SENTIMENT MODELS:")
    sent_results = [r for r in results['sentiment_results'] if r['success']]
    if sent_results:
        # Sort by response time
        sent_results.sort(key=lambda x: x['response_time'])
        
        print(f"{'Model':<20} {'Time (s)':<10} {'Sentiment':<10} {'Confidence':<10}")
        print("-" * 50)
        for result in sent_results:
            print(f"{result['model_name']:<20} {result['response_time']:<10.2f} {result['sentiment']['overall_sentiment']:<10.2f} {result['sentiment']['confidence']:<10.2f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if pred_results:
        fastest_pred = min(pred_results, key=lambda x: x['response_time'])
        print(f"⚡ Fastest Prediction: {fastest_pred['model_name']} ({fastest_pred['response_time']:.2f}s)")
        
        highest_conf_pred = max(pred_results, key=lambda x: x['prediction']['confidence'])
        print(f"🎯 Highest Confidence: {highest_conf_pred['model_name']} ({highest_conf_pred['prediction']['confidence']:.2f})")
    
    if sent_results:
        fastest_sent = min(sent_results, key=lambda x: x['response_time'])
        print(f"⚡ Fastest Sentiment: {fastest_sent['model_name']} ({fastest_sent['response_time']:.2f}s)")

async def test_single_model(api_key: str, model_name: str, test_type: str = "both"):
    """Test a single model"""
    print(f"🧪 Testing {model_name}")
    print("=" * 30)
    
    if model_name not in AVAILABLE_MODELS:
        print(f"❌ Model {model_name} not found in available models")
        return
    
    # Create test data
    test_df, news_items, social_posts = create_test_data()
    
    if test_type in ["prediction", "both"]:
        print("\n🔮 Testing Prediction...")
        pred_result = await test_prediction_model(api_key, model_name, test_df)
        
        if pred_result['success']:
            print(f"✅ Success! Response time: {pred_result['response_time']:.2f}s")
            print(f"Direction: {pred_result['prediction']['direction']}")
            print(f"Confidence: {pred_result['prediction']['confidence']:.2f}")
            print(f"Reasoning: {pred_result['prediction']['reasoning']}")
        else:
            print(f"❌ Failed: {pred_result['error']}")
    
    if test_type in ["sentiment", "both"]:
        print("\n😊 Testing Sentiment...")
        sent_result = await test_sentiment_model(api_key, model_name, news_items, social_posts)
        
        if sent_result['success']:
            print(f"✅ Success! Response time: {sent_result['response_time']:.2f}s")
            print(f"Overall Sentiment: {sent_result['sentiment']['overall_sentiment']:.2f}")
            print(f"Confidence: {sent_result['sentiment']['confidence']:.2f}")
            print(f"Key Themes: {', '.join(sent_result['sentiment']['key_themes'])}")
            print(f"Summary: {sent_result['sentiment']['summary']}")
        else:
            print(f"❌ Failed: {sent_result['error']}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Test Gemini models for forex trading")
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--model', type=str, help='Test specific model')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark all models')
    parser.add_argument('--type', choices=['prediction', 'sentiment', 'both'], default='both', help='Type of test to run')
    parser.add_argument('--models', nargs='+', help='Specific models to benchmark')
    
    args = parser.parse_args()
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    if args.list:
        print_available_models()
        return
    
    if args.model:
        asyncio.run(test_single_model(api_key, args.model, args.type))
    elif args.benchmark:
        results = asyncio.run(benchmark_models(api_key, args.models))
        print_benchmark_summary(results)
    else:
        print("🤖 Gemini Model Tester")
        print("Use --help for options")
        print("\nQuick examples:")
        print("  python model_tester.py --list")
        print("  python model_tester.py --model gemma-3n-e4b-it")
        print("  python model_tester.py --benchmark")
        print("  python model_tester.py --benchmark --models gemini-1.5-pro gemma-3n-e4b-it")

if __name__ == "__main__":
    main()