#!/usr/bin/env python3
"""
Test script for Gemini integration
Verifies that the Gemini API integration works correctly
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from price_predictor.gemini_predictor import GeminiPredictor, MarketData
from sentiment_analyzer.gemini_sentiment import GeminiSentimentAnalyzer, NewsItem, SocialPost

def create_sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Generate realistic EUR/USD price data
    np.random.seed(42)
    base_price = 1.0950
    returns = np.random.normal(0, 0.0005, 100)
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    ohlcv_data = []
    for i in range(len(dates)):
        price = prices[i]
        high = price * (1 + abs(np.random.normal(0, 0.0002)))
        low = price * (1 - abs(np.random.normal(0, 0.0002)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.randint(100, 1000)
        
        ohlcv_data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(ohlcv_data)

def create_sample_news_data():
    """Create sample news data for sentiment testing"""
    news_items = [
        NewsItem(
            title="EUR/USD Analysis: Euro Strengthens on ECB Policy Outlook",
            content="The Euro has gained ground against the US Dollar following positive economic indicators from the Eurozone. Market analysts expect continued strength in the coming sessions.",
            source="Financial Times",
            url="https://example.com/news1",
            published_at=datetime.now() - timedelta(hours=2)
        ),
        NewsItem(
            title="Federal Reserve Signals Potential Rate Adjustments",
            content="The Federal Reserve has indicated potential changes to monetary policy in response to recent economic data, which could impact USD strength in forex markets.",
            source="Reuters",
            url="https://example.com/news2",
            published_at=datetime.now() - timedelta(hours=4)
        )
    ]
    return news_items

def create_sample_social_data():
    """Create sample social media data for sentiment testing"""
    social_posts = [
        SocialPost(
            content="EUR/USD looking bullish! Technical indicators suggesting upward momentum. #forex #trading",
            platform="Twitter",
            author="trader_pro",
            engagement=45,
            posted_at=datetime.now() - timedelta(hours=1)
        ),
        SocialPost(
            content="Dollar weakness continuing across major pairs. Watch for EUR/USD breakout above 1.10",
            platform="Reddit",
            author="forex_analyst",
            engagement=123,
            posted_at=datetime.now() - timedelta(hours=3)
        )
    ]
    return social_posts

async def test_gemini_predictor():
    """Test the Gemini price predictor"""
    print("🧪 Testing Gemini Price Predictor...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize predictor
        predictor = GeminiPredictor(api_key)
        print("✅ Gemini predictor initialized successfully")
        
        # Create sample data
        df = create_sample_ohlcv_data()
        print(f"📊 Created sample OHLCV data with {len(df)} points")
        
        # Calculate technical indicators
        indicators = predictor.calculate_technical_indicators(df)
        print(f"📈 Calculated {len(indicators)} technical indicators")
        
        # Create market data
        market_data = MarketData(
            symbol="EUR/USD",
            timeframe="1h",
            ohlcv=df,
            indicators=indicators,
            sentiment_score=0.3,
            news_summary="Positive economic outlook for Eurozone"
        )
        
        # Generate prediction
        print("🔮 Generating prediction...")
        prediction = await predictor.predict_price_movement(market_data)
        
        print(f"✅ Prediction generated successfully:")
        print(f"   Direction: {prediction.direction}")
        print(f"   Confidence: {prediction.confidence:.2f}")
        print(f"   Target Price: {prediction.target_price}")
        print(f"   Stop Loss: {prediction.stop_loss}")
        print(f"   Risk Level: {prediction.risk_level}")
        print(f"   Reasoning: {prediction.reasoning[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini predictor: {e}")
        return False

async def test_gemini_sentiment():
    """Test the Gemini sentiment analyzer"""
    print("\n🧪 Testing Gemini Sentiment Analyzer...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize sentiment analyzer
        analyzer = GeminiSentimentAnalyzer(api_key)
        print("✅ Gemini sentiment analyzer initialized successfully")
        
        # Create sample data
        news_items = create_sample_news_data()
        social_posts = create_sample_social_data()
        print(f"📰 Created {len(news_items)} news items and {len(social_posts)} social posts")
        
        # Generate sentiment analysis
        print("😊 Analyzing sentiment...")
        sentiment = await analyzer.analyze_sentiment("EUR/USD", news_items, social_posts)
        
        print(f"✅ Sentiment analysis completed:")
        print(f"   Overall Sentiment: {sentiment.overall_sentiment:.2f}")
        print(f"   Confidence: {sentiment.confidence:.2f}")
        print(f"   News Sentiment: {sentiment.news_sentiment:.2f}")
        print(f"   Social Sentiment: {sentiment.social_sentiment:.2f}")
        print(f"   Key Themes: {', '.join(sentiment.key_themes[:3])}")
        print(f"   Summary: {sentiment.summary[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Gemini sentiment analyzer: {e}")
        return False

async def test_multi_timeframe_analysis():
    """Test multi-timeframe analysis"""
    print("\n🧪 Testing Multi-Timeframe Analysis...")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
    
    try:
        predictor = GeminiPredictor(api_key)
        
        # Create market data for different timeframes
        market_data_dict = {}
        for timeframe in ['1h', '4h', '1d']:
            df = create_sample_ohlcv_data()
            indicators = predictor.calculate_technical_indicators(df)
            
            market_data_dict[timeframe] = MarketData(
                symbol="EUR/USD",
                timeframe=timeframe,
                ohlcv=df,
                indicators=indicators
            )
        
        print(f"📊 Created market data for {len(market_data_dict)} timeframes")
        
        # Analyze multiple timeframes
        print("🔄 Analyzing multiple timeframes...")
        results = await predictor.analyze_multiple_timeframes("EUR/USD", market_data_dict)
        
        print(f"✅ Multi-timeframe analysis completed:")
        for timeframe, prediction in results.items():
            print(f"   {timeframe}: {prediction.direction} (conf: {prediction.confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing multi-timeframe analysis: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Gemini Integration Tests")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("❌ Please set GEMINI_API_KEY environment variable")
        print("   Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    tests = [
        test_gemini_predictor,
        test_gemini_sentiment,
        test_multi_timeframe_analysis
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Gemini integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    # Check if running in async context
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        sys.exit(1)