"""
Gemini-powered Sentiment Analyzer for Financial Markets
Uses Google's Gemma 3 27B model for intelligent sentiment analysis
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dataclasses import dataclass
import time
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

@dataclass
class NewsItem:
    """News item structure"""
    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    symbol: Optional[str] = None

@dataclass
class SocialPost:
    """Social media post structure"""
    content: str
    platform: str
    author: str
    engagement: int  # likes, retweets, etc.
    posted_at: datetime
    symbol: Optional[str] = None

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    symbol: str
    overall_sentiment: float  # -1.0 (very bearish) to 1.0 (very bullish)
    confidence: float  # 0.0 to 1.0
    key_themes: List[str]
    news_sentiment: float
    social_sentiment: float
    risk_factors: List[str]
    opportunities: List[str]
    summary: str
    timestamp: datetime

class GeminiSentimentAnalyzer:
    """Gemini-powered financial sentiment analyzer"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """
        Initialize Gemini sentiment analyzer
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self._setup_gemini()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        
        # Sentiment cache
        self.sentiment_cache = {}
        self.cache_duration = 600  # 10 minutes for sentiment
        
        # Financial keywords for relevance filtering
        self.forex_keywords = {
            'EUR/USD': ['euro', 'dollar', 'eurusd', 'eur/usd', 'ecb', 'fed'],
            'GBP/USD': ['pound', 'sterling', 'gbpusd', 'gbp/usd', 'boe', 'cable'],
            'USD/JPY': ['yen', 'usdjpy', 'usd/jpy', 'boj', 'dollar-yen'],
            'AUD/USD': ['aussie', 'audusd', 'aud/usd', 'rba', 'australian'],
            'USD/CHF': ['franc', 'usdchf', 'usd/chf', 'snb', 'swiss'],
            'NZD/USD': ['kiwi', 'nzdusd', 'nzd/usd', 'rbnz', 'new zealand'],
            'USD/CAD': ['loonie', 'usdcad', 'usd/cad', 'boc', 'canadian']
        }
    
    def _setup_gemini(self):
        """Setup Gemini AI client"""
        try:
            genai.configure(api_key=self.api_key)
            
            # Configure safety settings for financial analysis
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
            
            # Initialize model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings
            )
            
            logger.info(f"Gemini sentiment model {self.model_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini sentiment analyzer: {e}")
            raise
    
    def _rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, symbol: str, data_hash: str) -> str:
        """Generate cache key for sentiment analysis"""
        timestamp = int(time.time() // self.cache_duration)
        return f"sentiment_{symbol}_{data_hash}_{timestamp}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached sentiment is still valid"""
        return cache_key in self.sentiment_cache
    
    def _calculate_data_hash(self, news_items: List[NewsItem], social_posts: List[SocialPost]) -> str:
        """Calculate hash of input data for caching"""
        content_list = []
        
        for item in news_items:
            content_list.append(f"{item.title}_{item.published_at}")
        
        for post in social_posts:
            content_list.append(f"{post.content[:50]}_{post.posted_at}")
        
        content_str = "".join(content_list)
        return str(hash(content_str))
    
    def filter_relevant_content(
        self, 
        symbol: str, 
        news_items: List[NewsItem], 
        social_posts: List[SocialPost]
    ) -> Tuple[List[NewsItem], List[SocialPost]]:
        """Filter content relevant to the specific forex pair"""
        
        keywords = self.forex_keywords.get(symbol, [])
        if not keywords:
            return news_items, social_posts
        
        # Filter news items
        relevant_news = []
        for item in news_items:
            content_lower = f"{item.title} {item.content}".lower()
            if any(keyword in content_lower for keyword in keywords):
                item.symbol = symbol
                relevant_news.append(item)
        
        # Filter social posts
        relevant_posts = []
        for post in social_posts:
            content_lower = post.content.lower()
            if any(keyword in content_lower for keyword in keywords):
                post.symbol = symbol
                relevant_posts.append(post)
        
        logger.info(f"Filtered to {len(relevant_news)} news items and {len(relevant_posts)} social posts for {symbol}")
        
        return relevant_news, relevant_posts
    
    def _format_sentiment_analysis_prompt(
        self, 
        symbol: str, 
        news_items: List[NewsItem], 
        social_posts: List[SocialPost]
    ) -> str:
        """Format comprehensive sentiment analysis prompt for Gemini"""
        
        # Prepare news content
        news_content = ""
        for i, item in enumerate(news_items[:10], 1):  # Limit to 10 items
            news_content += f"""
News {i}:
Title: {item.title}
Source: {item.source}
Published: {item.published_at.strftime('%Y-%m-%d %H:%M')}
Content: {item.content[:500]}...
---
"""
        
        # Prepare social content
        social_content = ""
        for i, post in enumerate(social_posts[:20], 1):  # Limit to 20 posts
            social_content += f"""
Post {i}:
Platform: {post.platform}
Engagement: {post.engagement}
Posted: {post.posted_at.strftime('%Y-%m-%d %H:%M')}
Content: {post.content[:200]}...
---
"""
        
        prompt = f"""
You are an expert financial sentiment analyst specializing in forex markets. Analyze the following news articles and social media posts related to {symbol} and provide a comprehensive sentiment assessment.

ANALYSIS TARGET: {symbol}
ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

NEWS ARTICLES:
{news_content if news_content else "No news articles provided"}

SOCIAL MEDIA POSTS:
{social_content if social_content else "No social media posts provided"}

ANALYSIS REQUIREMENTS:
1. Assess overall market sentiment (-1.0 to 1.0 scale)
2. Separate news sentiment from social sentiment
3. Identify key themes and market drivers
4. Highlight risk factors and opportunities
5. Provide confidence level in the analysis
6. Consider the credibility and recency of sources

SENTIMENT SCALE:
- -1.0 to -0.6: Very Bearish (Strong selling pressure expected)
- -0.6 to -0.2: Bearish (Moderate selling pressure)
- -0.2 to 0.2: Neutral (Sideways movement expected)
- 0.2 to 0.6: Bullish (Moderate buying pressure)
- 0.6 to 1.0: Very Bullish (Strong buying pressure expected)

RESPONSE FORMAT (JSON):
{{
    "overall_sentiment": 0.3,
    "confidence": 0.85,
    "news_sentiment": 0.4,
    "social_sentiment": 0.2,
    "key_themes": ["Central bank policy", "Economic data", "Geopolitical tensions"],
    "risk_factors": ["Factor 1", "Factor 2"],
    "opportunities": ["Opportunity 1", "Opportunity 2"],
    "summary": "Comprehensive 2-3 sentence summary of the sentiment analysis including key drivers and outlook",
    "market_drivers": {{
        "fundamental": ["Economic indicators", "Central bank policy"],
        "technical": ["Support/resistance levels", "Chart patterns"],
        "sentiment": ["Risk appetite", "Market positioning"]
    }},
    "time_sensitivity": "HIGH|MEDIUM|LOW",
    "credibility_score": 0.8
}}

Provide only the JSON response without any additional text or markdown formatting.
"""
        return prompt
    
    async def analyze_sentiment(
        self, 
        symbol: str, 
        news_items: List[NewsItem], 
        social_posts: List[SocialPost]
    ) -> SentimentResult:
        """Analyze sentiment for a forex pair using news and social data"""
        
        # Filter relevant content
        filtered_news, filtered_posts = self.filter_relevant_content(symbol, news_items, social_posts)
        
        # Check cache
        data_hash = self._calculate_data_hash(filtered_news, filtered_posts)
        cache_key = self._get_cache_key(symbol, data_hash)
        
        if self._is_cache_valid(cache_key):
            logger.info(f"Returning cached sentiment for {symbol}")
            return self.sentiment_cache[cache_key]
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Generate prompt
            prompt = self._format_sentiment_analysis_prompt(symbol, filtered_news, filtered_posts)
            
            # Get sentiment analysis from Gemini
            logger.info(f"Requesting sentiment analysis from Gemini for {symbol}")
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Low temperature for consistent analysis
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1024,
                )
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Clean JSON response
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            # Parse JSON
            try:
                sentiment_data = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON response, using fallback parsing")
                sentiment_data = self._parse_fallback_sentiment_response(response_text)
            
            # Create sentiment result
            sentiment = SentimentResult(
                symbol=symbol,
                overall_sentiment=float(sentiment_data.get('overall_sentiment', 0.0)),
                confidence=float(sentiment_data.get('confidence', 0.5)),
                key_themes=sentiment_data.get('key_themes', []),
                news_sentiment=float(sentiment_data.get('news_sentiment', 0.0)),
                social_sentiment=float(sentiment_data.get('social_sentiment', 0.0)),
                risk_factors=sentiment_data.get('risk_factors', []),
                opportunities=sentiment_data.get('opportunities', []),
                summary=sentiment_data.get('summary', 'Sentiment analysis completed'),
                timestamp=datetime.now()
            )
            
            # Validate sentiment
            sentiment = self._validate_sentiment(sentiment)
            
            # Cache the result
            self.sentiment_cache[cache_key] = sentiment
            
            logger.info(f"Generated sentiment for {symbol}: {sentiment.overall_sentiment:.2f} with {sentiment.confidence:.2f} confidence")
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error generating sentiment for {symbol}: {e}")
            return self._generate_fallback_sentiment(symbol)
    
    def _parse_fallback_sentiment_response(self, response_text: str) -> Dict:
        """Parse response when JSON parsing fails"""
        fallback = {
            'overall_sentiment': 0.0,
            'confidence': 0.3,
            'news_sentiment': 0.0,
            'social_sentiment': 0.0,
            'key_themes': ['Market analysis'],
            'risk_factors': ['Analysis error'],
            'opportunities': ['Monitor for updates'],
            'summary': 'Fallback sentiment analysis due to parsing error'
        }
        
        # Try to extract basic sentiment
        response_lower = response_text.lower()
        if 'bullish' in response_lower or 'positive' in response_lower:
            fallback['overall_sentiment'] = 0.3
        elif 'bearish' in response_lower or 'negative' in response_lower:
            fallback['overall_sentiment'] = -0.3
        
        return fallback
    
    def _validate_sentiment(self, sentiment: SentimentResult) -> SentimentResult:
        """Validate and adjust sentiment parameters"""
        
        # Validate sentiment bounds
        sentiment.overall_sentiment = max(-1.0, min(1.0, sentiment.overall_sentiment))
        sentiment.news_sentiment = max(-1.0, min(1.0, sentiment.news_sentiment))
        sentiment.social_sentiment = max(-1.0, min(1.0, sentiment.social_sentiment))
        sentiment.confidence = max(0.0, min(1.0, sentiment.confidence))
        
        # Ensure lists are not empty
        if not sentiment.key_themes:
            sentiment.key_themes = ['Market analysis']
        
        if not sentiment.risk_factors:
            sentiment.risk_factors = ['Monitor market conditions']
        
        if not sentiment.opportunities:
            sentiment.opportunities = ['Watch for trend changes']
        
        return sentiment
    
    def _generate_fallback_sentiment(self, symbol: str) -> SentimentResult:
        """Generate neutral sentiment when analysis fails"""
        return SentimentResult(
            symbol=symbol,
            overall_sentiment=0.0,
            confidence=0.2,
            key_themes=['Analysis unavailable'],
            news_sentiment=0.0,
            social_sentiment=0.0,
            risk_factors=['Limited data'],
            opportunities=['Monitor for updates'],
            summary='Neutral sentiment due to analysis limitations',
            timestamp=datetime.now()
        )
    
    async def batch_sentiment_analysis(
        self, 
        symbols: List[str], 
        news_items: List[NewsItem], 
        social_posts: List[SocialPost]
    ) -> Dict[str, SentimentResult]:
        """Analyze sentiment for multiple symbols"""
        
        results = {}
        
        for symbol in symbols:
            try:
                sentiment = await self.analyze_sentiment(symbol, news_items, social_posts)
                results[symbol] = sentiment
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                results[symbol] = self._generate_fallback_sentiment(symbol)
        
        return results
    
    def get_sentiment_summary(self, symbol: str, sentiment: SentimentResult) -> Dict:
        """Get human-readable sentiment summary"""
        
        def sentiment_to_text(score: float) -> str:
            if score >= 0.6:
                return "Very Bullish"
            elif score >= 0.2:
                return "Bullish"
            elif score >= -0.2:
                return "Neutral"
            elif score >= -0.6:
                return "Bearish"
            else:
                return "Very Bearish"
        
        return {
            'symbol': symbol,
            'sentiment_text': sentiment_to_text(sentiment.overall_sentiment),
            'sentiment_score': sentiment.overall_sentiment,
            'confidence_percentage': int(sentiment.confidence * 100),
            'key_themes': sentiment.key_themes[:3],  # Top 3 themes
            'summary': sentiment.summary,
            'last_updated': sentiment.timestamp.strftime('%Y-%m-%d %H:%M UTC')
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the sentiment model"""
        return {
            'model_name': self.model_name,
            'provider': 'Google Gemini',
            'capabilities': [
                'Financial News Analysis',
                'Social Media Sentiment',
                'Multi-source Sentiment Fusion',
                'Risk Factor Identification',
                'Market Theme Extraction'
            ],
            'supported_symbols': list(self.forex_keywords.keys()),
            'max_requests_per_minute': 60,
            'cache_duration_minutes': self.cache_duration // 60
        }