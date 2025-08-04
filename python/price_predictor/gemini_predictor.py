"""
Gemini-powered Price Predictor for Forex Markets
Uses Google's Gemma 3 27B model for intelligent market analysis
"""

import logging
import json
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import ta
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data structure for analysis"""
    symbol: str
    timeframe: str
    ohlcv: pd.DataFrame
    indicators: Dict
    sentiment_score: Optional[float] = None
    news_summary: Optional[str] = None

@dataclass
class PredictionResult:
    """Prediction result structure"""
    symbol: str
    direction: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    target_price: Optional[float]
    stop_loss: Optional[float]
    time_horizon: str  # "1h", "4h", "1d"
    reasoning: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    timestamp: datetime

class GeminiPredictor:
    """Gemini-powered forex price predictor"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        """
        Initialize Gemini predictor
        
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
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_duration = 300  # 5 minutes
        
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
            
            logger.info(f"Gemini model {self.model_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def _rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        """Generate cache key for predictions"""
        timestamp = int(time.time() // self.cache_duration)
        return f"{symbol}_{timeframe}_{timestamp}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached prediction is still valid"""
        return cache_key in self.prediction_cache
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        try:
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50).iloc[-1]
            indicators['ema_12'] = ta.trend.ema_indicator(df['close'], window=12).iloc[-1]
            indicators['ema_26'] = ta.trend.ema_indicator(df['close'], window=26).iloc[-1]
            
            # MACD
            macd_line = ta.trend.macd(df['close'])
            macd_signal = ta.trend.macd_signal(df['close'])
            indicators['macd'] = macd_line.iloc[-1] if not macd_line.empty else 0
            indicators['macd_signal'] = macd_signal.iloc[-1] if not macd_signal.empty else 0
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # RSI
            indicators['rsi'] = ta.momentum.rsi(df['close'], window=14).iloc[-1]
            
            # Bollinger Bands
            bb_high = ta.volatility.bollinger_hband(df['close'])
            bb_low = ta.volatility.bollinger_lband(df['close'])
            bb_mid = ta.volatility.bollinger_mavg(df['close'])
            indicators['bb_upper'] = bb_high.iloc[-1] if not bb_high.empty else df['close'].iloc[-1]
            indicators['bb_lower'] = bb_low.iloc[-1] if not bb_low.empty else df['close'].iloc[-1]
            indicators['bb_middle'] = bb_mid.iloc[-1] if not bb_mid.empty else df['close'].iloc[-1]
            indicators['bb_position'] = (df['close'].iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # ATR (Average True Range)
            indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']).iloc[-1]
            
            # Stochastic
            stoch_k = ta.momentum.stoch(df['high'], df['low'], df['close'])
            stoch_d = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            indicators['stoch_k'] = stoch_k.iloc[-1] if not stoch_k.empty else 50
            indicators['stoch_d'] = stoch_d.iloc[-1] if not stoch_d.empty else 50
            
            # Volume indicators
            if 'volume' in df.columns:
                indicators['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume']).iloc[-1]
                indicators['volume_ratio'] = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            else:
                indicators['volume_sma'] = 0
                indicators['volume_ratio'] = 1
            
            # Price action
            indicators['current_price'] = df['close'].iloc[-1]
            indicators['price_change_24h'] = (df['close'].iloc[-1] - df['close'].iloc[-24]) / df['close'].iloc[-24] * 100 if len(df) >= 24 else 0
            indicators['volatility'] = df['close'].pct_change().std() * np.sqrt(24) * 100
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def _format_market_analysis_prompt(self, market_data: MarketData) -> str:
        """Format comprehensive market analysis prompt for Gemini"""
        
        indicators = market_data.indicators
        current_price = indicators.get('current_price', 0)
        
        prompt = f"""
You are an expert forex trading analyst with 20+ years of experience. Analyze the following market data for {market_data.symbol} and provide a detailed trading recommendation.

MARKET DATA:
- Symbol: {market_data.symbol}
- Timeframe: {market_data.timeframe}
- Current Price: {current_price:.5f}
- 24h Change: {indicators.get('price_change_24h', 0):.2f}%

TECHNICAL INDICATORS:
- SMA 20: {indicators.get('sma_20', 0):.5f}
- SMA 50: {indicators.get('sma_50', 0):.5f}
- EMA 12: {indicators.get('ema_12', 0):.5f}
- EMA 26: {indicators.get('ema_26', 0):.5f}
- RSI (14): {indicators.get('rsi', 0):.2f}
- MACD: {indicators.get('macd', 0):.5f}
- MACD Signal: {indicators.get('macd_signal', 0):.5f}
- MACD Histogram: {indicators.get('macd_histogram', 0):.5f}
- Bollinger Bands: Upper {indicators.get('bb_upper', 0):.5f}, Middle {indicators.get('bb_middle', 0):.5f}, Lower {indicators.get('bb_lower', 0):.5f}
- BB Position: {indicators.get('bb_position', 0):.2f} (0=lower band, 1=upper band)
- ATR: {indicators.get('atr', 0):.5f}
- Stochastic K: {indicators.get('stoch_k', 0):.2f}
- Stochastic D: {indicators.get('stoch_d', 0):.2f}
- Volatility: {indicators.get('volatility', 0):.2f}%

SENTIMENT DATA:
- Market Sentiment Score: {market_data.sentiment_score if market_data.sentiment_score else 'N/A'}
- News Summary: {market_data.news_summary if market_data.news_summary else 'No recent news'}

ANALYSIS REQUIREMENTS:
1. Determine trend direction (bullish, bearish, sideways)
2. Identify key support and resistance levels
3. Assess momentum and volatility
4. Consider sentiment factors
5. Provide clear trading recommendation

RESPONSE FORMAT (JSON):
{{
    "direction": "BUY|SELL|HOLD",
    "confidence": 0.85,
    "target_price": 1.09500,
    "stop_loss": 1.09200,
    "time_horizon": "4h",
    "risk_level": "MEDIUM",
    "reasoning": "Detailed explanation of the analysis and recommendation including key technical levels, trend analysis, and risk factors",
    "key_levels": {{
        "support": [1.09200, 1.09000],
        "resistance": [1.09600, 1.09800]
    }},
    "risk_factors": ["factor1", "factor2"],
    "confluence_factors": ["Moving average crossover", "RSI oversold", "Bullish MACD divergence"]
}}

Provide only the JSON response without any additional text or markdown formatting.
"""
        return prompt
    
    async def predict_price_movement(self, market_data: MarketData) -> PredictionResult:
        """Generate price movement prediction using Gemini"""
        
        # Check cache first
        cache_key = self._get_cache_key(market_data.symbol, market_data.timeframe)
        if self._is_cache_valid(cache_key):
            logger.info(f"Returning cached prediction for {market_data.symbol}")
            return self.prediction_cache[cache_key]
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Generate prompt
            prompt = self._format_market_analysis_prompt(market_data)
            
            # Get prediction from Gemini
            logger.info(f"Requesting prediction from Gemini for {market_data.symbol}")
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more consistent financial analysis
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1024,
                )
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Clean JSON response (remove any markdown formatting)
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            # Parse JSON
            try:
                prediction_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback parsing if JSON is malformed
                logger.warning("Malformed JSON response, using fallback parsing")
                prediction_data = self._parse_fallback_response(response_text, market_data)
            
            # Create prediction result
            prediction = PredictionResult(
                symbol=market_data.symbol,
                direction=prediction_data.get('direction', 'HOLD'),
                confidence=float(prediction_data.get('confidence', 0.5)),
                target_price=float(prediction_data.get('target_price', 0)) if prediction_data.get('target_price') else None,
                stop_loss=float(prediction_data.get('stop_loss', 0)) if prediction_data.get('stop_loss') else None,
                time_horizon=prediction_data.get('time_horizon', '1h'),
                reasoning=prediction_data.get('reasoning', 'AI analysis completed'),
                risk_level=prediction_data.get('risk_level', 'MEDIUM'),
                timestamp=datetime.now()
            )
            
            # Validate prediction
            prediction = self._validate_prediction(prediction, market_data)
            
            # Cache the prediction
            self.prediction_cache[cache_key] = prediction
            
            logger.info(f"Generated prediction for {market_data.symbol}: {prediction.direction} with {prediction.confidence:.2f} confidence")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {market_data.symbol}: {e}")
            # Return safe fallback prediction
            return self._generate_fallback_prediction(market_data)
    
    def _parse_fallback_response(self, response_text: str, market_data: MarketData) -> Dict:
        """Parse response when JSON parsing fails"""
        fallback = {
            'direction': 'HOLD',
            'confidence': 0.5,
            'target_price': market_data.indicators.get('current_price', 0),
            'stop_loss': market_data.indicators.get('current_price', 0) * 0.99,
            'time_horizon': '1h',
            'risk_level': 'MEDIUM',
            'reasoning': 'Fallback analysis due to parsing error'
        }
        
        # Try to extract basic information
        response_lower = response_text.lower()
        if 'buy' in response_lower or 'bullish' in response_lower:
            fallback['direction'] = 'BUY'
        elif 'sell' in response_lower or 'bearish' in response_lower:
            fallback['direction'] = 'SELL'
        
        return fallback
    
    def _validate_prediction(self, prediction: PredictionResult, market_data: MarketData) -> PredictionResult:
        """Validate and adjust prediction parameters"""
        current_price = market_data.indicators.get('current_price', 0)
        atr = market_data.indicators.get('atr', current_price * 0.001)
        
        # Validate confidence bounds
        prediction.confidence = max(0.0, min(1.0, prediction.confidence))
        
        # Validate target and stop loss prices
        if prediction.target_price is None or prediction.target_price <= 0:
            if prediction.direction == 'BUY':
                prediction.target_price = current_price + atr * 2
            elif prediction.direction == 'SELL':
                prediction.target_price = current_price - atr * 2
            else:
                prediction.target_price = current_price
        
        if prediction.stop_loss is None or prediction.stop_loss <= 0:
            if prediction.direction == 'BUY':
                prediction.stop_loss = current_price - atr
            elif prediction.direction == 'SELL':
                prediction.stop_loss = current_price + atr
            else:
                prediction.stop_loss = current_price
        
        # Ensure logical price relationships
        if prediction.direction == 'BUY':
            if prediction.target_price <= current_price:
                prediction.target_price = current_price + atr * 2
            if prediction.stop_loss >= current_price:
                prediction.stop_loss = current_price - atr
        elif prediction.direction == 'SELL':
            if prediction.target_price >= current_price:
                prediction.target_price = current_price - atr * 2
            if prediction.stop_loss <= current_price:
                prediction.stop_loss = current_price + atr
        
        return prediction
    
    def _generate_fallback_prediction(self, market_data: MarketData) -> PredictionResult:
        """Generate safe fallback prediction when Gemini fails"""
        current_price = market_data.indicators.get('current_price', 0)
        atr = market_data.indicators.get('atr', current_price * 0.001)
        
        return PredictionResult(
            symbol=market_data.symbol,
            direction='HOLD',
            confidence=0.3,
            target_price=current_price,
            stop_loss=current_price - atr,
            time_horizon='1h',
            reasoning='Fallback prediction due to API error',
            risk_level='LOW',
            timestamp=datetime.now()
        )
    
    async def analyze_multiple_timeframes(
        self, 
        symbol: str, 
        market_data_dict: Dict[str, MarketData]
    ) -> Dict[str, PredictionResult]:
        """Analyze multiple timeframes for comprehensive view"""
        
        results = {}
        
        # Analyze each timeframe
        for timeframe, market_data in market_data_dict.items():
            try:
                prediction = await self.predict_price_movement(market_data)
                results[timeframe] = prediction
            except Exception as e:
                logger.error(f"Error analyzing {timeframe} for {symbol}: {e}")
                results[timeframe] = self._generate_fallback_prediction(market_data)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            'model_name': self.model_name,
            'provider': 'Google Gemini',
            'capabilities': [
                'Technical Analysis',
                'Market Sentiment',
                'Multi-timeframe Analysis',
                'Risk Assessment'
            ],
            'supported_symbols': 'All major forex pairs',
            'max_requests_per_minute': 60
        }