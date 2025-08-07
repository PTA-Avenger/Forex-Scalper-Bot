#!/usr/bin/env python3
"""
Linux Trading Bot - Receives signals from MT5 Windows VM and processes them with AI
"""

import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import aiohttp
import google.generativeai as genai
from aiohttp import web
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        # Configuration
        self.config = {
            'influxdb_url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
            'influxdb_token': os.getenv('INFLUXDB_TOKEN', 'your-token-here'),
            'influxdb_org': os.getenv('INFLUXDB_ORG', 'trading-org'),
            'influxdb_bucket': os.getenv('INFLUXDB_BUCKET', 'trading-signals'),
            'gemini_api_key': os.getenv('GEMINI_API_KEY', 'your-gemini-key-here'),
            'bot_port': int(os.getenv('BOT_PORT', 8080)),
            'mt5_vm_ip': os.getenv('MT5_VM_IP', '192.168.1.101'),
        }
        
        # Initialize InfluxDB client
        self.influx_client = InfluxDBClient(
            url=self.config['influxdb_url'],
            token=self.config['influxdb_token'],
            org=self.config['influxdb_org']
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Initialize Gemini AI
        genai.configure(api_key=self.config['gemini_api_key'])
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Trading state
        self.signal_history = []
        self.active_positions = {}
        
        logger.info("Trading Bot initialized successfully")

    async def handle_signal(self, request):
        """Handle incoming signals from MT5"""
        try:
            signal_data = await request.json()
            logger.info(f"Received signal: {signal_data.get('symbol', 'Unknown')}")
            
            # Store raw signal in InfluxDB
            await self.store_signal(signal_data)
            
            # Process signal with AI
            ai_decision = await self.process_with_ai(signal_data)
            
            # Store AI decision
            await self.store_ai_decision(signal_data, ai_decision)
            
            # Execute trading decision if needed
            execution_result = await self.execute_decision(signal_data, ai_decision)
            
            # Store execution result
            if execution_result:
                await self.store_execution_result(signal_data, ai_decision, execution_result)
            
            response = {
                'status': 'success',
                'signal_received': True,
                'ai_decision': ai_decision,
                'execution_result': execution_result,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Error handling signal: {str(e)}")
            logger.error(traceback.format_exc())
            return web.json_response(
                {'status': 'error', 'message': str(e)}, 
                status=500
            )

    async def handle_trade_execution(self, request):
        """Handle trade execution updates from MT5"""
        try:
            execution_data = await request.json()
            logger.info(f"Received trade execution: {execution_data.get('ticket', 'Unknown')}")
            
            # Store execution data in InfluxDB
            await self.store_trade_execution(execution_data)
            
            return web.json_response({'status': 'success'})
            
        except Exception as e:
            logger.error(f"Error handling trade execution: {str(e)}")
            return web.json_response(
                {'status': 'error', 'message': str(e)}, 
                status=500
            )

    async def store_signal(self, signal_data: Dict[str, Any]):
        """Store raw signal data in InfluxDB"""
        try:
            point = (
                Point("market_signals")
                .tag("symbol", signal_data.get('symbol', 'UNKNOWN'))
                .tag("source", signal_data.get('source', 'MT5'))
                .field("bid", float(signal_data.get('bid', 0)))
                .field("ask", float(signal_data.get('ask', 0)))
                .field("spread", float(signal_data.get('spread', 0)))
                .field("volume", int(signal_data.get('volume', 0)))
                .time(datetime.now(timezone.utc))
            )
            
            # Add OHLC data if available
            if 'ohlc' in signal_data:
                ohlc = signal_data['ohlc']
                point = point.field("open", float(ohlc.get('open', 0)))
                point = point.field("high", float(ohlc.get('high', 0)))
                point = point.field("low", float(ohlc.get('low', 0)))
                point = point.field("close", float(ohlc.get('close', 0)))
            
            # Add indicator data if available
            if 'indicators' in signal_data:
                indicators = signal_data['indicators']
                for key, value in indicators.items():
                    if isinstance(value, (int, float)) and value != 0:
                        point = point.field(f"indicator_{key}", float(value))
            
            self.write_api.write(
                bucket=self.config['influxdb_bucket'],
                org=self.config['influxdb_org'],
                record=point
            )
            
            logger.info(f"Stored signal for {signal_data.get('symbol')} in InfluxDB")
            
        except Exception as e:
            logger.error(f"Error storing signal: {str(e)}")

    async def process_with_ai(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process signal with Gemini AI to make trading decision"""
        try:
            # Prepare context for AI
            market_context = self.prepare_market_context(signal_data)
            
            # Create AI prompt
            prompt = self.create_ai_prompt(signal_data, market_context)
            
            # Generate AI response
            response = await asyncio.to_thread(
                self.gemini_model.generate_content, prompt
            )
            
            # Parse AI decision
            ai_decision = self.parse_ai_response(response.text)
            
            logger.info(f"AI Decision for {signal_data.get('symbol')}: {ai_decision.get('action', 'HOLD')}")
            
            return ai_decision
            
        except Exception as e:
            logger.error(f"Error in AI processing: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': f'AI processing error: {str(e)}',
                'risk_level': 'HIGH'
            }

    def prepare_market_context(self, signal_data: Dict[str, Any]) -> str:
        """Prepare market context for AI analysis"""
        symbol = signal_data.get('symbol', 'UNKNOWN')
        
        context = f"""
        Market Data for {symbol}:
        - Current Bid: {signal_data.get('bid', 'N/A')}
        - Current Ask: {signal_data.get('ask', 'N/A')}
        - Spread: {signal_data.get('spread', 'N/A')}
        - Volume: {signal_data.get('volume', 'N/A')}
        """
        
        # Add OHLC data
        if 'ohlc' in signal_data:
            ohlc = signal_data['ohlc']
            context += f"""
        - Open: {ohlc.get('open', 'N/A')}
        - High: {ohlc.get('high', 'N/A')}
        - Low: {ohlc.get('low', 'N/A')}
        - Close: {ohlc.get('close', 'N/A')}
        """
        
        # Add technical indicators
        if 'indicators' in signal_data:
            indicators = signal_data['indicators']
            context += "\nTechnical Indicators:"
            for key, value in indicators.items():
                context += f"\n- {key.upper()}: {value}"
        
        return context

    def create_ai_prompt(self, signal_data: Dict[str, Any], market_context: str) -> str:
        """Create AI prompt for trading decision"""
        return f"""
        You are an expert forex trading AI assistant. Analyze the following market data and provide a trading recommendation.

        {market_context}

        Based on this data, provide your analysis in the following JSON format:
        {{
            "action": "BUY|SELL|HOLD",
            "confidence": 0.0-1.0,
            "reasoning": "Detailed explanation of your decision",
            "risk_level": "LOW|MEDIUM|HIGH",
            "stop_loss": suggested_stop_loss_price,
            "take_profit": suggested_take_profit_price,
            "position_size": suggested_position_size_percentage,
            "time_horizon": "SHORT|MEDIUM|LONG"
        }}

        Consider:
        1. Technical indicator signals (RSI, Moving Averages, Bollinger Bands, ATR)
        2. Price action and momentum
        3. Risk management principles
        4. Current market conditions
        5. Spread and liquidity factors

        Be conservative and prioritize capital preservation. Only recommend BUY/SELL if you have high confidence.
        """

    def parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response and extract trading decision"""
        try:
            # Try to extract JSON from the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                decision = json.loads(json_str)
                
                # Validate required fields
                required_fields = ['action', 'confidence', 'reasoning', 'risk_level']
                for field in required_fields:
                    if field not in decision:
                        decision[field] = 'UNKNOWN'
                
                return decision
            else:
                raise ValueError("No valid JSON found in AI response")
                
        except Exception as e:
            logger.error(f"Error parsing AI response: {str(e)}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': f'Failed to parse AI response: {response_text[:200]}...',
                'risk_level': 'HIGH'
            }

    async def store_ai_decision(self, signal_data: Dict[str, Any], ai_decision: Dict[str, Any]):
        """Store AI decision in InfluxDB"""
        try:
            point = (
                Point("ai_decisions")
                .tag("symbol", signal_data.get('symbol', 'UNKNOWN'))
                .tag("action", ai_decision.get('action', 'HOLD'))
                .tag("risk_level", ai_decision.get('risk_level', 'UNKNOWN'))
                .field("confidence", float(ai_decision.get('confidence', 0.0)))
                .field("reasoning", ai_decision.get('reasoning', ''))
                .time(datetime.now(timezone.utc))
            )
            
            # Add optional fields if present
            if 'stop_loss' in ai_decision:
                point = point.field("stop_loss", float(ai_decision['stop_loss']))
            if 'take_profit' in ai_decision:
                point = point.field("take_profit", float(ai_decision['take_profit']))
            if 'position_size' in ai_decision:
                point = point.field("position_size", float(ai_decision['position_size']))
            
            self.write_api.write(
                bucket=self.config['influxdb_bucket'],
                org=self.config['influxdb_org'],
                record=point
            )
            
            logger.info(f"Stored AI decision for {signal_data.get('symbol')} in InfluxDB")
            
        except Exception as e:
            logger.error(f"Error storing AI decision: {str(e)}")

    async def execute_decision(self, signal_data: Dict[str, Any], ai_decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute trading decision (placeholder for actual execution logic)"""
        try:
            action = ai_decision.get('action', 'HOLD')
            confidence = ai_decision.get('confidence', 0.0)
            
            # Only execute if confidence is high enough
            if action in ['BUY', 'SELL'] and confidence >= 0.7:
                # This is where you would integrate with your actual trading execution
                # For now, we'll simulate execution
                execution_result = {
                    'executed': True,
                    'action': action,
                    'symbol': signal_data.get('symbol'),
                    'simulated': True,  # Remove this when implementing real execution
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'confidence': confidence
                }
                
                logger.info(f"Simulated execution: {action} {signal_data.get('symbol')} (confidence: {confidence})")
                return execution_result
            else:
                logger.info(f"Decision not executed - Action: {action}, Confidence: {confidence}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing decision: {str(e)}")
            return None

    async def store_execution_result(self, signal_data: Dict[str, Any], ai_decision: Dict[str, Any], execution_result: Dict[str, Any]):
        """Store execution result in InfluxDB"""
        try:
            point = (
                Point("trade_executions")
                .tag("symbol", signal_data.get('symbol', 'UNKNOWN'))
                .tag("action", execution_result.get('action', 'UNKNOWN'))
                .tag("executed", str(execution_result.get('executed', False)))
                .field("confidence", float(ai_decision.get('confidence', 0.0)))
                .field("simulated", execution_result.get('simulated', True))
                .time(datetime.now(timezone.utc))
            )
            
            self.write_api.write(
                bucket=self.config['influxdb_bucket'],
                org=self.config['influxdb_org'],
                record=point
            )
            
            logger.info(f"Stored execution result for {signal_data.get('symbol')} in InfluxDB")
            
        except Exception as e:
            logger.error(f"Error storing execution result: {str(e)}")

    async def store_trade_execution(self, execution_data: Dict[str, Any]):
        """Store trade execution data from MT5"""
        try:
            point = (
                Point("mt5_executions")
                .tag("symbol", execution_data.get('symbol', 'UNKNOWN'))
                .tag("type", execution_data.get('type', 'UNKNOWN'))
                .field("ticket", int(execution_data.get('ticket', 0)))
                .field("volume", float(execution_data.get('volume', 0)))
                .field("price", float(execution_data.get('price', 0)))
                .time(datetime.now(timezone.utc))
            )
            
            self.write_api.write(
                bucket=self.config['influxdb_bucket'],
                org=self.config['influxdb_org'],
                record=point
            )
            
            logger.info(f"Stored MT5 execution for ticket {execution_data.get('ticket')} in InfluxDB")
            
        except Exception as e:
            logger.error(f"Error storing MT5 execution: {str(e)}")

    async def health_check(self, request):
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.0.0'
        })

    async def get_statistics(self, request):
        """Get trading statistics"""
        try:
            # Query InfluxDB for recent statistics
            query = f'''
            from(bucket: "{self.config['influxdb_bucket']}")
              |> range(start: -24h)
              |> filter(fn: (r) => r["_measurement"] == "ai_decisions")
              |> group(columns: ["action"])
              |> count()
            '''
            
            # This is a simplified version - you would implement proper querying
            stats = {
                'signals_received_24h': len(self.signal_history),
                'last_signal_time': datetime.now(timezone.utc).isoformat(),
                'active_positions': len(self.active_positions),
                'bot_uptime': 'N/A'  # Implement uptime tracking
            }
            
            return web.json_response(stats)
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            return web.json_response(
                {'status': 'error', 'message': str(e)}, 
                status=500
            )

    def create_app(self):
        """Create aiohttp web application"""
        app = web.Application()
        
        # Add routes
        app.router.add_post('/api/signals', self.handle_signal)
        app.router.add_post('/api/trade-execution', self.handle_trade_execution)
        app.router.add_get('/api/health', self.health_check)
        app.router.add_get('/api/statistics', self.get_statistics)
        
        return app

    async def start_server(self):
        """Start the web server"""
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.config['bot_port'])
        await site.start()
        
        logger.info(f"Trading bot server started on port {self.config['bot_port']}")
        
        # Keep the server running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down trading bot...")
        finally:
            await runner.cleanup()

def main():
    """Main function"""
    try:
        # Create and start the trading bot
        bot = TradingBot()
        asyncio.run(bot.start_server())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()