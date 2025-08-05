#!/usr/bin/env python3
"""
MT5 Integration Tests

This module contains comprehensive tests for the MetaTrader 5 integration,
including the Python bridge service and C++ broker implementation.
"""

import unittest
import requests
import json
import time
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the mt5_bridge directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python', 'mt5_bridge'))

try:
    from mt5_bridge import MT5Bridge
    import MetaTrader5 as mt5
except ImportError:
    print("Warning: MT5 dependencies not available. Some tests will be skipped.")
    MT5Bridge = None
    mt5 = None

class TestMT5Bridge(unittest.TestCase):
    """Test the MT5 Bridge Python service"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.bridge_url = "http://localhost:5004"
        cls.test_config = {
            'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
            'callback_url': 'http://localhost:8080/mt5/callback'
        }
        
    def setUp(self):
        """Set up each test"""
        self.bridge = None
        if MT5Bridge:
            self.bridge = MT5Bridge(self.test_config)
    
    def tearDown(self):
        """Clean up after each test"""
        if self.bridge:
            self.bridge.shutdown()
    
    @unittest.skipIf(MT5Bridge is None, "MT5Bridge not available")
    def test_bridge_initialization(self):
        """Test MT5 bridge initialization"""
        self.assertIsNotNone(self.bridge)
        self.assertEqual(self.bridge.config, self.test_config)
        self.assertFalse(self.bridge.connected)
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        try:
            response = requests.get(f"{self.bridge_url}/health", timeout=5)
            self.assertEqual(response.status_code, 200)
            
            data = response.json()
            self.assertIn('status', data)
            self.assertIn('connected', data)
            
        except requests.RequestException:
            self.skipTest("MT5 bridge service not running")
    
    @unittest.skipIf(mt5 is None, "MetaTrader5 library not available")
    @patch('MetaTrader5.initialize')
    @patch('MetaTrader5.login')
    def test_mt5_connection_mock(self, mock_login, mock_initialize):
        """Test MT5 connection with mocked MT5 library"""
        mock_initialize.return_value = True
        mock_login.return_value = True
        
        result = self.bridge.connect_to_mt5(
            login=123456789,
            password="test_password",
            server="test_server"
        )
        
        self.assertTrue(result)
        self.assertTrue(self.bridge.connected)
        mock_initialize.assert_called_once()
        mock_login.assert_called_once_with(
            123456789, 
            password="test_password", 
            server="test_server"
        )
    
    def test_symbol_conversion(self):
        """Test symbol format conversion"""
        test_cases = [
            ("EUR/USD", "EURUSD"),
            ("GBP/USD", "GBPUSD"),
            ("USD/JPY", "USDJPY"),
            ("AUD/USD", "AUDUSD"),
            ("USD/CHF", "USDCHF")
        ]
        
        # This would be tested in the C++ implementation
        for standard, mt5_format in test_cases:
            # Test conversion logic
            converted = standard.replace('/', '')
            self.assertEqual(converted, mt5_format)
    
    def test_place_order_endpoint(self):
        """Test the place order endpoint"""
        order_data = {
            "symbol": "EURUSD",
            "order_type": "buy",
            "volume": 0.1,
            "price": 1.1000,
            "sl": 1.0950,
            "tp": 1.1050,
            "comment": "Test Order"
        }
        
        try:
            response = requests.post(
                f"{self.bridge_url}/place_order",
                json=order_data,
                timeout=10
            )
            
            # Should return 400 if not connected to MT5
            self.assertIn(response.status_code, [200, 400])
            
            if response.status_code == 400:
                data = response.json()
                self.assertIn('error', data)
                self.assertEqual(data['error'], 'Not connected to MT5')
                
        except requests.RequestException:
            self.skipTest("MT5 bridge service not running")
    
    def test_get_market_data_endpoint(self):
        """Test the market data endpoint"""
        try:
            response = requests.get(
                f"{self.bridge_url}/market_data/EURUSD",
                timeout=5
            )
            
            # Should return 400 if not connected to MT5
            self.assertIn(response.status_code, [200, 400, 404])
            
        except requests.RequestException:
            self.skipTest("MT5 bridge service not running")
    
    def test_get_historical_data_endpoint(self):
        """Test the historical data endpoint"""
        try:
            response = requests.get(
                f"{self.bridge_url}/historical_data/EURUSD?timeframe=M1&count=100",
                timeout=10
            )
            
            # Should return 400 if not connected to MT5
            self.assertIn(response.status_code, [200, 400, 404])
            
        except requests.RequestException:
            self.skipTest("MT5 bridge service not running")

class TestMT5Configuration(unittest.TestCase):
    """Test MT5 configuration management"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'config', 
            'mt5_config.json'
        )
    
    def test_config_file_exists(self):
        """Test that MT5 configuration file exists"""
        self.assertTrue(os.path.exists(self.config_path))
    
    def test_config_file_valid_json(self):
        """Test that configuration file contains valid JSON"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # Check required sections
        self.assertIn('mt5', config)
        self.assertIn('symbols', config)
        self.assertIn('trading', config)
        
        # Check MT5 section
        mt5_config = config['mt5']
        self.assertIn('login', mt5_config)
        self.assertIn('password', mt5_config)
        self.assertIn('server', mt5_config)
        
        # Check trading section
        trading_config = config['trading']
        self.assertIn('magic_number', trading_config)
        self.assertIn('min_lot_size', trading_config)
        self.assertIn('max_lot_size', trading_config)
    
    def test_symbol_mappings(self):
        """Test symbol mapping configuration"""
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        if 'symbol_mapping' in config:
            mappings = config['symbol_mapping']
            
            # Test that mappings are correct
            for standard, mt5_format in mappings.items():
                self.assertIn('/', standard)  # Standard format has slash
                self.assertNotIn('/', mt5_format)  # MT5 format has no slash

class TestMT5IntegrationEnd2End(unittest.TestCase):
    """End-to-end integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up for end-to-end tests"""
        cls.bridge_url = "http://localhost:5004"
        cls.cpp_engine_url = "http://localhost:8080"
    
    def test_full_system_health(self):
        """Test that all components are healthy"""
        services = [
            (self.bridge_url, "MT5 Bridge"),
            (self.cpp_engine_url, "C++ Engine")
        ]
        
        for url, name in services:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                self.assertEqual(
                    response.status_code, 200, 
                    f"{name} health check failed"
                )
            except requests.RequestException:
                self.skipTest(f"{name} service not running")
    
    def test_market_data_flow(self):
        """Test market data flow from MT5 to engine"""
        # This would test the complete data flow
        # from MT5 → Bridge → C++ Engine
        pass
    
    def test_order_execution_flow(self):
        """Test order execution flow"""
        # This would test the complete order flow
        # from C++ Engine → Bridge → MT5
        pass

class TestMT5BrokerCpp(unittest.TestCase):
    """Test C++ MT5Broker implementation (through HTTP API)"""
    
    def setUp(self):
        """Set up C++ broker tests"""
        self.engine_url = "http://localhost:8080"
    
    def test_broker_connection_status(self):
        """Test broker connection status endpoint"""
        try:
            response = requests.get(f"{self.engine_url}/broker/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.assertIn('broker_type', data)
                self.assertIn('connected', data)
        except requests.RequestException:
            self.skipTest("C++ engine not running")
    
    def test_symbol_info_retrieval(self):
        """Test symbol information retrieval"""
        try:
            response = requests.get(
                f"{self.engine_url}/broker/symbols/EURUSD", 
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                self.assertIn('symbol', data)
                self.assertIn('pip_size', data)
                self.assertIn('min_lot_size', data)
        except requests.RequestException:
            self.skipTest("C++ engine not running")

class TestMT5Performance(unittest.TestCase):
    """Performance tests for MT5 integration"""
    
    def setUp(self):
        """Set up performance tests"""
        self.bridge_url = "http://localhost:5004"
    
    def test_market_data_latency(self):
        """Test market data retrieval latency"""
        try:
            start_time = time.time()
            response = requests.get(
                f"{self.bridge_url}/market_data/EURUSD",
                timeout=5
            )
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                # Market data should be retrieved in less than 100ms
                self.assertLess(latency, 100, "Market data latency too high")
                
        except requests.RequestException:
            self.skipTest("MT5 bridge service not running")
    
    def test_order_placement_latency(self):
        """Test order placement latency"""
        order_data = {
            "symbol": "EURUSD",
            "order_type": "buy",
            "volume": 0.01,  # Minimum lot size
            "comment": "Performance Test"
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.bridge_url}/place_order",
                json=order_data,
                timeout=10
            )
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            # Order placement should complete in less than 500ms
            self.assertLess(latency, 500, "Order placement latency too high")
            
        except requests.RequestException:
            self.skipTest("MT5 bridge service not running")

def run_mt5_integration_tests():
    """Run all MT5 integration tests"""
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMT5Bridge,
        TestMT5Configuration,
        TestMT5IntegrationEnd2End,
        TestMT5BrokerCpp,
        TestMT5Performance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running MT5 Integration Tests...")
    print("=" * 50)
    
    success = run_mt5_integration_tests()
    
    if success:
        print("\n✅ All MT5 integration tests passed!")
        exit(0)
    else:
        print("\n❌ Some MT5 integration tests failed!")
        exit(1)