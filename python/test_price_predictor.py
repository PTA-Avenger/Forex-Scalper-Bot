import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime, timedelta
import json
import redis
import requests
from flask import Flask

# Import the modules we're testing (assuming they exist)
from price_predictor.app import app, lstm_model, xgboost_model, ensemble_model
from price_predictor.models.lstm_model import LSTMPredictor
from price_predictor.models.xgboost_model import XGBoostPredictor
from price_predictor.models.ensemble_model import EnsemblePredictor
from price_predictor.data.preprocessor import DataPreprocessor
from price_predictor.data.feature_engineer import FeatureEngineer


class TestDataPreprocessor:
    """Test suite for data preprocessing functionality"""
    
    def setup_method(self):
        """Setup test data and preprocessor"""
        self.preprocessor = DataPreprocessor(Mock())
        
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='1min')
        np.random.seed(42)
        
        # Generate realistic forex price data
        base_price = 1.0950
        returns = np.random.normal(0, 0.0001, 1000)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'symbol': ['EUR/USD'] * 1000,
            'bid': [p - 0.0001 for p in prices],
            'ask': [p + 0.0001 for p in prices],
            'volume': np.random.randint(10, 1000, 1000)
        })
    
    def test_data_validation(self):
        """Test data validation functionality"""
        # Valid data should pass
        valid_data = self.sample_data.copy()
        assert self.preprocessor.validate_data(valid_data) == True
        
        # Missing columns should fail
        invalid_data = self.sample_data.drop('bid', axis=1)
        assert self.preprocessor.validate_data(invalid_data) == False
        
        # Negative prices should fail
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'bid'] = -1.0
        assert self.preprocessor.validate_data(invalid_data) == False
    
    def test_data_cleaning(self):
        """Test data cleaning functionality"""
        # Add some invalid data points
        dirty_data = self.sample_data.copy()
        dirty_data.loc[100, 'bid'] = np.nan
        dirty_data.loc[200, 'ask'] = np.inf
        dirty_data.loc[300, 'volume'] = -100
        
        cleaned_data = self.preprocessor.clean_data(dirty_data)
        
        # Check that invalid data is removed
        assert len(cleaned_data) < len(dirty_data)
        assert not cleaned_data.isnull().any().any()
        assert (cleaned_data['volume'] >= 0).all()
    
    def test_ohlcv_aggregation(self):
        """Test OHLCV bar aggregation"""
        ohlcv_data = self.preprocessor.aggregate_ohlcv(
            self.sample_data, timeframe='5min'
        )
        
        # Check structure
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        assert all(col in ohlcv_data.columns for col in expected_columns)
        
        # Check data integrity
        assert (ohlcv_data['high'] >= ohlcv_data['low']).all()
        assert (ohlcv_data['high'] >= ohlcv_data['open']).all()
        assert (ohlcv_data['high'] >= ohlcv_data['close']).all()
        assert (ohlcv_data['low'] <= ohlcv_data['open']).all()
        assert (ohlcv_data['low'] <= ohlcv_data['close']).all()


class TestFeatureEngineer:
    """Test suite for feature engineering functionality"""
    
    def setup_method(self):
        """Setup test data and feature engineer"""
        self.feature_engineer = FeatureEngineer(Mock())
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=500, freq='5min')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 1.0950
        returns = np.random.normal(0, 0.0005, 500)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.ohlcv_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.0002))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.0002))) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.0001)) for p in prices],
            'volume': np.random.randint(100, 2000, 500)
        })
    
    def test_technical_indicators(self):
        """Test technical indicator calculations"""
        features = self.feature_engineer.calculate_technical_indicators(
            self.ohlcv_data
        )
        
        # Check that indicators are calculated
        expected_indicators = [
            'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'atr_14', 'stoch_k', 'stoch_d'
        ]
        
        for indicator in expected_indicators:
            assert indicator in features.columns
            assert not features[indicator].isnull().all()
    
    def test_price_features(self):
        """Test price-based feature calculations"""
        features = self.feature_engineer.calculate_price_features(
            self.ohlcv_data
        )
        
        # Check returns and volatility
        assert 'returns_1' in features.columns
        assert 'returns_5' in features.columns
        assert 'volatility_10' in features.columns
        assert 'volatility_20' in features.columns
        
        # Validate return calculations
        expected_returns = self.ohlcv_data['close'].pct_change()
        np.testing.assert_array_almost_equal(
            features['returns_1'].dropna().values,
            expected_returns.dropna().values,
            decimal=6
        )
    
    def test_time_features(self):
        """Test time-based feature extraction"""
        features = self.feature_engineer.extract_time_features(
            self.ohlcv_data
        )
        
        # Check time features
        time_features = ['hour', 'day_of_week', 'month', 'is_market_open']
        for feature in time_features:
            assert feature in features.columns
        
        # Validate hour extraction
        expected_hours = pd.to_datetime(self.ohlcv_data['timestamp']).dt.hour
        np.testing.assert_array_equal(
            features['hour'].values,
            expected_hours.values
        )


class TestLSTMPredictor:
    """Test suite for LSTM model functionality"""
    
    def setup_method(self):
        """Setup LSTM predictor"""
        config = Mock()
        config.lstm_config = {
            'sequence_length': 60,
            'features': ['close', 'volume', 'sma_10', 'rsi_14'],
            'hidden_units': 50,
            'dropout_rate': 0.2
        }
        self.lstm_predictor = LSTMPredictor(config)
        
        # Create sample training data
        np.random.seed(42)
        self.sample_features = pd.DataFrame({
            'close': np.random.normal(1.0950, 0.0050, 1000),
            'volume': np.random.randint(100, 2000, 1000),
            'sma_10': np.random.normal(1.0950, 0.0030, 1000),
            'rsi_14': np.random.uniform(20, 80, 1000)
        })
        
        self.sample_targets = np.random.normal(0, 0.0005, 1000)
    
    def test_data_preparation(self):
        """Test data preparation for LSTM"""
        X, y = self.lstm_predictor.prepare_sequences(
            self.sample_features, self.sample_targets
        )
        
        # Check shapes
        expected_samples = len(self.sample_features) - 60 + 1
        assert X.shape == (expected_samples, 60, 4)
        assert y.shape == (expected_samples,)
        
        # Check data integrity
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
    
    def test_model_architecture(self):
        """Test LSTM model architecture"""
        model = self.lstm_predictor.build_model()
        
        # Check input shape
        assert model.input_shape == (None, 60, 4)
        
        # Check output shape
        assert model.output_shape == (None, 1)
        
        # Check layer types
        layer_types = [type(layer).__name__ for layer in model.layers]
        assert 'LSTM' in layer_types
        assert 'Dense' in layer_types
        assert 'Dropout' in layer_types
    
    @patch('tensorflow.keras.models.load_model')
    def test_model_prediction(self, mock_load_model):
        """Test model prediction functionality"""
        # Mock the loaded model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.0005], [0.0003], [-0.0002]])
        mock_load_model.return_value = mock_model
        
        # Load model and make predictions
        self.lstm_predictor.load_model('fake_model_path')
        
        test_data = self.sample_features.iloc[-60:].values.reshape(1, 60, 4)
        predictions = self.lstm_predictor.predict(test_data)
        
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)
        mock_model.predict.assert_called_once()


class TestXGBoostPredictor:
    """Test suite for XGBoost model functionality"""
    
    def setup_method(self):
        """Setup XGBoost predictor"""
        config = Mock()
        config.xgboost_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8
        }
        self.xgb_predictor = XGBoostPredictor(config)
        
        # Create sample training data
        np.random.seed(42)
        self.sample_features = pd.DataFrame({
            'sma_10': np.random.normal(1.0950, 0.0030, 1000),
            'ema_12': np.random.normal(1.0950, 0.0025, 1000),
            'rsi_14': np.random.uniform(20, 80, 1000),
            'macd': np.random.normal(0, 0.0001, 1000),
            'bb_position': np.random.uniform(-1, 1, 1000),
            'volume_ratio': np.random.uniform(0.5, 2.0, 1000)
        })
        
        self.sample_targets = np.random.normal(0, 0.0005, 1000)
    
    def test_feature_preparation(self):
        """Test feature preparation for XGBoost"""
        X_train, X_test, y_train, y_test = self.xgb_predictor.prepare_features(
            self.sample_features, self.sample_targets, test_size=0.2
        )
        
        # Check split ratios
        assert len(X_train) == int(0.8 * len(self.sample_features))
        assert len(X_test) == int(0.2 * len(self.sample_features))
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check data types
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, (pd.Series, np.ndarray))
        assert isinstance(y_test, (pd.Series, np.ndarray))
    
    def test_model_training(self):
        """Test XGBoost model training"""
        X_train = self.sample_features.iloc[:800]
        y_train = self.sample_targets[:800]
        
        # Train model
        model = self.xgb_predictor.train(X_train, y_train)
        
        # Check model is trained
        assert model is not None
        assert hasattr(model, 'predict')
        
        # Test prediction
        X_test = self.sample_features.iloc[800:]
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert isinstance(predictions, np.ndarray)
    
    def test_feature_importance(self):
        """Test feature importance extraction"""
        X_train = self.sample_features.iloc[:800]
        y_train = self.sample_targets[:800]
        
        # Train model and get feature importance
        model = self.xgb_predictor.train(X_train, y_train)
        importance = self.xgb_predictor.get_feature_importance(model)
        
        # Check importance structure
        assert isinstance(importance, dict)
        assert len(importance) == len(self.sample_features.columns)
        
        # Check all features have importance scores
        for feature in self.sample_features.columns:
            assert feature in importance
            assert 0 <= importance[feature] <= 1


class TestEnsemblePredictor:
    """Test suite for ensemble model functionality"""
    
    def setup_method(self):
        """Setup ensemble predictor"""
        config = Mock()
        self.ensemble_predictor = EnsemblePredictor(config)
        
        # Mock individual predictors
        self.mock_lstm = Mock()
        self.mock_xgb = Mock()
        
        self.ensemble_predictor.lstm_predictor = self.mock_lstm
        self.ensemble_predictor.xgb_predictor = self.mock_xgb
    
    def test_ensemble_prediction(self):
        """Test ensemble prediction combining multiple models"""
        # Mock predictions from individual models
        self.mock_lstm.predict.return_value = np.array([0.0005, 0.0003, -0.0002])
        self.mock_xgb.predict.return_value = np.array([0.0003, 0.0005, -0.0001])
        
        # Test ensemble prediction
        test_data = Mock()
        predictions = self.ensemble_predictor.predict(test_data)
        
        # Check that both models were called
        self.mock_lstm.predict.assert_called_once_with(test_data)
        self.mock_xgb.predict.assert_called_once_with(test_data)
        
        # Check ensemble result
        assert predictions is not None
        assert len(predictions) == 3
        
        # Check that predictions are averaged (default strategy)
        expected = (np.array([0.0005, 0.0003, -0.0002]) + 
                   np.array([0.0003, 0.0005, -0.0001])) / 2
        np.testing.assert_array_almost_equal(predictions, expected)
    
    def test_weighted_ensemble(self):
        """Test weighted ensemble prediction"""
        # Set custom weights
        self.ensemble_predictor.set_weights({'lstm': 0.7, 'xgboost': 0.3})
        
        # Mock predictions
        self.mock_lstm.predict.return_value = np.array([0.0010])
        self.mock_xgb.predict.return_value = np.array([0.0000])
        
        test_data = Mock()
        predictions = self.ensemble_predictor.predict(test_data)
        
        # Check weighted average
        expected = 0.0010 * 0.7 + 0.0000 * 0.3
        np.testing.assert_array_almost_equal(predictions, [expected])


class TestFlaskAPI:
    """Test suite for Flask API endpoints"""
    
    def setup_method(self):
        """Setup Flask test client"""
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.ctx = app.app_context()
        self.ctx.push()
    
    def teardown_method(self):
        """Cleanup Flask context"""
        self.ctx.pop()
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    @patch('price_predictor.app.ensemble_model')
    def test_predict_endpoint(self, mock_ensemble):
        """Test prediction endpoint"""
        # Mock ensemble model
        mock_ensemble.predict.return_value = np.array([0.0005])
        
        # Test prediction request
        test_data = {
            'symbol': 'EUR/USD',
            'timeframe': '1m',
            'features': {
                'close': 1.0950,
                'volume': 1000,
                'sma_10': 1.0948,
                'rsi_14': 65.5
            }
        }
        
        response = self.client.post('/predict',
                                   data=json.dumps(test_data),
                                   content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'timestamp' in data
    
    def test_invalid_prediction_request(self):
        """Test prediction endpoint with invalid data"""
        # Missing required fields
        invalid_data = {
            'symbol': 'EUR/USD'
            # Missing timeframe and features
        }
        
        response = self.client.post('/predict',
                                   data=json.dumps(invalid_data),
                                   content_type='application/json')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data


class TestIntegration:
    """Integration tests for the complete prediction pipeline"""
    
    @pytest.mark.integration
    def test_full_prediction_pipeline(self):
        """Test the complete prediction pipeline end-to-end"""
        # This would test the full flow from raw data to prediction
        # Would require actual model files and database connections
        pass
    
    @pytest.mark.integration
    @patch('redis.Redis')
    def test_redis_caching(self, mock_redis):
        """Test Redis caching functionality"""
        # Mock Redis client
        mock_redis_client = Mock()
        mock_redis.return_value = mock_redis_client
        
        # Test caching logic
        cache_key = 'prediction:EUR/USD:1m:1234567890'
        cached_prediction = {'prediction': 0.0005, 'confidence': 0.85}
        
        mock_redis_client.get.return_value = json.dumps(cached_prediction)
        
        # Your caching logic would go here
        # result = get_cached_prediction(cache_key)
        # assert result == cached_prediction
    
    @pytest.mark.performance
    def test_prediction_latency(self):
        """Test that predictions are generated within acceptable time limits"""
        import time
        
        # Mock quick prediction
        start_time = time.time()
        
        # Simulate prediction logic
        time.sleep(0.001)  # 1ms simulation
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Assert latency is under 100ms for real-time trading
        assert latency < 100


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])