import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import logging
from datetime import datetime
from typing import Tuple, Optional, List
import os

logger = logging.getLogger(__name__)

class LSTMPredictor:
    """
    LSTM-based price prediction model for forex trading
    
    Features:
    - Multi-layer LSTM architecture with attention mechanism
    - Dropout and batch normalization for regularization
    - Dynamic sequence length based on data availability
    - Multiple prediction horizons
    - Confidence estimation
    """
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = StandardScaler()
        
        # Model parameters
        self.sequence_length = getattr(config, 'LSTM_SEQUENCE_LENGTH', 60)
        self.features = getattr(config, 'LSTM_FEATURES', ['close', 'volume', 'high', 'low', 'open'])
        self.prediction_horizon = getattr(config, 'LSTM_PREDICTION_HORIZON', 1)
        
        # Architecture parameters
        self.lstm_units = getattr(config, 'LSTM_UNITS', [128, 64, 32])
        self.dropout_rate = getattr(config, 'LSTM_DROPOUT', 0.2)
        self.learning_rate = getattr(config, 'LSTM_LEARNING_RATE', 0.001)
        self.batch_size = getattr(config, 'LSTM_BATCH_SIZE', 32)
        self.epochs = getattr(config, 'LSTM_EPOCHS', 100)
        
        # Training state
        self.is_trained = False
        self.training_history = None
        self.last_trained = None
        self.confidence_threshold = 0.7
        
        logger.info(f"LSTM Predictor initialized with sequence length: {self.sequence_length}")
    
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build the LSTM model architecture
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(
            self.lstm_units[0],
            return_sequences=True,
            input_shape=input_shape,
            kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
        ))
        model.add(Dropout(self.dropout_rate))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        if len(self.lstm_units) > 1:
            model.add(LSTM(
                self.lstm_units[1],
                return_sequences=len(self.lstm_units) > 2,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            ))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())
        
        # Third LSTM layer (optional)
        if len(self.lstm_units) > 2:
            model.add(LSTM(
                self.lstm_units[2],
                return_sequences=False,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01)
            ))
            model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout_rate / 2))
        
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(self.dropout_rate / 2))
        
        # Output layer
        model.add(Dense(self.prediction_horizon, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info(f"LSTM model built with {model.count_params()} parameters")
        return model
    
    def prepare_sequences(self, data: np.ndarray, target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for LSTM training/prediction
        
        Args:
            data: Input data array
            target: Target data array (optional, for training)
            
        Returns:
            Tuple of (sequences, targets)
        """
        sequences = []
        targets = [] if target is not None else None
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
            
            # Target sequence (if provided)
            if target is not None:
                if self.prediction_horizon == 1:
                    targets.append(target[i + self.sequence_length])
                else:
                    targets.append(target[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        sequences = np.array(sequences)
        targets = np.array(targets) if targets else None
        
        logger.debug(f"Prepared {len(sequences)} sequences with shape {sequences.shape}")
        return sequences, targets
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for training/prediction
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features, target)
        """
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create additional features
        df = data.copy()
        
        # Technical indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])
        df['atr'] = self._calculate_atr(df)
        
        # Price ratios
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Select features for model
        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'log_returns', 'volatility', 'rsi',
            'sma_20', 'ema_12', 'ema_26', 'macd',
            'bb_upper', 'bb_lower', 'atr',
            'high_low_ratio', 'close_open_ratio',
            'volume_ratio'
        ]
        
        # Add time features if available
        if 'hour' in df.columns:
            feature_columns.extend(['hour', 'day_of_week', 'month'])
        
        # Remove NaN values
        df = df.dropna()
        
        if len(df) < self.sequence_length + self.prediction_horizon:
            raise ValueError(f"Insufficient data: need at least {self.sequence_length + self.prediction_horizon} rows")
        
        # Extract features and target
        features = df[feature_columns].values
        target = df['close'].values
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()
        
        logger.info(f"Preprocessed data: {features_scaled.shape[0]} samples, {features_scaled.shape[1]} features")
        return features_scaled, target_scaled
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> dict:
        """
        Train the LSTM model
        
        Args:
            data: Training data DataFrame
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history dictionary
        """
        try:
            logger.info("Starting LSTM model training...")
            
            # Preprocess data
            features, target = self.preprocess_data(data)
            
            # Prepare sequences
            X, y = self.prepare_sequences(features, target)
            
            if len(X) == 0:
                raise ValueError("No sequences generated from data")
            
            # Build model
            self.model = self.build_model((self.sequence_length, features.shape[1]))
            
            # Prepare callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath='models/lstm_best_model.h5',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model
            history = self.model.fit(
                X, y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            self.training_history = history.history
            self.is_trained = True
            self.last_trained = datetime.utcnow()
            
            # Calculate training metrics
            train_loss = min(history.history['loss'])
            val_loss = min(history.history['val_loss'])
            
            logger.info(f"Training completed - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            return {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epochs_trained': len(history.history['loss']),
                'training_time': (datetime.utcnow() - self.last_trained).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame, horizon: int = 1) -> np.ndarray:
        """
        Generate price predictions
        
        Args:
            data: Input data DataFrame
            horizon: Number of time steps to predict
            
        Returns:
            Array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Preprocess data
            features, _ = self.preprocess_data(data)
            
            # Use the last sequence for prediction
            if len(features) < self.sequence_length:
                raise ValueError(f"Need at least {self.sequence_length} data points for prediction")
            
            last_sequence = features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(horizon):
                # Predict next value
                pred_scaled = self.model.predict(current_sequence, verbose=0)
                pred_price = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
                predictions.append(pred_price)
                
                # Update sequence for next prediction (simplified approach)
                # In practice, you'd want to update all features, not just the price
                if horizon > 1:
                    # Create new feature vector (simplified)
                    new_features = current_sequence[0, -1, :].copy()
                    new_features[3] = pred_scaled[0, 0]  # Update close price feature
                    
                    # Shift sequence and add new features
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1, :] = new_features
            
            predictions = np.array(predictions)
            logger.debug(f"Generated {len(predictions)} predictions")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_confidence(self) -> float:
        """
        Calculate prediction confidence based on model performance
        
        Returns:
            Confidence score between 0 and 1
        """
        if not self.training_history:
            return 0.5
        
        # Base confidence on validation loss
        val_losses = self.training_history.get('val_loss', [])
        if not val_losses:
            return 0.5
        
        final_val_loss = val_losses[-1]
        min_val_loss = min(val_losses)
        
        # Calculate confidence based on loss stability and magnitude
        loss_stability = 1.0 - (final_val_loss - min_val_loss) / (min_val_loss + 1e-8)
        loss_magnitude = 1.0 / (1.0 + final_val_loss)
        
        confidence = (loss_stability + loss_magnitude) / 2.0
        return max(0.0, min(1.0, confidence))
    
    def save_model(self, filepath: str):
        """Save model and scalers"""
        if self.model is not None:
            # Save model
            self.model.save(f"{filepath}_model.h5")
            
            # Save scalers
            joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
            joblib.dump(self.feature_scaler, f"{filepath}_feature_scaler.pkl")
            
            # Save metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'features': self.features,
                'prediction_horizon': self.prediction_horizon,
                'last_trained': self.last_trained.isoformat() if self.last_trained else None,
                'is_trained': self.is_trained
            }
            joblib.dump(metadata, f"{filepath}_metadata.pkl")
            
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and scalers"""
        try:
            # Load model
            self.model = load_model(f"{filepath}_model.h5")
            
            # Load scalers
            self.scaler = joblib.load(f"{filepath}_scaler.pkl")
            self.feature_scaler = joblib.load(f"{filepath}_feature_scaler.pkl")
            
            # Load metadata
            metadata = joblib.load(f"{filepath}_metadata.pkl")
            self.sequence_length = metadata['sequence_length']
            self.features = metadata['features']
            self.prediction_horizon = metadata['prediction_horizon']
            self.last_trained = datetime.fromisoformat(metadata['last_trained']) if metadata['last_trained'] else None
            self.is_trained = metadata['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_architecture(self) -> dict:
        """Get model architecture information"""
        if self.model is None:
            return {}
        
        return {
            'layers': len(self.model.layers),
            'parameters': self.model.count_params(),
            'lstm_units': self.lstm_units,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'optimizer': self.model.optimizer.__class__.__name__,
            'learning_rate': float(self.model.optimizer.learning_rate)
        }
    
    def get_last_trained(self) -> Optional[str]:
        """Get last training timestamp"""
        return self.last_trained.isoformat() if self.last_trained else None
    
    # Helper methods for technical indicators
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, lower_band
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()