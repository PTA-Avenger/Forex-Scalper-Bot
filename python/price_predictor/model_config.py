"""
Model Configuration for Gemini-powered Forex Trading Bot
Supports multiple Gemini models including Gemma variants
"""

from typing import Dict, Any
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a specific Gemini model"""
    name: str
    display_name: str
    max_tokens: int
    temperature_range: tuple
    recommended_temperature: float
    rate_limit_rpm: int  # requests per minute
    context_window: int
    multimodal: bool = False
    description: str = ""

# Available Gemini models configuration
AVAILABLE_MODELS = {
    # Standard Gemini models
    "gemini-1.5-pro": ModelConfig(
        name="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        max_tokens=8192,
        temperature_range=(0.0, 2.0),
        recommended_temperature=0.3,
        rate_limit_rpm=60,
        context_window=1000000,  # 1M tokens
        multimodal=True,
        description="Most capable model for complex reasoning and analysis"
    ),
    
    "gemini-1.5-flash": ModelConfig(
        name="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        max_tokens=8192,
        temperature_range=(0.0, 2.0),
        recommended_temperature=0.3,
        rate_limit_rpm=300,
        context_window=1000000,
        multimodal=True,
        description="Faster model optimized for speed and efficiency"
    ),
    
    "gemini-1.0-pro": ModelConfig(
        name="gemini-1.0-pro",
        display_name="Gemini 1.0 Pro",
        max_tokens=2048,
        temperature_range=(0.0, 1.0),
        recommended_temperature=0.3,
        rate_limit_rpm=60,
        context_window=30720,
        description="Stable model for general text tasks"
    ),
    
    # Gemma models (text-only)
    "gemma-2-27b-it": ModelConfig(
        name="gemma-2-27b-it",
        display_name="Gemma 2 27B Instruct",
        max_tokens=8192,
        temperature_range=(0.0, 2.0),
        recommended_temperature=0.3,
        rate_limit_rpm=60,
        context_window=8192,
        description="Large Gemma model optimized for instruction following"
    ),
    
    "gemma-2-9b-it": ModelConfig(
        name="gemma-2-9b-it",
        display_name="Gemma 2 9B Instruct",
        max_tokens=8192,
        temperature_range=(0.0, 2.0),
        recommended_temperature=0.3,
        rate_limit_rpm=60,
        context_window=8192,
        description="Medium-sized Gemma model for balanced performance"
    ),
    
    "gemma-2-2b-it": ModelConfig(
        name="gemma-2-2b-it",
        display_name="Gemma 2 2B Instruct",
        max_tokens=8192,
        temperature_range=(0.0, 2.0),
        recommended_temperature=0.3,
        rate_limit_rpm=60,
        context_window=8192,
        description="Compact Gemma model for resource-constrained environments"
    ),
    
    # Newer Gemma 3 models (multimodal)
    "gemma-3-27b-it": ModelConfig(
        name="gemma-3-27b-it",
        display_name="Gemma 3 27B Instruct",
        max_tokens=8192,
        temperature_range=(0.0, 2.0),
        recommended_temperature=0.3,
        rate_limit_rpm=60,
        context_window=128000,  # 128K context
        multimodal=True,
        description="Latest Gemma 3 model with multimodal capabilities"
    ),
    
    # Experimental models (if available)
    "gemma-3n-e4b-it": ModelConfig(
        name="gemma-3n-e4b-it",
        display_name="Gemma 3N E4B Instruct",
        max_tokens=8192,
        temperature_range=(0.0, 2.0),
        recommended_temperature=0.3,
        rate_limit_rpm=60,
        context_window=128000,
        multimodal=True,
        description="Experimental Gemma 3N model with enhanced capabilities"
    ),
    
    "gemma-3n-e2b-it": ModelConfig(
        name="gemma-3n-e2b-it", 
        display_name="Gemma 3N E2B Instruct",
        max_tokens=8192,
        temperature_range=(0.0, 2.0),
        recommended_temperature=0.3,
        rate_limit_rpm=60,
        context_window=128000,
        multimodal=True,
        description="Experimental Gemma 3N model - smaller variant"
    ),
}

class ModelManager:
    """Manages model configuration and selection"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.current_model = None
        
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get all available models"""
        return AVAILABLE_MODELS
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
        return AVAILABLE_MODELS[model_name]
    
    def set_model(self, model_name: str) -> ModelConfig:
        """Set the current model and return its configuration"""
        config = self.get_model_config(model_name)
        self.current_model = model_name
        return config
    
    def get_recommended_models_for_trading(self) -> Dict[str, str]:
        """Get recommended models for different trading use cases"""
        return {
            "high_accuracy": "gemma-3-27b-it",  # Best for complex analysis
            "balanced": "gemma-2-27b-it",       # Good balance of speed/accuracy
            "fast": "gemini-1.5-flash",         # Fastest responses
            "experimental": "gemma-3n-e4b-it",  # Latest experimental features
            "resource_efficient": "gemma-2-9b-it"  # Lower resource usage
        }
    
    def validate_model_availability(self, model_name: str) -> bool:
        """Check if a model is available (basic validation)"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Try to list available models to verify access
            models = genai.list_models()
            available_model_names = [m.name.split('/')[-1] for m in models]
            
            return model_name in available_model_names
        except Exception as e:
            print(f"Could not validate model availability: {e}")
            return model_name in AVAILABLE_MODELS  # Fallback to our known list

def get_model_from_env() -> str:
    """Get model name from environment variable with fallback"""
    model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
    
    if model_name not in AVAILABLE_MODELS:
        print(f"Warning: Model {model_name} not in known models list. Using gemini-1.5-pro as fallback.")
        model_name = 'gemini-1.5-pro'
    
    return model_name

def print_available_models():
    """Print all available models with descriptions"""
    print("Available Gemini Models:")
    print("=" * 50)
    
    for model_name, config in AVAILABLE_MODELS.items():
        multimodal_str = "📸 Multimodal" if config.multimodal else "📝 Text-only"
        print(f"🤖 {config.display_name}")
        print(f"   Model ID: {model_name}")
        print(f"   Type: {multimodal_str}")
        print(f"   Context: {config.context_window:,} tokens")
        print(f"   Rate Limit: {config.rate_limit_rpm} RPM")
        print(f"   Description: {config.description}")
        print()

if __name__ == "__main__":
    # Demo usage
    print_available_models()
    
    print("\nRecommended models for trading:")
    print("=" * 30)
    
    manager = ModelManager("dummy_key")
    recommendations = manager.get_recommended_models_for_trading()
    
    for use_case, model_name in recommendations.items():
        config = manager.get_model_config(model_name)
        print(f"• {use_case.title()}: {config.display_name}")
        print(f"  └─ {config.description}")