# Gemini Model Usage Guide

This guide explains how to use different Gemini models with your Forex Scalping Bot, including the new **Gemma 3N E4B** model you asked about.

## 🤖 Available Models

### **Standard Gemini Models**
- **`gemini-1.5-pro`** - Most capable for complex analysis (1M context, multimodal)
- **`gemini-1.5-flash`** - Fastest responses (1M context, multimodal, 300 RPM)
- **`gemini-1.0-pro`** - Stable baseline (30K context, text-only)

### **Gemma Models (Open Source)**
- **`gemma-2-27b-it`** - Large instruction-tuned model (8K context)
- **`gemma-2-9b-it`** - Balanced performance (8K context)
- **`gemma-2-2b-it`** - Lightweight option (8K context)

### **Latest Gemma 3 Models**
- **`gemma-3-27b-it`** - Latest with multimodal support (128K context)
- **`gemma-3n-e4b-it`** ⭐ - **Experimental enhanced model** (128K context)
- **`gemma-3n-e2b-it`** - Smaller experimental variant (128K context)

## 🚀 How to Use Gemma-3N-E4B-IT

Yes, you can absolutely use `gemma-3n-e4b-it`! Here's how:

### **Method 1: Environment Variable (Recommended)**
```bash
# Set in your .env file
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemma-3n-e4b-it

# Restart your services
docker-compose down
docker-compose up -d
```

### **Method 2: API Call**
```bash
# Switch model via API
curl -X POST http://localhost:5001/models/switch \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gemma-3n-e4b-it",
    "service": "prediction"
  }'
```

### **Method 3: Test First**
```bash
# Test the model before switching
cd python
python model_tester.py --model gemma-3n-e4b-it

# Benchmark against other models
python model_tester.py --benchmark --models gemini-1.5-pro gemma-3n-e4b-it
```

## 📊 Model Comparison

| Model | Context | Speed | Accuracy | Multimodal | Best For |
|-------|---------|-------|----------|------------|----------|
| `gemini-1.5-pro` | 1M | Medium | High | ✅ | Complex analysis |
| `gemini-1.5-flash` | 1M | **Fastest** | Good | ✅ | Real-time trading |
| `gemma-2-27b-it` | 8K | Medium | High | ❌ | Balanced trading |
| `gemma-3-27b-it` | 128K | Medium | High | ✅ | Latest features |
| `gemma-3n-e4b-it` | 128K | Medium | **Experimental** | ✅ | **Cutting-edge** |

## 🎯 Recommendations by Use Case

### **For Maximum Performance**
```bash
GEMINI_MODEL=gemma-3n-e4b-it  # Latest experimental features
```

### **For Speed**
```bash
GEMINI_MODEL=gemini-1.5-flash  # 300 RPM rate limit
```

### **For Stability**
```bash
GEMINI_MODEL=gemini-1.5-pro    # Most tested and reliable
```

### **For Resource Efficiency**
```bash
GEMINI_MODEL=gemma-2-9b-it     # Lower resource usage
```

## 🔧 Configuration Options

### **Complete .env Setup**
```bash
# Core Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemma-3n-e4b-it

# Database Configuration
REDIS_HOST=redis
POSTGRES_HOST=postgres
INFLUXDB_HOST=influxdb

# Trading Configuration
RISK_PERCENTAGE=2.0
MAX_POSITIONS=5
```

### **API Endpoints for Model Management**

#### **List Available Models**
```bash
curl http://localhost:5001/models/available
```

#### **Get Current Model Info**
```bash
curl http://localhost:5001/model-info
```

#### **Switch Models**
```bash
# Switch prediction model
curl -X POST http://localhost:5001/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma-3n-e4b-it", "service": "prediction"}'

# Switch sentiment model
curl -X POST http://localhost:5001/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma-3n-e4b-it", "service": "sentiment"}'
```

#### **Test Model Performance**
```bash
curl -X POST http://localhost:5001/models/test \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma-3n-e4b-it"}'
```

## 🧪 Testing Your Model Choice

### **CLI Testing Tool**
```bash
cd python

# List all models
python model_tester.py --list

# Test specific model
python model_tester.py --model gemma-3n-e4b-it

# Compare multiple models
python model_tester.py --benchmark --models \
  gemini-1.5-pro gemma-3n-e4b-it gemma-2-27b-it

# Test only prediction or sentiment
python model_tester.py --model gemma-3n-e4b-it --type prediction
```

### **Integration Testing**
```bash
# Test full integration
python test_gemini_integration.py

# Check service health
curl http://localhost:5001/health
```

## ⚡ Performance Tips

### **Rate Limiting**
- **Gemini 1.5 Flash**: 300 requests/minute (fastest)
- **Other models**: 60 requests/minute
- Built-in caching reduces API calls

### **Context Usage**
- **Gemma 3N models**: 128K context (great for complex analysis)
- **Gemini models**: 1M context (best for comprehensive data)
- **Older Gemma**: 8K context (sufficient for most trading)

### **Caching Strategy**
```bash
# Check cache status
curl http://localhost:5001/cache-stats

# Predictions cached for 5 minutes
# Sentiment cached for 10 minutes
```

## 🚨 Important Notes

### **Experimental Models**
- `gemma-3n-e4b-it` is **experimental** - may have different behavior
- Monitor performance and fallback to stable models if needed
- Report any issues for future improvements

### **API Key Requirements**
- Get your key from: https://makersuite.google.com/app/apikey
- Free tier available with generous limits
- Paid tier for higher rate limits

### **Model Availability**
- Some models may not be available in all regions
- The system will fallback to `gemini-1.5-pro` if specified model fails
- Check logs for any model initialization issues

## 🔄 Switching Models in Production

### **Zero-Downtime Switch**
```bash
# Switch via API (no restart needed)
curl -X POST http://localhost:5001/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name": "gemma-3n-e4b-it", "service": "prediction"}'
```

### **Persistent Switch**
```bash
# Update .env file
echo "GEMINI_MODEL=gemma-3n-e4b-it" >> .env

# Restart services
docker-compose restart price-predictor
```

## 📈 Model Performance Monitoring

### **Built-in Metrics**
- Response times tracked automatically
- Success/failure rates logged
- Cache hit ratios monitored

### **Custom Monitoring**
```bash
# Monitor logs
docker logs forex-scalping-bot-price-predictor-1 -f

# Check performance
curl http://localhost:5001/cache-stats
```

## 🎉 Quick Start with Gemma-3N-E4B-IT

```bash
# 1. Set your API key and model
echo "GEMINI_API_KEY=your_key_here" > .env
echo "GEMINI_MODEL=gemma-3n-e4b-it" >> .env

# 2. Start the system
docker-compose up -d

# 3. Test the model
cd python && python model_tester.py --model gemma-3n-e4b-it

# 4. Check it's working
curl http://localhost:5001/health

# 5. Make a prediction
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EUR/USD",
    "timeframe": "1h",
    "ohlcv_data": [...]
  }'
```

## 🆘 Troubleshooting

### **Model Not Found**
```bash
# List available models
python model_tester.py --list

# Check if model is supported
curl http://localhost:5001/models/available
```

### **API Errors**
```bash
# Check API key
echo $GEMINI_API_KEY

# Test API connectivity
python model_tester.py --model gemini-1.5-pro
```

### **Performance Issues**
```bash
# Check rate limits
curl http://localhost:5001/model-info

# Monitor response times
curl http://localhost:5001/cache-stats
```

---

**Ready to use Gemma-3N-E4B-IT?** Just set `GEMINI_MODEL=gemma-3n-e4b-it` in your `.env` file and restart! 🚀