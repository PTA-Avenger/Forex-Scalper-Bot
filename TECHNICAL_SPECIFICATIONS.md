# Forex Scalping Bot - Technical Specifications

## Project Structure

```
forex-scalping-bot/
├── cpp/                           # C++ Core Engine
│   ├── src/
│   │   ├── core/                  # Core trading engine
│   │   │   ├── engine.cpp         # Main trading engine
│   │   │   ├── market_data.cpp    # Market data handler
│   │   │   ├── order_manager.cpp  # Order execution
│   │   │   └── event_dispatcher.cpp # Event system
│   │   ├── brokers/               # Broker API implementations
│   │   │   ├── broker_interface.cpp # Abstract broker interface
│   │   │   ├── oanda_broker.cpp   # OANDA implementation
│   │   │   └── mt_broker.cpp      # MetaTrader implementation
│   │   ├── strategies/            # Trading strategies
│   │   │   ├── strategy_base.cpp  # Base strategy class
│   │   │   ├── ema_crossover.cpp  # EMA crossover strategy
│   │   │   ├── mean_reversion.cpp # Mean reversion strategy
│   │   │   └── atr_breakout.cpp   # ATR breakout strategy
│   │   ├── risk/                  # Risk management
│   │   │   ├── risk_manager.cpp   # Main risk manager
│   │   │   ├── position_sizer.cpp # Kelly criterion sizing
│   │   │   └── drawdown_monitor.cpp # Drawdown protection
│   │   ├── backtest/              # Backtesting engine
│   │   │   ├── backtester.cpp     # Main backtesting engine
│   │   │   ├── performance_metrics.cpp # Performance calculations
│   │   │   └── trade_analyzer.cpp # Trade analysis
│   │   └── utils/                 # Utilities and helpers
│   │       ├── config_manager.cpp # Configuration management
│   │       ├── logger.cpp         # Logging system
│   │       ├── json_parser.cpp    # JSON utilities
│   │       └── time_utils.cpp     # Time handling
│   ├── include/                   # Header files
│   │   ├── core/
│   │   ├── brokers/
│   │   ├── strategies/
│   │   ├── risk/
│   │   ├── backtest/
│   │   └── utils/
│   ├── tests/                     # Unit tests
│   │   ├── test_strategies.cpp
│   │   ├── test_risk_manager.cpp
│   │   └── test_backtester.cpp
│   ├── CMakeLists.txt            # CMake build configuration
│   └── main.cpp                  # Application entry point
├── python/                        # AI Microservices
│   ├── price_predictor/           # LSTM/XGBoost services
│   │   ├── app.py                # Flask API server
│   │   ├── models/
│   │   │   ├── lstm_model.py     # LSTM implementation
│   │   │   └── xgboost_model.py  # XGBoost implementation
│   │   ├── data/
│   │   │   ├── preprocessor.py   # Data preprocessing
│   │   │   └── feature_engineer.py # Feature engineering
│   │   └── requirements.txt
│   ├── sentiment_analyzer/        # News/social sentiment
│   │   ├── app.py                # Flask API server
│   │   ├── analyzers/
│   │   │   ├── news_analyzer.py  # News sentiment
│   │   │   ├── social_analyzer.py # Social media sentiment
│   │   │   └── bert_analyzer.py  # BERT-based analysis
│   │   ├── data_sources/
│   │   │   ├── twitter_client.py # Twitter/X API client
│   │   │   └── news_client.py    # News API client
│   │   └── requirements.txt
│   ├── signal_processor/          # Signal aggregation
│   │   ├── app.py                # Flask API server
│   │   ├── aggregator.py         # Signal aggregation logic
│   │   └── requirements.txt
│   └── common/                    # Shared utilities
│       ├── database.py           # Database connections
│       ├── redis_client.py       # Redis client
│       └── utils.py              # Common utilities
├── frontend/                      # React Dashboard
│   ├── src/
│   │   ├── components/           # React components
│   │   │   ├── Dashboard.jsx     # Main dashboard
│   │   │   ├── TradingChart.jsx  # TradingView integration
│   │   │   ├── PositionTable.jsx # Position monitoring
│   │   │   ├── RiskMetrics.jsx   # Risk display
│   │   │   └── StrategyConfig.jsx # Strategy configuration
│   │   ├── services/             # API services
│   │   │   ├── websocket.js      # WebSocket client
│   │   │   ├── api.js            # REST API client
│   │   │   └── tradingview.js    # TradingView integration
│   │   ├── utils/                # Utilities
│   │   │   ├── formatters.js     # Data formatters
│   │   │   └── constants.js      # Constants
│   │   ├── App.jsx               # Main app component
│   │   └── index.js              # Entry point
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── package.json              # NPM dependencies
│   └── webpack.config.js         # Webpack configuration
├── config/                        # Configuration files
│   ├── trading_config.json       # Trading parameters
│   ├── broker_config.json        # Broker settings
│   ├── risk_config.json          # Risk management settings
│   ├── ai_config.json            # AI service settings
│   └── database_config.json      # Database settings
├── docker/                        # Docker configurations
│   ├── cpp/
│   │   └── Dockerfile            # C++ engine container
│   ├── python/
│   │   └── Dockerfile            # Python services container
│   ├── frontend/
│   │   └── Dockerfile            # React app container
│   └── nginx/
│       ├── Dockerfile            # NGINX container
│       └── nginx.conf            # NGINX configuration
├── scripts/                       # Deployment scripts
│   ├── build.sh                  # Build script
│   ├── deploy.sh                 # Deployment script
│   ├── backup.sh                 # Backup script
│   └── monitor.sh                # Monitoring script
├── docs/                          # Documentation
│   ├── API.md                    # API documentation
│   ├── DEPLOYMENT.md             # Deployment guide
│   ├── CONFIGURATION.md          # Configuration guide
│   └── TROUBLESHOOTING.md        # Troubleshooting guide
├── docker-compose.yml            # Docker Compose configuration
├── docker-compose.prod.yml       # Production Docker Compose
└── README.md                     # Project README
```

## Core C++ Engine Specifications

### 1. CMakeLists.txt Configuration

```cmake
cmake_minimum_required(VERSION 3.16)
project(ForexScalpingBot VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

# Find required packages
find_package(Boost REQUIRED COMPONENTS system thread chrono)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(spdlog REQUIRED)
find_package(Redis++ REQUIRED)
find_package(PostgreSQL REQUIRED)

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Boost::system
    Boost::thread
    Boost::chrono
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    redis++
    pq
)
```

### 2. Core Engine Classes

#### TradingEngine Class
```cpp
class TradingEngine {
private:
    std::unique_ptr<BrokerInterface> broker_;
    std::unique_ptr<MarketDataHandler> market_data_;
    std::unique_ptr<StrategyManager> strategy_manager_;
    std::unique_ptr<RiskManager> risk_manager_;
    std::unique_ptr<OrderManager> order_manager_;
    std::unique_ptr<EventDispatcher> event_dispatcher_;
    
public:
    TradingEngine(const Config& config);
    ~TradingEngine();
    
    bool Initialize();
    void Start();
    void Stop();
    void ProcessMarketData(const TickData& tick);
    void ProcessSignal(const TradingSignal& signal);
    void ProcessOrder(const Order& order);
};
```

#### MarketDataHandler Class
```cpp
class MarketDataHandler {
private:
    std::unordered_map<std::string, TickBuffer> tick_buffers_;
    std::unordered_map<std::string, OHLCVBuffer> ohlcv_buffers_;
    std::unique_ptr<RedisClient> redis_client_;
    
public:
    MarketDataHandler(const Config& config);
    
    void ProcessTick(const TickData& tick);
    void UpdateOHLCV(const std::string& symbol, const TickData& tick);
    std::vector<OHLCV> GetHistoricalData(const std::string& symbol, 
                                        const TimeRange& range);
    void SubscribeToSymbol(const std::string& symbol);
    void UnsubscribeFromSymbol(const std::string& symbol);
};
```

#### StrategyBase Class
```cpp
class StrategyBase {
protected:
    std::string name_;
    std::unordered_map<std::string, double> parameters_;
    std::unique_ptr<TechnicalIndicators> indicators_;
    
public:
    StrategyBase(const std::string& name, const Config& config);
    virtual ~StrategyBase() = default;
    
    virtual void Initialize() = 0;
    virtual TradingSignal ProcessData(const MarketData& data) = 0;
    virtual void UpdateParameters(const std::unordered_map<std::string, double>& params);
    virtual double CalculatePositionSize(const TradingSignal& signal) = 0;
};
```

### 3. Broker Interface Specifications

#### Abstract Broker Interface
```cpp
class BrokerInterface {
public:
    virtual ~BrokerInterface() = default;
    
    virtual bool Connect() = 0;
    virtual void Disconnect() = 0;
    virtual bool IsConnected() const = 0;
    
    virtual OrderResult PlaceOrder(const Order& order) = 0;
    virtual bool CancelOrder(const std::string& order_id) = 0;
    virtual bool ModifyOrder(const std::string& order_id, const Order& new_order) = 0;
    
    virtual std::vector<Position> GetPositions() = 0;
    virtual std::vector<Order> GetOrders() = 0;
    virtual AccountInfo GetAccountInfo() = 0;
    
    virtual void SubscribeToMarketData(const std::vector<std::string>& symbols) = 0;
    virtual void UnsubscribeFromMarketData(const std::vector<std::string>& symbols) = 0;
    
    virtual std::vector<TickData> GetHistoricalTicks(const std::string& symbol,
                                                    const TimeRange& range) = 0;
};
```

#### OANDA Broker Implementation
```cpp
class OANDABroker : public BrokerInterface {
private:
    std::string api_key_;
    std::string account_id_;
    std::string base_url_;
    std::unique_ptr<CurlClient> http_client_;
    std::unique_ptr<WebSocketClient> ws_client_;
    std::atomic<bool> connected_;
    
public:
    OANDABroker(const Config& config);
    ~OANDABroker();
    
    bool Connect() override;
    void Disconnect() override;
    bool IsConnected() const override;
    
    OrderResult PlaceOrder(const Order& order) override;
    bool CancelOrder(const std::string& order_id) override;
    bool ModifyOrder(const std::string& order_id, const Order& new_order) override;
    
    std::vector<Position> GetPositions() override;
    std::vector<Order> GetOrders() override;
    AccountInfo GetAccountInfo() override;
    
    void SubscribeToMarketData(const std::vector<std::string>& symbols) override;
    void UnsubscribeFromMarketData(const std::vector<std::string>& symbols) override;
    
    std::vector<TickData> GetHistoricalTicks(const std::string& symbol,
                                           const TimeRange& range) override;
};
```

### 4. Trading Strategies Specifications

#### EMA Crossover Strategy
```cpp
class EMACrossoverStrategy : public StrategyBase {
private:
    double fast_ema_period_;
    double slow_ema_period_;
    double macd_fast_period_;
    double macd_slow_period_;
    double macd_signal_period_;
    
public:
    EMACrossoverStrategy(const Config& config);
    
    void Initialize() override;
    TradingSignal ProcessData(const MarketData& data) override;
    double CalculatePositionSize(const TradingSignal& signal) override;
    
private:
    bool CheckEMACrossover(const std::vector<double>& prices);
    bool CheckMACDConfirmation(const std::vector<double>& prices);
};
```

#### Mean Reversion Strategy
```cpp
class MeanReversionStrategy : public StrategyBase {
private:
    double rsi_period_;
    double rsi_oversold_;
    double rsi_overbought_;
    double bb_period_;
    double bb_std_dev_;
    
public:
    MeanReversionStrategy(const Config& config);
    
    void Initialize() override;
    TradingSignal ProcessData(const MarketData& data) override;
    double CalculatePositionSize(const TradingSignal& signal) override;
    
private:
    bool CheckRSICondition(const std::vector<double>& prices);
    bool CheckBollingerBandCondition(const std::vector<double>& prices);
};
```

#### ATR Breakout Strategy
```cpp
class ATRBreakoutStrategy : public StrategyBase {
private:
    double atr_period_;
    double breakout_multiplier_;
    int lookback_period_;
    
public:
    ATRBreakoutStrategy(const Config& config);
    
    void Initialize() override;
    TradingSignal ProcessData(const MarketData& data) override;
    double CalculatePositionSize(const TradingSignal& signal) override;
    
private:
    bool CheckBreakout(const std::vector<double>& prices);
    double CalculateATR(const std::vector<OHLCV>& bars);
    std::pair<double, double> FindPriceExtremes(const std::vector<double>& prices);
};
```

### 5. Risk Management Specifications

#### Risk Manager Class
```cpp
class RiskManager {
private:
    double max_daily_drawdown_;
    double max_position_size_;
    double max_correlation_exposure_;
    std::unordered_map<std::string, double> position_limits_;
    std::unique_ptr<DrawdownMonitor> drawdown_monitor_;
    std::unique_ptr<PositionSizer> position_sizer_;
    
public:
    RiskManager(const Config& config);
    
    bool ValidateOrder(const Order& order);
    double CalculatePositionSize(const TradingSignal& signal);
    bool CheckDrawdownLimit();
    bool CheckCorrelationLimit(const std::string& symbol);
    void UpdateRiskMetrics();
    
private:
    double CalculatePortfolioCorrelation();
    double CalculateCurrentDrawdown();
};
```

#### Kelly Criterion Position Sizer
```cpp
class KellyCriterionSizer {
private:
    double win_rate_;
    double avg_win_;
    double avg_loss_;
    double kelly_fraction_;
    
public:
    KellyCriterionSizer(const Config& config);
    
    double CalculateOptimalSize(const TradingSignal& signal);
    void UpdateStatistics(const TradeResult& result);
    double GetKellyFraction() const;
    
private:
    double CalculateKellyFraction();
    void UpdateWinLossStats(const TradeResult& result);
};
```

### 6. Backtesting Engine Specifications

#### Backtester Class
```cpp
class Backtester {
private:
    std::unique_ptr<StrategyBase> strategy_;
    std::unique_ptr<RiskManager> risk_manager_;
    std::vector<OHLCV> historical_data_;
    std::vector<Trade> trades_;
    double initial_capital_;
    double current_capital_;
    
public:
    Backtester(const Config& config);
    
    void LoadHistoricalData(const std::string& symbol, const TimeRange& range);
    void SetStrategy(std::unique_ptr<StrategyBase> strategy);
    void RunBacktest();
    PerformanceMetrics CalculateMetrics();
    void GenerateReport();
    
private:
    void ProcessBar(const OHLCV& bar);
    void ExecuteTrade(const TradingSignal& signal);
    double CalculateSlippage(const Order& order);
};
```

#### Performance Metrics Calculator
```cpp
class PerformanceMetrics {
public:
    struct Metrics {
        double total_return;
        double sharpe_ratio;
        double max_drawdown;
        double win_rate;
        double profit_factor;
        double avg_trade_duration;
        int total_trades;
        double volatility;
        double calmar_ratio;
    };
    
    static Metrics Calculate(const std::vector<Trade>& trades,
                           double initial_capital,
                           double risk_free_rate = 0.02);
    
private:
    static double CalculateSharpeRatio(const std::vector<double>& returns,
                                     double risk_free_rate);
    static double CalculateMaxDrawdown(const std::vector<double>& equity_curve);
    static double CalculateVolatility(const std::vector<double>& returns);
};
```

## Data Structures

### Core Data Types
```cpp
struct TickData {
    std::string symbol;
    double bid;
    double ask;
    double volume;
    std::chrono::system_clock::time_point timestamp;
};

struct OHLCV {
    std::string symbol;
    double open;
    double high;
    double low;
    double close;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    std::chrono::minutes timeframe;
};

struct TradingSignal {
    std::string symbol;
    SignalType type; // BUY, SELL, CLOSE
    double confidence;
    double suggested_size;
    std::string strategy_name;
    std::chrono::system_clock::time_point timestamp;
};

struct Order {
    std::string id;
    std::string symbol;
    OrderType type; // MARKET, LIMIT, STOP
    OrderSide side; // BUY, SELL
    double quantity;
    double price;
    double stop_loss;
    double take_profit;
    std::chrono::system_clock::time_point timestamp;
};

struct Position {
    std::string symbol;
    double quantity;
    double avg_price;
    double unrealized_pnl;
    double realized_pnl;
    std::chrono::system_clock::time_point open_time;
};

struct Trade {
    std::string id;
    std::string symbol;
    OrderSide side;
    double entry_price;
    double exit_price;
    double quantity;
    double pnl;
    std::chrono::system_clock::time_point entry_time;
    std::chrono::system_clock::time_point exit_time;
    std::string strategy_name;
};
```

## Configuration Management

### JSON Configuration Structure
```json
{
  "trading": {
    "symbols": ["EUR/USD", "GBP/USD", "USD/JPY"],
    "timeframes": ["1m", "5m", "15m"],
    "max_positions": 10,
    "paper_trading": true
  },
  "broker": {
    "type": "oanda",
    "api_key": "${OANDA_API_KEY}",
    "account_id": "${OANDA_ACCOUNT_ID}",
    "environment": "practice"
  },
  "risk": {
    "max_daily_drawdown": 0.05,
    "max_position_size": 0.02,
    "kelly_fraction": 0.25,
    "correlation_limit": 0.7
  },
  "strategies": {
    "ema_crossover": {
      "enabled": true,
      "fast_ema": 12,
      "slow_ema": 26,
      "macd_fast": 12,
      "macd_slow": 26,
      "macd_signal": 9
    },
    "mean_reversion": {
      "enabled": true,
      "rsi_period": 14,
      "rsi_oversold": 30,
      "rsi_overbought": 70,
      "bb_period": 20,
      "bb_std_dev": 2.0
    },
    "atr_breakout": {
      "enabled": true,
      "atr_period": 14,
      "breakout_multiplier": 2.0,
      "lookback_period": 20
    }
  }
}
```

This technical specification provides the complete blueprint for implementing the C++ core engine with all necessary classes, interfaces, and data structures.