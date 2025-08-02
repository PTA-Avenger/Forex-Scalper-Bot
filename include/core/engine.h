#pragma once

#include "core/types.h"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>

namespace forex_bot {

// Forward declarations
class BrokerInterface;
class MarketDataHandler;
class StrategyManager;
class RiskManager;
class OrderManager;
class EventDispatcher;
class ConfigManager;
class Logger;

/**
 * @brief Main trading engine that orchestrates all components
 * 
 * The TradingEngine is the central component that coordinates:
 * - Market data processing
 * - Strategy execution
 * - Risk management
 * - Order execution
 * - Event handling
 */
class TradingEngine {
public:
    /**
     * @brief Constructor
     * @param config_path Path to configuration file
     */
    explicit TradingEngine(const std::string& config_path);
    
    /**
     * @brief Destructor
     */
    ~TradingEngine();
    
    /**
     * @brief Initialize the trading engine
     * @return true if initialization successful, false otherwise
     */
    bool Initialize();
    
    /**
     * @brief Start the trading engine
     * @return true if started successfully, false otherwise
     */
    bool Start();
    
    /**
     * @brief Stop the trading engine
     */
    void Stop();
    
    /**
     * @brief Check if the engine is running
     * @return true if running, false otherwise
     */
    bool IsRunning() const { return running_.load(); }
    
    /**
     * @brief Get current account information
     * @return AccountInfo structure
     */
    AccountInfo GetAccountInfo() const;
    
    /**
     * @brief Get all open positions
     * @return Vector of Position structures
     */
    std::vector<Position> GetPositions() const;
    
    /**
     * @brief Get all orders
     * @return Vector of Order structures
     */
    std::vector<Order> GetOrders() const;
    
    /**
     * @brief Get recent trades
     * @param limit Maximum number of trades to return
     * @return Vector of Trade structures
     */
    std::vector<Trade> GetRecentTrades(int limit = 100) const;
    
    /**
     * @brief Get current risk metrics
     * @return RiskMetrics structure
     */
    RiskMetrics GetRiskMetrics() const;
    
    /**
     * @brief Get performance metrics
     * @return PerformanceMetrics structure
     */
    PerformanceMetrics GetPerformanceMetrics() const;
    
    /**
     * @brief Place a manual order
     * @param order Order to place
     * @return OrderResult with execution details
     */
    OrderResult PlaceOrder(const Order& order);
    
    /**
     * @brief Cancel an existing order
     * @param order_id ID of the order to cancel
     * @return true if cancellation successful, false otherwise
     */
    bool CancelOrder(const std::string& order_id);
    
    /**
     * @brief Close a position
     * @param position_id ID of the position to close
     * @return true if closure successful, false otherwise
     */
    bool ClosePosition(const std::string& position_id);
    
    /**
     * @brief Close all positions
     * @return Number of positions closed
     */
    int CloseAllPositions();
    
    /**
     * @brief Enable or disable a strategy
     * @param strategy_name Name of the strategy
     * @param enabled Whether to enable or disable
     */
    void SetStrategyEnabled(const std::string& strategy_name, bool enabled);
    
    /**
     * @brief Update strategy parameters
     * @param strategy_name Name of the strategy
     * @param parameters New parameters
     */
    void UpdateStrategyParameters(const std::string& strategy_name, 
                                const std::unordered_map<std::string, double>& parameters);
    
    /**
     * @brief Get current market data for a symbol
     * @param symbol Symbol to get data for
     * @return MarketData structure
     */
    MarketData GetMarketData(const std::string& symbol) const;
    
    /**
     * @brief Subscribe to market data for additional symbols
     * @param symbols Vector of symbols to subscribe to
     */
    void SubscribeToSymbols(const std::vector<std::string>& symbols);
    
    /**
     * @brief Unsubscribe from market data for symbols
     * @param symbols Vector of symbols to unsubscribe from
     */
    void UnsubscribeFromSymbols(const std::vector<std::string>& symbols);
    
    /**
     * @brief Register event callback
     * @param event_type Type of event to listen for
     * @param callback Callback function
     */
    template<typename EventType>
    void RegisterEventCallback(std::function<void(const EventType&)> callback);
    
    /**
     * @brief Get system health status
     * @return JSON object with health information
     */
    nlohmann::json GetHealthStatus() const;
    
    /**
     * @brief Get system statistics
     * @return JSON object with statistics
     */
    nlohmann::json GetStatistics() const;

private:
    // Core components
    std::unique_ptr<ConfigManager> config_manager_;
    std::unique_ptr<Logger> logger_;
    std::unique_ptr<BrokerInterface> broker_;
    std::unique_ptr<MarketDataHandler> market_data_handler_;
    std::unique_ptr<StrategyManager> strategy_manager_;
    std::unique_ptr<RiskManager> risk_manager_;
    std::unique_ptr<OrderManager> order_manager_;
    std::unique_ptr<EventDispatcher> event_dispatcher_;
    
    // Threading
    std::atomic<bool> running_;
    std::atomic<bool> shutdown_requested_;
    std::thread main_thread_;
    std::thread market_data_thread_;
    std::thread strategy_thread_;
    std::thread risk_thread_;
    
    // Synchronization
    mutable std::mutex engine_mutex_;
    std::condition_variable shutdown_cv_;
    
    // Event queue
    std::queue<std::function<void()>> event_queue_;
    std::mutex event_queue_mutex_;
    std::condition_variable event_queue_cv_;
    
    // Configuration
    std::string config_path_;
    TradingConfig trading_config_;
    BrokerConfig broker_config_;
    RiskConfig risk_config_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    std::chrono::system_clock::time_point start_time_;
    std::atomic<uint64_t> ticks_processed_;
    std::atomic<uint64_t> orders_placed_;
    std::atomic<uint64_t> trades_executed_;
    
    /**
     * @brief Main engine loop
     */
    void MainLoop();
    
    /**
     * @brief Market data processing loop
     */
    void MarketDataLoop();
    
    /**
     * @brief Strategy processing loop
     */
    void StrategyLoop();
    
    /**
     * @brief Risk monitoring loop
     */
    void RiskLoop();
    
    /**
     * @brief Process events from the event queue
     */
    void ProcessEvents();
    
    /**
     * @brief Initialize all components
     * @return true if successful, false otherwise
     */
    bool InitializeComponents();
    
    /**
     * @brief Load configuration from file
     * @return true if successful, false otherwise
     */
    bool LoadConfiguration();
    
    /**
     * @brief Setup event handlers
     */
    void SetupEventHandlers();
    
    /**
     * @brief Handle tick data event
     * @param tick Tick data
     */
    void OnTickData(const TickData& tick);
    
    /**
     * @brief Handle trading signal event
     * @param signal Trading signal
     */
    void OnTradingSignal(const TradingSignal& signal);
    
    /**
     * @brief Handle order filled event
     * @param order Filled order
     */
    void OnOrderFilled(const Order& order);
    
    /**
     * @brief Handle position opened event
     * @param position New position
     */
    void OnPositionOpened(const Position& position);
    
    /**
     * @brief Handle position closed event
     * @param position Closed position
     */
    void OnPositionClosed(const Position& position);
    
    /**
     * @brief Handle risk alert event
     * @param alert Risk alert message
     */
    void OnRiskAlert(const std::string& alert);
    
    /**
     * @brief Update system statistics
     */
    void UpdateStatistics();
    
    /**
     * @brief Validate system state
     * @return true if system is healthy, false otherwise
     */
    bool ValidateSystemState() const;
    
    /**
     * @brief Emergency shutdown procedure
     */
    void EmergencyShutdown();
    
    // Non-copyable
    TradingEngine(const TradingEngine&) = delete;
    TradingEngine& operator=(const TradingEngine&) = delete;
};

/**
 * @brief Engine factory for creating configured instances
 */
class EngineFactory {
public:
    /**
     * @brief Create a trading engine instance
     * @param config_path Path to configuration file
     * @return Unique pointer to TradingEngine
     */
    static std::unique_ptr<TradingEngine> CreateEngine(const std::string& config_path);
    
    /**
     * @brief Create a paper trading engine instance
     * @param config_path Path to configuration file
     * @return Unique pointer to TradingEngine configured for paper trading
     */
    static std::unique_ptr<TradingEngine> CreatePaperTradingEngine(const std::string& config_path);
    
    /**
     * @brief Create a backtesting engine instance
     * @param config_path Path to configuration file
     * @param historical_data_path Path to historical data
     * @return Unique pointer to TradingEngine configured for backtesting
     */
    static std::unique_ptr<TradingEngine> CreateBacktestingEngine(
        const std::string& config_path,
        const std::string& historical_data_path);
};

} // namespace forex_bot