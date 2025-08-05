#pragma once

#include "broker_interface.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>

namespace forex_bot {

/**
 * @brief MetaTrader 5 broker implementation
 * 
 * This class implements the BrokerInterface for MetaTrader 5 platform.
 * It uses both the MetaTrader 5 WebAPI and the MetaTrader5 Python library
 * for comprehensive MT5 integration.
 */
class MT5Broker : public BrokerInterface {
public:
    explicit MT5Broker(const BrokerConfig& config);
    ~MT5Broker() override;

    // Connection management
    bool Connect() override;
    bool Disconnect() override;
    bool IsConnected() const override;
    ConnectionStatus GetConnectionStatus() const override;

    // Market data
    bool SubscribeToMarketData(const std::vector<std::string>& symbols) override;
    bool UnsubscribeFromMarketData(const std::vector<std::string>& symbols) override;
    std::vector<MarketData> GetLatestMarketData(const std::string& symbol) const override;
    std::vector<OHLCV> GetHistoricalData(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    ) const override;

    // Order management
    std::future<OrderResult> PlaceOrder(const OrderRequest& request) override;
    std::future<OrderResult> ModifyOrder(const std::string& order_id, const OrderModification& modification) override;
    std::future<OrderResult> CancelOrder(const std::string& order_id) override;
    std::future<OrderResult> ClosePosition(const std::string& position_id) override;

    // Account information
    AccountInfo GetAccountInfo() const override;
    std::vector<Position> GetOpenPositions() const override;
    std::vector<Order> GetPendingOrders() const override;
    std::vector<Trade> GetTradeHistory(
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    ) const override;

    // Symbol information
    std::vector<SymbolInfo> GetAvailableSymbols() const override;
    SymbolInfo GetSymbolInfo(const std::string& symbol) const override;

    // Event callbacks
    void SetMarketDataCallback(std::function<void(const MarketData&)> callback) override;
    void SetOrderUpdateCallback(std::function<void(const OrderUpdate&)> callback) override;
    void SetAccountUpdateCallback(std::function<void(const AccountUpdate&)> callback) override;

    // Configuration
    void SetConfig(const BrokerConfig& config) override;
    BrokerConfig GetConfig() const override;

private:
    // WebSocket client for real-time data
    using WebSocketClient = websocketpp::client<websocketpp::config::asio_tls_client>;
    using WebSocketMessage = websocketpp::config::asio_tls_client::message_type::ptr;

    // Connection state
    mutable std::mutex connection_mutex_;
    std::atomic<bool> connected_{false};
    std::atomic<bool> should_stop_{false};
    
    // WebSocket connection
    std::unique_ptr<WebSocketClient> ws_client_;
    std::thread ws_thread_;
    websocketpp::connection_hdl ws_connection_;
    
    // Market data streaming
    std::thread market_data_thread_;
    mutable std::mutex market_data_mutex_;
    std::unordered_map<std::string, MarketData> latest_market_data_;
    std::vector<std::string> subscribed_symbols_;
    
    // Python bridge for MT5 operations
    std::thread python_bridge_thread_;
    std::queue<std::string> python_command_queue_;
    std::mutex python_queue_mutex_;
    std::condition_variable python_queue_cv_;
    
    // Internal methods
    bool InitializeWebSocket();
    bool InitializePythonBridge();
    void StartMarketDataStream();
    void StopMarketDataStream();
    void ProcessWebSocketMessage(const WebSocketMessage& msg);
    void ProcessPythonCommands();
    
    // MT5 specific methods
    bool LoginToMT5() const;
    std::string ExecutePythonCommand(const std::string& command) const;
    MarketData ParseMarketDataFromMT5(const std::string& data) const;
    OrderResult ParseOrderResultFromMT5(const std::string& data) const;
    
    // Helper methods
    std::string ConvertSymbolToMT5Format(const std::string& symbol) const;
    std::string ConvertSymbolFromMT5Format(const std::string& symbol) const;
    int ConvertTimeFrameToMT5(TimeFrame timeframe) const;
    TimeFrame ConvertTimeFrameFromMT5(int mt5_timeframe) const;
    
    // Logging
    void LogInfo(const std::string& message) const;
    void LogError(const std::string& message) const;
    void LogDebug(const std::string& message) const;
};

} // namespace forex_bot