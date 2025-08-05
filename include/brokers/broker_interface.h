#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <future>
#include "core/types.h"

namespace forex_bot {

/**
 * @brief Abstract base class for all broker implementations
 * 
 * This interface defines the contract that all broker implementations
 * (OANDA, MT4, MT5) must follow. It provides methods for market data,
 * order management, and account information.
 */
class BrokerInterface {
public:
    virtual ~BrokerInterface() = default;

    // Connection management
    virtual bool Connect() = 0;
    virtual bool Disconnect() = 0;
    virtual bool IsConnected() const = 0;
    virtual ConnectionStatus GetConnectionStatus() const = 0;

    // Market data
    virtual bool SubscribeToMarketData(const std::vector<std::string>& symbols) = 0;
    virtual bool UnsubscribeFromMarketData(const std::vector<std::string>& symbols) = 0;
    virtual std::vector<MarketData> GetLatestMarketData(const std::string& symbol) const = 0;
    virtual std::vector<OHLCV> GetHistoricalData(
        const std::string& symbol,
        TimeFrame timeframe,
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    ) const = 0;

    // Order management
    virtual std::future<OrderResult> PlaceOrder(const OrderRequest& request) = 0;
    virtual std::future<OrderResult> ModifyOrder(const std::string& order_id, const OrderModification& modification) = 0;
    virtual std::future<OrderResult> CancelOrder(const std::string& order_id) = 0;
    virtual std::future<OrderResult> ClosePosition(const std::string& position_id) = 0;

    // Account information
    virtual AccountInfo GetAccountInfo() const = 0;
    virtual std::vector<Position> GetOpenPositions() const = 0;
    virtual std::vector<Order> GetPendingOrders() const = 0;
    virtual std::vector<Trade> GetTradeHistory(
        const std::chrono::system_clock::time_point& start,
        const std::chrono::system_clock::time_point& end
    ) const = 0;

    // Symbol information
    virtual std::vector<SymbolInfo> GetAvailableSymbols() const = 0;
    virtual SymbolInfo GetSymbolInfo(const std::string& symbol) const = 0;

    // Event callbacks
    virtual void SetMarketDataCallback(std::function<void(const MarketData&)> callback) = 0;
    virtual void SetOrderUpdateCallback(std::function<void(const OrderUpdate&)> callback) = 0;
    virtual void SetAccountUpdateCallback(std::function<void(const AccountUpdate&)> callback) = 0;

    // Configuration
    virtual void SetConfig(const BrokerConfig& config) = 0;
    virtual BrokerConfig GetConfig() const = 0;

protected:
    BrokerConfig config_;
    std::function<void(const MarketData&)> market_data_callback_;
    std::function<void(const OrderUpdate&)> order_update_callback_;
    std::function<void(const AccountUpdate&)> account_update_callback_;
};

/**
 * @brief Factory function to create broker instances
 * @param config Broker configuration
 * @return Unique pointer to broker interface
 */
std::unique_ptr<BrokerInterface> CreateBroker(const BrokerConfig& config);

} // namespace forex_bot