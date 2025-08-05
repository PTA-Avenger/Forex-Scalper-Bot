#pragma once

#include "broker_interface.hpp"
#include "../common/types.hpp"
#include "../common/http_client.hpp"
#include "../common/json_utils.hpp"
#include <string>
#include <memory>
#include <future>
#include <chrono>

namespace forex_bot {

/**
 * FXCM Broker Interface
 * 
 * Provides integration with FXCM through the Python FXCM service
 * Uses HTTP REST API calls to communicate with the FXCM service
 */
class FXCMBroker : public BrokerInterface {
public:
    /**
     * Constructor
     * @param service_url URL of the FXCM service (e.g., "http://fxcm-service:5003")
     * @param timeout_ms HTTP request timeout in milliseconds
     */
    explicit FXCMBroker(const std::string& service_url, int timeout_ms = 30000);
    
    /**
     * Destructor
     */
    virtual ~FXCMBroker() = default;

    // BrokerInterface implementation
    bool initialize() override;
    void shutdown() override;
    bool is_connected() const override;
    
    OrderResult place_order(const OrderRequest& request) override;
    OrderResult close_position(const std::string& position_id) override;
    OrderResult modify_order(const std::string& order_id, const OrderModification& modification) override;
    OrderResult cancel_order(const std::string& order_id) override;
    
    std::vector<Position> get_open_positions() override;
    std::vector<Order> get_open_orders() override;
    AccountInfo get_account_info() override;
    
    MarketData get_market_data(const std::string& symbol) override;
    std::vector<OHLCV> get_historical_data(const std::string& symbol, 
                                          const std::string& timeframe, 
                                          int periods) override;
    
    // FXCM-specific methods
    
    /**
     * Get connection status from FXCM service
     */
    ConnectionStatus get_connection_status();
    
    /**
     * Force reconnection to FXCM
     */
    bool force_reconnect();
    
    /**
     * Get available trading symbols
     */
    std::vector<SymbolInfo> get_available_symbols();
    
    /**
     * Get cache statistics
     */
    CacheStats get_cache_stats();
    
    /**
     * Test webhook connectivity
     */
    bool test_webhook(const nlohmann::json& test_data);

private:
    std::string service_url_;
    int timeout_ms_;
    std::unique_ptr<HttpClient> http_client_;
    mutable std::mutex connection_mutex_;
    bool is_connected_;
    std::chrono::steady_clock::time_point last_health_check_;
    
    // Health check interval (30 seconds)
    static constexpr std::chrono::seconds HEALTH_CHECK_INTERVAL{30};
    
    /**
     * Perform health check
     */
    bool perform_health_check();
    
    /**
     * Make HTTP request to FXCM service
     */
    HttpResponse make_request(const std::string& endpoint, 
                             const std::string& method = "GET",
                             const nlohmann::json& payload = {});
    
    /**
     * Parse order result from HTTP response
     */
    OrderResult parse_order_result(const HttpResponse& response);
    
    /**
     * Parse positions from HTTP response
     */
    std::vector<Position> parse_positions(const HttpResponse& response);
    
    /**
     * Parse orders from HTTP response
     */
    std::vector<Order> parse_orders(const HttpResponse& response);
    
    /**
     * Parse account info from HTTP response
     */
    AccountInfo parse_account_info(const HttpResponse& response);
    
    /**
     * Parse market data from HTTP response
     */
    MarketData parse_market_data(const HttpResponse& response);
    
    /**
     * Parse historical data from HTTP response
     */
    std::vector<OHLCV> parse_historical_data(const HttpResponse& response);
    
    /**
     * Convert OrderRequest to JSON
     */
    nlohmann::json order_request_to_json(const OrderRequest& request);
    
    /**
     * Convert OrderModification to JSON
     */
    nlohmann::json order_modification_to_json(const OrderModification& modification);
    
    /**
     * Handle HTTP errors
     */
    void handle_http_error(const HttpResponse& response, const std::string& operation);
    
    /**
     * Log API call details
     */
    void log_api_call(const std::string& endpoint, const std::string& method, 
                     const nlohmann::json& payload, const HttpResponse& response);
};

/**
 * FXCM-specific data structures
 */

struct ConnectionStatus {
    bool fxcm_connected;
    bool fxcm_streaming;
    std::string server_type;
    int reconnect_attempts;
    int heartbeat_interval;
    int max_requests_per_minute;
    int max_orders_per_minute;
    std::chrono::system_clock::time_point timestamp;
};

struct SymbolInfo {
    std::string symbol;
    double bid;
    double ask;
    double spread;
    std::chrono::system_clock::time_point timestamp;
};

struct CacheStats {
    struct CacheInfo {
        size_t size;
        size_t maxsize;
        int ttl;
        size_t hits;
        size_t misses;
    };
    
    CacheInfo price_cache;
    CacheInfo account_cache;
    bool fxcm_connected;
    bool fxcm_streaming;
    std::chrono::system_clock::time_point timestamp;
};

} // namespace forex_bot