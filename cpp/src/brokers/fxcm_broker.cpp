#include "fxcm_broker.hpp"
#include "../common/logger.hpp"
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>

using json = nlohmann::json;

namespace forex_bot {

FXCMBroker::FXCMBroker(const std::string& service_url, int timeout_ms)
    : service_url_(service_url)
    , timeout_ms_(timeout_ms)
    , http_client_(std::make_unique<HttpClient>(timeout_ms))
    , is_connected_(false)
    , last_health_check_(std::chrono::steady_clock::now()) {
    
    Logger::info("Initializing FXCM broker with service URL: {}", service_url_);
}

bool FXCMBroker::initialize() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    Logger::info("Initializing FXCM broker connection...");
    
    try {
        // Perform initial health check
        if (perform_health_check()) {
            is_connected_ = true;
            Logger::info("FXCM broker initialized successfully");
            return true;
        } else {
            Logger::error("FXCM broker initialization failed - health check failed");
            return false;
        }
    } catch (const std::exception& e) {
        Logger::error("FXCM broker initialization error: {}", e.what());
        return false;
    }
}

void FXCMBroker::shutdown() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    Logger::info("Shutting down FXCM broker...");
    is_connected_ = false;
    
    // Clean up HTTP client
    http_client_.reset();
    
    Logger::info("FXCM broker shutdown complete");
}

bool FXCMBroker::is_connected() const {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    // Check if we need to perform a health check
    auto now = std::chrono::steady_clock::now();
    if (now - last_health_check_ > HEALTH_CHECK_INTERVAL) {
        // Cast away const for health check (this is a reasonable exception)
        const_cast<FXCMBroker*>(this)->last_health_check_ = now;
        return const_cast<FXCMBroker*>(this)->perform_health_check();
    }
    
    return is_connected_;
}

OrderResult FXCMBroker::place_order(const OrderRequest& request) {
    Logger::info("Placing {} order for {} {} at {}", 
                request.side, request.amount, request.symbol, 
                request.price.has_value() ? std::to_string(request.price.value()) : "market");
    
    try {
        json payload = order_request_to_json(request);
        HttpResponse response = make_request("/orders", "POST", payload);
        
        OrderResult result = parse_order_result(response);
        
        if (result.success) {
            Logger::info("Order placed successfully: {} (trade_id: {})", 
                        result.order_id, result.trade_id);
        } else {
            Logger::error("Order placement failed: {}", result.error_message);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        Logger::error("Order placement exception: {}", e.what());
        return OrderResult{
            .success = false,
            .error_message = e.what(),
            .timestamp = std::chrono::system_clock::now()
        };
    }
}

OrderResult FXCMBroker::close_position(const std::string& position_id) {
    Logger::info("Closing position: {}", position_id);
    
    try {
        std::string endpoint = "/positions/" + position_id + "/close";
        HttpResponse response = make_request(endpoint, "POST");
        
        OrderResult result = parse_order_result(response);
        
        if (result.success) {
            Logger::info("Position closed successfully: {}", position_id);
        } else {
            Logger::error("Position close failed: {}", result.error_message);
        }
        
        return result;
        
    } catch (const std::exception& e) {
        Logger::error("Position close exception: {}", e.what());
        return OrderResult{
            .success = false,
            .error_message = e.what(),
            .timestamp = std::chrono::system_clock::now()
        };
    }
}

OrderResult FXCMBroker::modify_order(const std::string& order_id, const OrderModification& modification) {
    // FXCM doesn't support order modification directly
    // We would need to cancel and re-place the order
    Logger::warn("Order modification not directly supported by FXCM - would require cancel and re-place");
    
    return OrderResult{
        .success = false,
        .error_message = "Order modification not supported by FXCM broker",
        .timestamp = std::chrono::system_clock::now()
    };
}

OrderResult FXCMBroker::cancel_order(const std::string& order_id) {
    // FXCM order cancellation would be handled through position closing
    Logger::info("Cancelling order/position: {}", order_id);
    
    // For FXCM, we treat this as closing a position
    return close_position(order_id);
}

std::vector<Position> FXCMBroker::get_open_positions() {
    try {
        HttpResponse response = make_request("/positions", "GET");
        return parse_positions(response);
        
    } catch (const std::exception& e) {
        Logger::error("Get positions exception: {}", e.what());
        return {};
    }
}

std::vector<Order> FXCMBroker::get_open_orders() {
    // FXCM doesn't distinguish between orders and positions in the same way
    // We'll return positions as orders for compatibility
    try {
        auto positions = get_open_positions();
        std::vector<Order> orders;
        
        for (const auto& position : positions) {
            Order order;
            order.order_id = position.position_id;
            order.symbol = position.symbol;
            order.side = position.side;
            order.amount = position.amount;
            order.order_type = "market"; // FXCM positions are market orders
            order.status = "filled";
            order.filled_amount = position.amount;
            order.average_price = position.open_price;
            order.timestamp = position.timestamp;
            
            orders.push_back(order);
        }
        
        return orders;
        
    } catch (const std::exception& e) {
        Logger::error("Get orders exception: {}", e.what());
        return {};
    }
}

AccountInfo FXCMBroker::get_account_info() {
    try {
        HttpResponse response = make_request("/account", "GET");
        return parse_account_info(response);
        
    } catch (const std::exception& e) {
        Logger::error("Get account info exception: {}", e.what());
        return AccountInfo{};
    }
}

MarketData FXCMBroker::get_market_data(const std::string& symbol) {
    try {
        std::string endpoint = "/market-data/" + symbol;
        HttpResponse response = make_request(endpoint, "GET");
        return parse_market_data(response);
        
    } catch (const std::exception& e) {
        Logger::error("Get market data exception for {}: {}", symbol, e.what());
        return MarketData{};
    }
}

std::vector<OHLCV> FXCMBroker::get_historical_data(const std::string& symbol, 
                                                   const std::string& timeframe, 
                                                   int periods) {
    try {
        std::string endpoint = "/historical-data/" + symbol + 
                              "?timeframe=" + timeframe + 
                              "&periods=" + std::to_string(periods);
        
        HttpResponse response = make_request(endpoint, "GET");
        return parse_historical_data(response);
        
    } catch (const std::exception& e) {
        Logger::error("Get historical data exception for {}: {}", symbol, e.what());
        return {};
    }
}

ConnectionStatus FXCMBroker::get_connection_status() {
    try {
        HttpResponse response = make_request("/connection/status", "GET");
        
        if (response.status_code == 200) {
            json data = json::parse(response.body);
            
            return ConnectionStatus{
                .fxcm_connected = data["connection"]["fxcm_connected"],
                .fxcm_streaming = data["connection"]["fxcm_streaming"],
                .server_type = data["connection"]["server_type"],
                .reconnect_attempts = data["connection"]["reconnect_attempts"],
                .heartbeat_interval = data["connection"]["heartbeat_interval"],
                .max_requests_per_minute = data["rate_limits"]["max_requests_per_minute"],
                .max_orders_per_minute = data["rate_limits"]["max_orders_per_minute"],
                .timestamp = std::chrono::system_clock::now()
            };
        }
        
    } catch (const std::exception& e) {
        Logger::error("Get connection status exception: {}", e.what());
    }
    
    return ConnectionStatus{};
}

bool FXCMBroker::force_reconnect() {
    Logger::info("Forcing FXCM reconnection...");
    
    try {
        HttpResponse response = make_request("/connection/reconnect", "POST");
        
        if (response.status_code == 200) {
            json data = json::parse(response.body);
            bool success = data.value("success", false);
            
            if (success) {
                Logger::info("FXCM reconnection successful");
                is_connected_ = true;
                return true;
            } else {
                Logger::error("FXCM reconnection failed: {}", 
                            data.value("error", "Unknown error"));
                return false;
            }
        }
        
        return false;
        
    } catch (const std::exception& e) {
        Logger::error("Force reconnect exception: {}", e.what());
        return false;
    }
}

std::vector<SymbolInfo> FXCMBroker::get_available_symbols() {
    try {
        HttpResponse response = make_request("/symbols", "GET");
        
        if (response.status_code == 200) {
            json data = json::parse(response.body);
            std::vector<SymbolInfo> symbols;
            
            if (data.contains("current_prices")) {
                for (const auto& price_data : data["current_prices"]) {
                    SymbolInfo symbol_info{
                        .symbol = price_data["symbol"],
                        .bid = price_data["bid"],
                        .ask = price_data["ask"],
                        .spread = price_data["spread"],
                        .timestamp = std::chrono::system_clock::now()
                    };
                    symbols.push_back(symbol_info);
                }
            }
            
            return symbols;
        }
        
    } catch (const std::exception& e) {
        Logger::error("Get available symbols exception: {}", e.what());
    }
    
    return {};
}

CacheStats FXCMBroker::get_cache_stats() {
    try {
        HttpResponse response = make_request("/cache-stats", "GET");
        
        if (response.status_code == 200) {
            json data = json::parse(response.body);
            
            CacheStats stats;
            
            // Parse price cache stats
            if (data.contains("cache_stats") && data["cache_stats"].contains("price_cache")) {
                const auto& pc = data["cache_stats"]["price_cache"];
                stats.price_cache = {
                    .size = pc.value("size", 0),
                    .maxsize = pc.value("maxsize", 0),
                    .ttl = pc.value("ttl", 0),
                    .hits = pc.value("hits", 0),
                    .misses = pc.value("misses", 0)
                };
            }
            
            // Parse account cache stats
            if (data.contains("cache_stats") && data["cache_stats"].contains("account_cache")) {
                const auto& ac = data["cache_stats"]["account_cache"];
                stats.account_cache = {
                    .size = ac.value("size", 0),
                    .maxsize = ac.value("maxsize", 0),
                    .ttl = ac.value("ttl", 0),
                    .hits = ac.value("hits", 0),
                    .misses = ac.value("misses", 0)
                };
            }
            
            // Parse connection status
            if (data.contains("connection_status")) {
                stats.fxcm_connected = data["connection_status"].value("fxcm_connected", false);
                stats.fxcm_streaming = data["connection_status"].value("fxcm_streaming", false);
            }
            
            stats.timestamp = std::chrono::system_clock::now();
            return stats;
        }
        
    } catch (const std::exception& e) {
        Logger::error("Get cache stats exception: {}", e.what());
    }
    
    return CacheStats{};
}

bool FXCMBroker::test_webhook(const nlohmann::json& test_data) {
    try {
        HttpResponse response = make_request("/webhook/test", "POST", test_data);
        return response.status_code == 200;
        
    } catch (const std::exception& e) {
        Logger::error("Test webhook exception: {}", e.what());
        return false;
    }
}

// Private methods

bool FXCMBroker::perform_health_check() {
    try {
        HttpResponse response = make_request("/health", "GET");
        
        if (response.status_code == 200) {
            json data = json::parse(response.body);
            std::string status = data.value("status", "unhealthy");
            
            bool healthy = (status == "healthy");
            if (healthy != is_connected_) {
                Logger::info("FXCM connection status changed: {}", healthy ? "connected" : "disconnected");
                is_connected_ = healthy;
            }
            
            return healthy;
        }
        
        return false;
        
    } catch (const std::exception& e) {
        Logger::error("Health check exception: {}", e.what());
        return false;
    }
}

HttpResponse FXCMBroker::make_request(const std::string& endpoint, 
                                     const std::string& method,
                                     const nlohmann::json& payload) {
    std::string url = service_url_ + endpoint;
    
    HttpRequest request;
    request.url = url;
    request.method = method;
    request.headers["Content-Type"] = "application/json";
    request.headers["Accept"] = "application/json";
    
    if (!payload.empty()) {
        request.body = payload.dump();
    }
    
    HttpResponse response = http_client_->make_request(request);
    
    // Log the API call
    log_api_call(endpoint, method, payload, response);
    
    // Handle errors
    if (response.status_code >= 400) {
        handle_http_error(response, endpoint);
    }
    
    return response;
}

OrderResult FXCMBroker::parse_order_result(const HttpResponse& response) {
    OrderResult result;
    
    try {
        json data = json::parse(response.body);
        
        result.success = data.value("success", false);
        result.timestamp = std::chrono::system_clock::now();
        
        if (result.success) {
            if (data.contains("order_id")) {
                result.order_id = data["order_id"];
            }
            if (data.contains("trade_id")) {
                result.trade_id = data["trade_id"];
            }
            if (data.contains("executed_price")) {
                result.executed_price = data["executed_price"];
            }
            if (data.contains("executed_amount")) {
                result.executed_amount = data["executed_amount"];
            }
        } else {
            result.error_message = data.value("error", "Unknown error");
        }
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = "Failed to parse order result: " + std::string(e.what());
    }
    
    return result;
}

std::vector<Position> FXCMBroker::parse_positions(const HttpResponse& response) {
    std::vector<Position> positions;
    
    try {
        json data = json::parse(response.body);
        
        if (data.contains("positions")) {
            for (const auto& pos_data : data["positions"]) {
                Position position;
                position.position_id = pos_data.value("position_id", "");
                position.symbol = pos_data.value("symbol", "");
                position.side = pos_data.value("side", "");
                position.amount = pos_data.value("amount", 0.0);
                position.open_price = pos_data.value("open_price", 0.0);
                position.current_price = pos_data.value("current_price", 0.0);
                position.unrealized_pnl = pos_data.value("unrealized_pnl", 0.0);
                position.realized_pnl = pos_data.value("realized_pnl", 0.0);
                position.timestamp = std::chrono::system_clock::now();
                
                positions.push_back(position);
            }
        }
        
    } catch (const std::exception& e) {
        Logger::error("Failed to parse positions: {}", e.what());
    }
    
    return positions;
}

std::vector<Order> FXCMBroker::parse_orders(const HttpResponse& response) {
    // FXCM doesn't have separate orders, so we convert positions to orders
    auto positions = parse_positions(response);
    std::vector<Order> orders;
    
    for (const auto& position : positions) {
        Order order;
        order.order_id = position.position_id;
        order.symbol = position.symbol;
        order.side = position.side;
        order.amount = position.amount;
        order.order_type = "market";
        order.status = "filled";
        order.filled_amount = position.amount;
        order.average_price = position.open_price;
        order.timestamp = position.timestamp;
        
        orders.push_back(order);
    }
    
    return orders;
}

AccountInfo FXCMBroker::parse_account_info(const HttpResponse& response) {
    AccountInfo account_info;
    
    try {
        json data = json::parse(response.body);
        
        if (data.contains("account")) {
            const auto& acc = data["account"];
            account_info.account_id = acc.value("account_id", "");
            account_info.balance = acc.value("balance", 0.0);
            account_info.equity = acc.value("equity", 0.0);
            account_info.used_margin = acc.value("used_margin", 0.0);
            account_info.free_margin = acc.value("free_margin", 0.0);
            account_info.currency = acc.value("currency", "USD");
            account_info.timestamp = std::chrono::system_clock::now();
        }
        
    } catch (const std::exception& e) {
        Logger::error("Failed to parse account info: {}", e.what());
    }
    
    return account_info;
}

MarketData FXCMBroker::parse_market_data(const HttpResponse& response) {
    MarketData market_data;
    
    try {
        json data = json::parse(response.body);
        
        market_data.symbol = data.value("symbol", "");
        market_data.bid = data.value("bid", 0.0);
        market_data.ask = data.value("ask", 0.0);
        market_data.spread = data.value("spread", 0.0);
        market_data.timestamp = std::chrono::system_clock::now();
        
    } catch (const std::exception& e) {
        Logger::error("Failed to parse market data: {}", e.what());
    }
    
    return market_data;
}

std::vector<OHLCV> FXCMBroker::parse_historical_data(const HttpResponse& response) {
    std::vector<OHLCV> ohlcv_data;
    
    try {
        json data = json::parse(response.body);
        
        if (data.contains("data")) {
            for (const auto& candle : data["data"]) {
                OHLCV ohlcv;
                ohlcv.open = candle.value("open", 0.0);
                ohlcv.high = candle.value("high", 0.0);
                ohlcv.low = candle.value("low", 0.0);
                ohlcv.close = candle.value("close", 0.0);
                ohlcv.volume = candle.value("volume", 0.0);
                ohlcv.timestamp = std::chrono::system_clock::now(); // TODO: Parse actual timestamp
                
                ohlcv_data.push_back(ohlcv);
            }
        }
        
    } catch (const std::exception& e) {
        Logger::error("Failed to parse historical data: {}", e.what());
    }
    
    return ohlcv_data;
}

nlohmann::json FXCMBroker::order_request_to_json(const OrderRequest& request) {
    json order_json;
    
    order_json["symbol"] = request.symbol;
    order_json["side"] = request.side;
    order_json["amount"] = request.amount;
    order_json["order_type"] = request.order_type;
    
    if (request.price.has_value()) {
        order_json["price"] = request.price.value();
    }
    
    if (request.stop_loss.has_value()) {
        order_json["stop_loss"] = request.stop_loss.value();
    }
    
    if (request.take_profit.has_value()) {
        order_json["take_profit"] = request.take_profit.value();
    }
    
    order_json["time_in_force"] = request.time_in_force;
    order_json["comment"] = "Gemini AI Bot";
    
    return order_json;
}

nlohmann::json FXCMBroker::order_modification_to_json(const OrderModification& modification) {
    json mod_json;
    
    if (modification.new_price.has_value()) {
        mod_json["price"] = modification.new_price.value();
    }
    
    if (modification.new_stop_loss.has_value()) {
        mod_json["stop_loss"] = modification.new_stop_loss.value();
    }
    
    if (modification.new_take_profit.has_value()) {
        mod_json["take_profit"] = modification.new_take_profit.value();
    }
    
    if (modification.new_amount.has_value()) {
        mod_json["amount"] = modification.new_amount.value();
    }
    
    return mod_json;
}

void FXCMBroker::handle_http_error(const HttpResponse& response, const std::string& operation) {
    std::string error_msg = "HTTP " + std::to_string(response.status_code) + " error for " + operation;
    
    try {
        json error_data = json::parse(response.body);
        if (error_data.contains("error")) {
            error_msg += ": " + error_data["error"].get<std::string>();
        }
    } catch (const std::exception&) {
        // If we can't parse the error, just use the status code
        error_msg += ": " + response.body;
    }
    
    Logger::error(error_msg);
}

void FXCMBroker::log_api_call(const std::string& endpoint, const std::string& method, 
                             const nlohmann::json& payload, const HttpResponse& response) {
    Logger::debug("FXCM API Call: {} {} -> {} ({}ms)", 
                 method, endpoint, response.status_code, response.duration_ms);
    
    if (!payload.empty()) {
        Logger::debug("Request payload: {}", payload.dump());
    }
    
    if (response.status_code >= 400) {
        Logger::debug("Error response: {}", response.body);
    }
}

} // namespace forex_bot