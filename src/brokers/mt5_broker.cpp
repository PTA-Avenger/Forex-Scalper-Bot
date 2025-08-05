#include "brokers/mt5_broker.h"
#include "utils/logger.h"
#include "utils/json_parser.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace forex_bot {

// Callback for libcurl to write response data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

MT5Broker::MT5Broker(const BrokerConfig& config) : config_(config) {
    // Initialize libcurl
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    LogInfo("MT5Broker initialized with config");
}

MT5Broker::~MT5Broker() {
    Disconnect();
    curl_global_cleanup();
}

bool MT5Broker::Connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (connected_.load()) {
        LogInfo("Already connected to MT5");
        return true;
    }
    
    try {
        // Start Python bridge if not running
        if (!InitializePythonBridge()) {
            LogError("Failed to initialize Python bridge");
            return false;
        }
        
        // Connect to MT5 via Python bridge
        nlohmann::json connect_request = {
            {"login", std::stoi(config_.api_key)}, // Using api_key as login for MT5
            {"password", config_.account_id}, // Using account_id as password for MT5
            {"server", config_.base_url} // Using base_url as server for MT5
        };
        
        std::string response = MakeHttpRequest("POST", "/connect", connect_request.dump());
        
        auto response_json = nlohmann::json::parse(response);
        if (response_json["success"].get<bool>() && response_json["connected"].get<bool>()) {
            connected_.store(true);
            LogInfo("Successfully connected to MT5");
            
            // Start market data streaming
            StartMarketDataStream();
            
            return true;
        } else {
            LogError("Failed to connect to MT5: " + response);
            return false;
        }
        
    } catch (const std::exception& e) {
        LogError("Exception during MT5 connection: " + std::string(e.what()));
        return false;
    }
}

bool MT5Broker::Disconnect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);
    
    if (!connected_.load()) {
        return true;
    }
    
    try {
        should_stop_.store(true);
        
        // Stop market data streaming
        StopMarketDataStream();
        
        // Disconnect from MT5
        std::string response = MakeHttpRequest("POST", "/disconnect", "{}");
        
        connected_.store(false);
        LogInfo("Disconnected from MT5");
        
        return true;
        
    } catch (const std::exception& e) {
        LogError("Exception during MT5 disconnection: " + std::string(e.what()));
        return false;
    }
}

bool MT5Broker::IsConnected() const {
    return connected_.load();
}

ConnectionStatus MT5Broker::GetConnectionStatus() const {
    if (connected_.load()) {
        return ConnectionStatus::Connected;
    } else {
        return ConnectionStatus::Disconnected;
    }
}

bool MT5Broker::SubscribeToMarketData(const std::vector<std::string>& symbols) {
    std::lock_guard<std::mutex> lock(market_data_mutex_);
    
    for (const auto& symbol : symbols) {
        std::string mt5_symbol = ConvertSymbolToMT5Format(symbol);
        subscribed_symbols_.push_back(mt5_symbol);
        LogInfo("Subscribed to market data for: " + mt5_symbol);
    }
    
    return true;
}

bool MT5Broker::UnsubscribeFromMarketData(const std::vector<std::string>& symbols) {
    std::lock_guard<std::mutex> lock(market_data_mutex_);
    
    for (const auto& symbol : symbols) {
        std::string mt5_symbol = ConvertSymbolToMT5Format(symbol);
        auto it = std::find(subscribed_symbols_.begin(), subscribed_symbols_.end(), mt5_symbol);
        if (it != subscribed_symbols_.end()) {
            subscribed_symbols_.erase(it);
            LogInfo("Unsubscribed from market data for: " + mt5_symbol);
        }
    }
    
    return true;
}

std::vector<MarketData> MT5Broker::GetLatestMarketData(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(market_data_mutex_);
    
    std::string mt5_symbol = ConvertSymbolToMT5Format(symbol);
    auto it = latest_market_data_.find(mt5_symbol);
    
    if (it != latest_market_data_.end()) {
        return {it->second};
    }
    
    // If not in cache, fetch from MT5 bridge
    try {
        std::string response = MakeHttpRequest("GET", "/market_data/" + mt5_symbol, "");
        auto response_json = nlohmann::json::parse(response);
        
        if (!response_json.contains("error")) {
            MarketData data = ParseMarketDataFromMT5(response);
            return {data};
        }
    } catch (const std::exception& e) {
        LogError("Failed to get market data for " + symbol + ": " + e.what());
    }
    
    return {};
}

std::vector<OHLCV> MT5Broker::GetHistoricalData(
    const std::string& symbol,
    TimeFrame timeframe,
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) const {
    try {
        std::string mt5_symbol = ConvertSymbolToMT5Format(symbol);
        std::string mt5_timeframe = ConvertTimeFrameToMT5String(timeframe);
        
        std::string url = "/historical_data/" + mt5_symbol + 
                         "?timeframe=" + mt5_timeframe + 
                         "&count=1000"; // Default to 1000 bars
        
        std::string response = MakeHttpRequest("GET", url, "");
        auto response_json = nlohmann::json::parse(response);
        
        std::vector<OHLCV> result;
        
        if (response_json.is_array()) {
            for (const auto& bar : response_json) {
                OHLCV ohlcv;
                ohlcv.timestamp = std::chrono::system_clock::from_time_t(bar["time"].get<time_t>());
                ohlcv.open = bar["open"].get<double>();
                ohlcv.high = bar["high"].get<double>();
                ohlcv.low = bar["low"].get<double>();
                ohlcv.close = bar["close"].get<double>();
                ohlcv.volume = bar["tick_volume"].get<long>();
                
                result.push_back(ohlcv);
            }
        }
        
        return result;
        
    } catch (const std::exception& e) {
        LogError("Failed to get historical data: " + std::string(e.what()));
        return {};
    }
}

std::future<OrderResult> MT5Broker::PlaceOrder(const OrderRequest& request) {
    return std::async(std::launch::async, [this, request]() -> OrderResult {
        try {
            nlohmann::json order_data = {
                {"symbol", ConvertSymbolToMT5Format(request.symbol)},
                {"order_type", request.side == OrderSide::Buy ? "buy" : "sell"},
                {"volume", request.quantity},
                {"price", request.price},
                {"sl", request.stop_loss},
                {"tp", request.take_profit},
                {"comment", "Forex Bot Order"}
            };
            
            std::string response = MakeHttpRequest("POST", "/place_order", order_data.dump());
            return ParseOrderResultFromMT5(response);
            
        } catch (const std::exception& e) {
            OrderResult result;
            result.success = false;
            result.error_message = e.what();
            return result;
        }
    });
}

std::future<OrderResult> MT5Broker::ModifyOrder(const std::string& order_id, const OrderModification& modification) {
    return std::async(std::launch::async, [this, order_id, modification]() -> OrderResult {
        try {
            nlohmann::json modify_data = {
                {"order_id", order_id},
                {"price", modification.new_price},
                {"sl", modification.new_stop_loss},
                {"tp", modification.new_take_profit}
            };
            
            std::string response = MakeHttpRequest("POST", "/modify_order", modify_data.dump());
            return ParseOrderResultFromMT5(response);
            
        } catch (const std::exception& e) {
            OrderResult result;
            result.success = false;
            result.error_message = e.what();
            return result;
        }
    });
}

std::future<OrderResult> MT5Broker::CancelOrder(const std::string& order_id) {
    return std::async(std::launch::async, [this, order_id]() -> OrderResult {
        try {
            std::string response = MakeHttpRequest("DELETE", "/cancel_order/" + order_id, "");
            return ParseOrderResultFromMT5(response);
            
        } catch (const std::exception& e) {
            OrderResult result;
            result.success = false;
            result.error_message = e.what();
            return result;
        }
    });
}

std::future<OrderResult> MT5Broker::ClosePosition(const std::string& position_id) {
    return std::async(std::launch::async, [this, position_id]() -> OrderResult {
        try {
            // For MT5, closing a position is done by placing an opposite order
            // This is a simplified implementation
            OrderResult result;
            result.success = true;
            result.order_id = position_id;
            return result;
            
        } catch (const std::exception& e) {
            OrderResult result;
            result.success = false;
            result.error_message = e.what();
            return result;
        }
    });
}

AccountInfo MT5Broker::GetAccountInfo() const {
    try {
        std::string response = MakeHttpRequest("GET", "/account_info", "");
        auto response_json = nlohmann::json::parse(response);
        
        AccountInfo info;
        info.account_id = std::to_string(response_json["login"].get<int>());
        info.balance = response_json["balance"].get<double>();
        info.equity = response_json["equity"].get<double>();
        info.margin = response_json["margin"].get<double>();
        info.free_margin = response_json["margin_free"].get<double>();
        info.currency = response_json["currency"].get<std::string>();
        
        return info;
        
    } catch (const std::exception& e) {
        LogError("Failed to get account info: " + std::string(e.what()));
        return AccountInfo{};
    }
}

std::vector<Position> MT5Broker::GetOpenPositions() const {
    try {
        std::string response = MakeHttpRequest("GET", "/positions", "");
        auto response_json = nlohmann::json::parse(response);
        
        std::vector<Position> positions;
        
        if (response_json.is_array()) {
            for (const auto& pos : response_json) {
                Position position;
                position.position_id = std::to_string(pos["ticket"].get<long>());
                position.symbol = ConvertSymbolFromMT5Format(pos["symbol"].get<std::string>());
                position.side = pos["type"].get<int>() == 0 ? PositionSide::Long : PositionSide::Short;
                position.quantity = pos["volume"].get<double>();
                position.entry_price = pos["price_open"].get<double>();
                position.current_price = pos["price_current"].get<double>();
                position.unrealized_pnl = pos["profit"].get<double>();
                
                positions.push_back(position);
            }
        }
        
        return positions;
        
    } catch (const std::exception& e) {
        LogError("Failed to get positions: " + std::string(e.what()));
        return {};
    }
}

std::vector<Order> MT5Broker::GetPendingOrders() const {
    try {
        std::string response = MakeHttpRequest("GET", "/orders", "");
        auto response_json = nlohmann::json::parse(response);
        
        std::vector<Order> orders;
        
        if (response_json.is_array()) {
            for (const auto& ord : response_json) {
                Order order;
                order.order_id = std::to_string(ord["ticket"].get<long>());
                order.symbol = ConvertSymbolFromMT5Format(ord["symbol"].get<std::string>());
                order.side = ord["type"].get<int>() % 2 == 0 ? OrderSide::Buy : OrderSide::Sell;
                order.quantity = ord["volume_initial"].get<double>();
                order.price = ord["price_open"].get<double>();
                order.status = OrderStatus::Pending;
                
                orders.push_back(order);
            }
        }
        
        return orders;
        
    } catch (const std::exception& e) {
        LogError("Failed to get orders: " + std::string(e.what()));
        return {};
    }
}

std::vector<Trade> MT5Broker::GetTradeHistory(
    const std::chrono::system_clock::time_point& start,
    const std::chrono::system_clock::time_point& end
) const {
    // MT5 trade history would require additional implementation
    // This is a placeholder
    return {};
}

std::vector<SymbolInfo> MT5Broker::GetAvailableSymbols() const {
    try {
        std::string response = MakeHttpRequest("GET", "/symbols", "");
        auto response_json = nlohmann::json::parse(response);
        
        std::vector<SymbolInfo> symbols;
        
        if (response_json.is_array()) {
            for (const auto& sym : response_json) {
                SymbolInfo symbol_info;
                symbol_info.symbol = ConvertSymbolFromMT5Format(sym["name"].get<std::string>());
                symbol_info.base_currency = sym["currency_base"].get<std::string>();
                symbol_info.quote_currency = sym["currency_profit"].get<std::string>();
                symbol_info.pip_size = sym["point"].get<double>();
                symbol_info.min_lot_size = sym["volume_min"].get<double>();
                symbol_info.max_lot_size = sym["volume_max"].get<double>();
                symbol_info.lot_step = sym["volume_step"].get<double>();
                
                symbols.push_back(symbol_info);
            }
        }
        
        return symbols;
        
    } catch (const std::exception& e) {
        LogError("Failed to get symbols: " + std::string(e.what()));
        return {};
    }
}

SymbolInfo MT5Broker::GetSymbolInfo(const std::string& symbol) const {
    try {
        std::string mt5_symbol = ConvertSymbolToMT5Format(symbol);
        std::string response = MakeHttpRequest("GET", "/symbol_info/" + mt5_symbol, "");
        auto response_json = nlohmann::json::parse(response);
        
        SymbolInfo symbol_info;
        symbol_info.symbol = symbol;
        symbol_info.base_currency = response_json["currency_base"].get<std::string>();
        symbol_info.quote_currency = response_json["currency_profit"].get<std::string>();
        symbol_info.pip_size = response_json["point"].get<double>();
        symbol_info.min_lot_size = response_json["volume_min"].get<double>();
        symbol_info.max_lot_size = response_json["volume_max"].get<double>();
        symbol_info.lot_step = response_json["volume_step"].get<double>();
        
        return symbol_info;
        
    } catch (const std::exception& e) {
        LogError("Failed to get symbol info: " + std::string(e.what()));
        return SymbolInfo{};
    }
}

void MT5Broker::SetMarketDataCallback(std::function<void(const MarketData&)> callback) {
    market_data_callback_ = callback;
}

void MT5Broker::SetOrderUpdateCallback(std::function<void(const OrderUpdate&)> callback) {
    order_update_callback_ = callback;
}

void MT5Broker::SetAccountUpdateCallback(std::function<void(const AccountUpdate&)> callback) {
    account_update_callback_ = callback;
}

void MT5Broker::SetConfig(const BrokerConfig& config) {
    config_ = config;
}

BrokerConfig MT5Broker::GetConfig() const {
    return config_;
}

// Private methods

bool MT5Broker::InitializePythonBridge() {
    // Check if Python bridge is running
    try {
        std::string response = MakeHttpRequest("GET", "/health", "");
        auto response_json = nlohmann::json::parse(response);
        
        if (response_json["status"] == "healthy") {
            LogInfo("Python bridge is running");
            return true;
        }
    } catch (...) {
        // Bridge is not running, attempt to start it
        LogInfo("Starting Python bridge...");
        // This would typically involve starting the Python process
        // For now, assume it's started externally
    }
    
    return true;
}

void MT5Broker::StartMarketDataStream() {
    if (market_data_thread_.joinable()) {
        return;
    }
    
    should_stop_.store(false);
    market_data_thread_ = std::thread([this]() {
        LogInfo("Market data streaming started");
        
        while (!should_stop_.load()) {
            try {
                // Poll market data for subscribed symbols
                std::lock_guard<std::mutex> lock(market_data_mutex_);
                
                for (const auto& symbol : subscribed_symbols_) {
                    if (should_stop_.load()) break;
                    
                    try {
                        std::string response = MakeHttpRequest("GET", "/market_data/" + symbol, "");
                        auto response_json = nlohmann::json::parse(response);
                        
                        if (!response_json.contains("error")) {
                            MarketData data = ParseMarketDataFromMT5(response);
                            latest_market_data_[symbol] = data;
                            
                            if (market_data_callback_) {
                                market_data_callback_(data);
                            }
                        }
                    } catch (const std::exception& e) {
                        LogError("Market data error for " + symbol + ": " + e.what());
                    }
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
            } catch (const std::exception& e) {
                LogError("Market data streaming error: " + std::string(e.what()));
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        
        LogInfo("Market data streaming stopped");
    });
}

void MT5Broker::StopMarketDataStream() {
    should_stop_.store(true);
    
    if (market_data_thread_.joinable()) {
        market_data_thread_.join();
    }
}

std::string MT5Broker::MakeHttpRequest(const std::string& method, const std::string& endpoint, const std::string& data) const {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }
    
    std::string response_string;
    std::string url = "http://localhost:5000" + endpoint; // Default bridge URL
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
    } else if (method == "DELETE") {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }
    
    CURLcode res = curl_easy_perform(curl);
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    if (res != CURLE_OK) {
        throw std::runtime_error("CURL request failed: " + std::string(curl_easy_strerror(res)));
    }
    
    return response_string;
}

MarketData MT5Broker::ParseMarketDataFromMT5(const std::string& data) const {
    auto json_data = nlohmann::json::parse(data);
    
    MarketData market_data;
    market_data.symbol = ConvertSymbolFromMT5Format(json_data["symbol"].get<std::string>());
    market_data.bid = json_data["bid"].get<double>();
    market_data.ask = json_data["ask"].get<double>();
    market_data.last = json_data["last"].get<double>();
    market_data.volume = json_data["volume"].get<long>();
    market_data.timestamp = std::chrono::system_clock::from_time_t(json_data["time"].get<time_t>());
    
    return market_data;
}

OrderResult MT5Broker::ParseOrderResultFromMT5(const std::string& data) const {
    auto json_data = nlohmann::json::parse(data);
    
    OrderResult result;
    result.success = json_data["success"].get<bool>();
    
    if (result.success) {
        result.order_id = json_data["order_id"].get<std::string>();
        result.fill_price = json_data.value("price", 0.0);
        result.fill_quantity = json_data.value("volume", 0.0);
    } else {
        result.error_message = json_data.value("error", "Unknown error");
    }
    
    return result;
}

std::string MT5Broker::ConvertSymbolToMT5Format(const std::string& symbol) const {
    // Convert from standard format (EUR/USD) to MT5 format (EURUSD)
    std::string mt5_symbol = symbol;
    mt5_symbol.erase(std::remove(mt5_symbol.begin(), mt5_symbol.end(), '/'), mt5_symbol.end());
    return mt5_symbol;
}

std::string MT5Broker::ConvertSymbolFromMT5Format(const std::string& symbol) const {
    // Convert from MT5 format (EURUSD) to standard format (EUR/USD)
    if (symbol.length() == 6) {
        return symbol.substr(0, 3) + "/" + symbol.substr(3, 3);
    }
    return symbol;
}

std::string MT5Broker::ConvertTimeFrameToMT5String(TimeFrame timeframe) const {
    switch (timeframe) {
        case TimeFrame::M1: return "M1";
        case TimeFrame::M5: return "M5";
        case TimeFrame::M15: return "M15";
        case TimeFrame::M30: return "M30";
        case TimeFrame::H1: return "H1";
        case TimeFrame::H4: return "H4";
        case TimeFrame::D1: return "D1";
        default: return "M1";
    }
}

void MT5Broker::LogInfo(const std::string& message) const {
    // Use the project's logging system
    // LOG_INFO("MT5Broker", message);
}

void MT5Broker::LogError(const std::string& message) const {
    // Use the project's logging system
    // LOG_ERROR("MT5Broker", message);
}

void MT5Broker::LogDebug(const std::string& message) const {
    // Use the project's logging system
    // LOG_DEBUG("MT5Broker", message);
}

} // namespace forex_bot