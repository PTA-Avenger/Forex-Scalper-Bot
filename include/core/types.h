#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <nlohmann/json.hpp>

namespace forex_bot {

// Forward declarations
class TradingEngine;
class BrokerInterface;
class StrategyBase;

// Enumerations
enum class OrderType {
    MARKET,
    LIMIT,
    STOP,
    STOP_LIMIT
};

enum class OrderSide {
    BUY,
    SELL
};

enum class OrderStatus {
    PENDING,
    FILLED,
    PARTIALLY_FILLED,
    CANCELLED,
    REJECTED
};

enum class SignalType {
    BUY,
    SELL,
    CLOSE,
    HOLD
};

enum class TimeFrame {
    TICK,
    M1,
    M5,
    M15,
    M30,
    H1,
    H4,
    D1
};

// Core data structures
struct TickData {
    std::string symbol;
    double bid;
    double ask;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    
    // Utility methods
    double mid() const { return (bid + ask) / 2.0; }
    double spread() const { return ask - bid; }
    
    // Serialization
    nlohmann::json to_json() const;
    static TickData from_json(const nlohmann::json& j);
};

struct OHLCV {
    std::string symbol;
    double open;
    double high;
    double low;
    double close;
    double volume;
    std::chrono::system_clock::time_point timestamp;
    TimeFrame timeframe;
    
    // Utility methods
    double typical_price() const { return (high + low + close) / 3.0; }
    double range() const { return high - low; }
    bool is_bullish() const { return close > open; }
    bool is_bearish() const { return close < open; }
    
    // Serialization
    nlohmann::json to_json() const;
    static OHLCV from_json(const nlohmann::json& j);
};

struct TradingSignal {
    std::string symbol;
    SignalType type;
    double confidence;
    double suggested_price;
    double suggested_size;
    double stop_loss;
    double take_profit;
    std::string strategy_name;
    std::chrono::system_clock::time_point timestamp;
    std::unordered_map<std::string, double> metadata;
    
    // Serialization
    nlohmann::json to_json() const;
    static TradingSignal from_json(const nlohmann::json& j);
};

struct Order {
    std::string id;
    std::string symbol;
    OrderType type;
    OrderSide side;
    double quantity;
    double price;
    double stop_loss;
    double take_profit;
    OrderStatus status;
    std::string strategy_name;
    std::chrono::system_clock::time_point created_at;
    std::chrono::system_clock::time_point updated_at;
    std::string broker_order_id;
    
    // Serialization
    nlohmann::json to_json() const;
    static Order from_json(const nlohmann::json& j);
};

struct Position {
    std::string id;
    std::string symbol;
    OrderSide side;
    double quantity;
    double avg_entry_price;
    double current_price;
    double unrealized_pnl;
    double realized_pnl;
    std::chrono::system_clock::time_point open_time;
    std::string strategy_name;
    std::vector<std::string> order_ids;
    
    // Utility methods
    double market_value() const { return quantity * current_price; }
    double total_pnl() const { return unrealized_pnl + realized_pnl; }
    std::chrono::duration<double> duration() const {
        return std::chrono::system_clock::now() - open_time;
    }
    
    // Serialization
    nlohmann::json to_json() const;
    static Position from_json(const nlohmann::json& j);
};

struct Trade {
    std::string id;
    std::string symbol;
    OrderSide side;
    double entry_price;
    double exit_price;
    double quantity;
    double pnl;
    double commission;
    std::chrono::system_clock::time_point entry_time;
    std::chrono::system_clock::time_point exit_time;
    std::string strategy_name;
    std::string entry_order_id;
    std::string exit_order_id;
    
    // Utility methods
    double return_percentage() const {
        return (exit_price - entry_price) / entry_price * 100.0 * (side == OrderSide::BUY ? 1.0 : -1.0);
    }
    
    std::chrono::duration<double> duration() const {
        return exit_time - entry_time;
    }
    
    bool is_winning() const { return pnl > 0; }
    
    // Serialization
    nlohmann::json to_json() const;
    static Trade from_json(const nlohmann::json& j);
};

struct AccountInfo {
    std::string account_id;
    double balance;
    double equity;
    double margin_used;
    double margin_available;
    double unrealized_pnl;
    double realized_pnl;
    std::string currency;
    std::chrono::system_clock::time_point last_updated;
    
    // Utility methods
    double margin_level() const {
        return margin_used > 0 ? (equity / margin_used) * 100.0 : 0.0;
    }
    
    double free_margin() const { return equity - margin_used; }
    
    // Serialization
    nlohmann::json to_json() const;
    static AccountInfo from_json(const nlohmann::json& j);
};

struct MarketData {
    std::string symbol;
    TickData current_tick;
    std::vector<OHLCV> bars;
    std::unordered_map<TimeFrame, std::vector<OHLCV>> historical_data;
    std::chrono::system_clock::time_point last_updated;
    
    // Utility methods
    const std::vector<OHLCV>& get_bars(TimeFrame tf) const {
        auto it = historical_data.find(tf);
        return it != historical_data.end() ? it->second : bars;
    }
    
    // Serialization
    nlohmann::json to_json() const;
    static MarketData from_json(const nlohmann::json& j);
};

struct OrderResult {
    bool success;
    std::string order_id;
    std::string broker_order_id;
    std::string error_message;
    double executed_price;
    double executed_quantity;
    std::chrono::system_clock::time_point execution_time;
    
    // Serialization
    nlohmann::json to_json() const;
    static OrderResult from_json(const nlohmann::json& j);
};

struct RiskMetrics {
    double daily_drawdown;
    double max_daily_drawdown;
    double total_exposure;
    double max_exposure;
    double portfolio_var; // Value at Risk
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double kelly_fraction;
    int risk_score; // 0-100
    std::vector<std::pair<std::string, double>> correlations;
    std::chrono::system_clock::time_point last_updated;
    
    // Serialization
    nlohmann::json to_json() const;
    static RiskMetrics from_json(const nlohmann::json& j);
};

struct PerformanceMetrics {
    double total_return;
    double annualized_return;
    double sharpe_ratio;
    double sortino_ratio;
    double max_drawdown;
    double max_drawdown_duration; // in days
    double win_rate;
    double profit_factor;
    double avg_win;
    double avg_loss;
    double avg_trade_duration; // in minutes
    int total_trades;
    int winning_trades;
    int losing_trades;
    double volatility;
    double calmar_ratio;
    double information_ratio;
    
    // Serialization
    nlohmann::json to_json() const;
    static PerformanceMetrics from_json(const nlohmann::json& j);
};

struct TimeRange {
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
    
    bool contains(const std::chrono::system_clock::time_point& time) const {
        return time >= start && time <= end;
    }
    
    std::chrono::duration<double> duration() const {
        return end - start;
    }
};

// Configuration structures
struct TradingConfig {
    std::vector<std::string> symbols;
    std::vector<TimeFrame> timeframes;
    int max_positions;
    bool paper_trading;
    double min_trade_size;
    double max_trade_size;
    
    nlohmann::json to_json() const;
    static TradingConfig from_json(const nlohmann::json& j);
};

struct BrokerConfig {
    std::string type; // "oanda", "mt4", "mt5"
    std::string api_key;
    std::string account_id;
    std::string environment; // "practice", "live"
    std::string base_url;
    int timeout_ms;
    int max_retries;
    
    nlohmann::json to_json() const;
    static BrokerConfig from_json(const nlohmann::json& j);
};

struct RiskConfig {
    double max_daily_drawdown;
    double max_position_size;
    double kelly_fraction;
    double correlation_limit;
    double var_confidence_level;
    int var_lookback_days;
    
    nlohmann::json to_json() const;
    static RiskConfig from_json(const nlohmann::json& j);
};

// Utility functions
std::string to_string(OrderType type);
std::string to_string(OrderSide side);
std::string to_string(OrderStatus status);
std::string to_string(SignalType signal);
std::string to_string(TimeFrame timeframe);

OrderType order_type_from_string(const std::string& str);
OrderSide order_side_from_string(const std::string& str);
OrderStatus order_status_from_string(const std::string& str);
SignalType signal_type_from_string(const std::string& str);
TimeFrame timeframe_from_string(const std::string& str);

// Time utilities
std::chrono::system_clock::time_point parse_iso8601(const std::string& iso_string);
std::string to_iso8601(const std::chrono::system_clock::time_point& time);
std::chrono::milliseconds get_timeframe_duration(TimeFrame tf);

} // namespace forex_bot