#include "core/types.h"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace forex_bot {

// Utility function implementations
std::string to_string(OrderType type) {
    switch (type) {
        case OrderType::MARKET: return "MARKET";
        case OrderType::LIMIT: return "LIMIT";
        case OrderType::STOP: return "STOP";
        case OrderType::STOP_LIMIT: return "STOP_LIMIT";
        default: return "UNKNOWN";
    }
}

std::string to_string(OrderSide side) {
    switch (side) {
        case OrderSide::BUY: return "BUY";
        case OrderSide::SELL: return "SELL";
        default: return "UNKNOWN";
    }
}

std::string to_string(OrderStatus status) {
    switch (status) {
        case OrderStatus::PENDING: return "PENDING";
        case OrderStatus::FILLED: return "FILLED";
        case OrderStatus::PARTIALLY_FILLED: return "PARTIALLY_FILLED";
        case OrderStatus::CANCELLED: return "CANCELLED";
        case OrderStatus::REJECTED: return "REJECTED";
        default: return "UNKNOWN";
    }
}

std::string to_string(SignalType signal) {
    switch (signal) {
        case SignalType::BUY: return "BUY";
        case SignalType::SELL: return "SELL";
        case SignalType::CLOSE: return "CLOSE";
        case SignalType::HOLD: return "HOLD";
        default: return "UNKNOWN";
    }
}

std::string to_string(TimeFrame timeframe) {
    switch (timeframe) {
        case TimeFrame::TICK: return "TICK";
        case TimeFrame::M1: return "M1";
        case TimeFrame::M5: return "M5";
        case TimeFrame::M15: return "M15";
        case TimeFrame::M30: return "M30";
        case TimeFrame::H1: return "H1";
        case TimeFrame::H4: return "H4";
        case TimeFrame::D1: return "D1";
        default: return "UNKNOWN";
    }
}

OrderType order_type_from_string(const std::string& str) {
    if (str == "MARKET") return OrderType::MARKET;
    if (str == "LIMIT") return OrderType::LIMIT;
    if (str == "STOP") return OrderType::STOP;
    if (str == "STOP_LIMIT") return OrderType::STOP_LIMIT;
    throw std::invalid_argument("Invalid OrderType: " + str);
}

OrderSide order_side_from_string(const std::string& str) {
    if (str == "BUY") return OrderSide::BUY;
    if (str == "SELL") return OrderSide::SELL;
    throw std::invalid_argument("Invalid OrderSide: " + str);
}

OrderStatus order_status_from_string(const std::string& str) {
    if (str == "PENDING") return OrderStatus::PENDING;
    if (str == "FILLED") return OrderStatus::FILLED;
    if (str == "PARTIALLY_FILLED") return OrderStatus::PARTIALLY_FILLED;
    if (str == "CANCELLED") return OrderStatus::CANCELLED;
    if (str == "REJECTED") return OrderStatus::REJECTED;
    throw std::invalid_argument("Invalid OrderStatus: " + str);
}

SignalType signal_type_from_string(const std::string& str) {
    if (str == "BUY") return SignalType::BUY;
    if (str == "SELL") return SignalType::SELL;
    if (str == "CLOSE") return SignalType::CLOSE;
    if (str == "HOLD") return SignalType::HOLD;
    throw std::invalid_argument("Invalid SignalType: " + str);
}

TimeFrame timeframe_from_string(const std::string& str) {
    if (str == "TICK") return TimeFrame::TICK;
    if (str == "M1") return TimeFrame::M1;
    if (str == "M5") return TimeFrame::M5;
    if (str == "M15") return TimeFrame::M15;
    if (str == "M30") return TimeFrame::M30;
    if (str == "H1") return TimeFrame::H1;
    if (str == "H4") return TimeFrame::H4;
    if (str == "D1") return TimeFrame::D1;
    throw std::invalid_argument("Invalid TimeFrame: " + str);
}

// Time utilities
std::chrono::system_clock::time_point parse_iso8601(const std::string& iso_string) {
    std::tm tm = {};
    std::istringstream ss(iso_string);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    
    if (ss.fail()) {
        throw std::invalid_argument("Invalid ISO8601 format: " + iso_string);
    }
    
    return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

std::string to_iso8601(const std::chrono::system_clock::time_point& time) {
    auto time_t = std::chrono::system_clock::to_time_t(time);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

std::chrono::milliseconds get_timeframe_duration(TimeFrame tf) {
    switch (tf) {
        case TimeFrame::TICK: return std::chrono::milliseconds(0);
        case TimeFrame::M1: return std::chrono::minutes(1);
        case TimeFrame::M5: return std::chrono::minutes(5);
        case TimeFrame::M15: return std::chrono::minutes(15);
        case TimeFrame::M30: return std::chrono::minutes(30);
        case TimeFrame::H1: return std::chrono::hours(1);
        case TimeFrame::H4: return std::chrono::hours(4);
        case TimeFrame::D1: return std::chrono::hours(24);
        default: return std::chrono::milliseconds(0);
    }
}

// JSON serialization implementations
nlohmann::json TickData::to_json() const {
    return nlohmann::json{
        {"symbol", symbol},
        {"bid", bid},
        {"ask", ask},
        {"volume", volume},
        {"timestamp", to_iso8601(timestamp)}
    };
}

TickData TickData::from_json(const nlohmann::json& j) {
    return TickData{
        j["symbol"].get<std::string>(),
        j["bid"].get<double>(),
        j["ask"].get<double>(),
        j["volume"].get<double>(),
        parse_iso8601(j["timestamp"].get<std::string>())
    };
}

nlohmann::json OHLCV::to_json() const {
    return nlohmann::json{
        {"symbol", symbol},
        {"open", open},
        {"high", high},
        {"low", low},
        {"close", close},
        {"volume", volume},
        {"timestamp", to_iso8601(timestamp)},
        {"timeframe", to_string(timeframe)}
    };
}

OHLCV OHLCV::from_json(const nlohmann::json& j) {
    return OHLCV{
        j["symbol"].get<std::string>(),
        j["open"].get<double>(),
        j["high"].get<double>(),
        j["low"].get<double>(),
        j["close"].get<double>(),
        j["volume"].get<double>(),
        parse_iso8601(j["timestamp"].get<std::string>()),
        timeframe_from_string(j["timeframe"].get<std::string>())
    };
}

nlohmann::json TradingSignal::to_json() const {
    nlohmann::json j{
        {"symbol", symbol},
        {"type", to_string(type)},
        {"confidence", confidence},
        {"suggested_price", suggested_price},
        {"suggested_size", suggested_size},
        {"stop_loss", stop_loss},
        {"take_profit", take_profit},
        {"strategy_name", strategy_name},
        {"timestamp", to_iso8601(timestamp)}
    };
    
    if (!metadata.empty()) {
        j["metadata"] = metadata;
    }
    
    return j;
}

TradingSignal TradingSignal::from_json(const nlohmann::json& j) {
    TradingSignal signal{
        j["symbol"].get<std::string>(),
        signal_type_from_string(j["type"].get<std::string>()),
        j["confidence"].get<double>(),
        j["suggested_price"].get<double>(),
        j["suggested_size"].get<double>(),
        j["stop_loss"].get<double>(),
        j["take_profit"].get<double>(),
        j["strategy_name"].get<std::string>(),
        parse_iso8601(j["timestamp"].get<std::string>())
    };
    
    if (j.contains("metadata")) {
        signal.metadata = j["metadata"].get<std::unordered_map<std::string, double>>();
    }
    
    return signal;
}

nlohmann::json Order::to_json() const {
    return nlohmann::json{
        {"id", id},
        {"symbol", symbol},
        {"type", to_string(type)},
        {"side", to_string(side)},
        {"quantity", quantity},
        {"price", price},
        {"stop_loss", stop_loss},
        {"take_profit", take_profit},
        {"status", to_string(status)},
        {"strategy_name", strategy_name},
        {"created_at", to_iso8601(created_at)},
        {"updated_at", to_iso8601(updated_at)},
        {"broker_order_id", broker_order_id}
    };
}

Order Order::from_json(const nlohmann::json& j) {
    return Order{
        j["id"].get<std::string>(),
        j["symbol"].get<std::string>(),
        order_type_from_string(j["type"].get<std::string>()),
        order_side_from_string(j["side"].get<std::string>()),
        j["quantity"].get<double>(),
        j["price"].get<double>(),
        j["stop_loss"].get<double>(),
        j["take_profit"].get<double>(),
        order_status_from_string(j["status"].get<std::string>()),
        j["strategy_name"].get<std::string>(),
        parse_iso8601(j["created_at"].get<std::string>()),
        parse_iso8601(j["updated_at"].get<std::string>()),
        j["broker_order_id"].get<std::string>()
    };
}

nlohmann::json Position::to_json() const {
    return nlohmann::json{
        {"id", id},
        {"symbol", symbol},
        {"side", to_string(side)},
        {"quantity", quantity},
        {"avg_entry_price", avg_entry_price},
        {"current_price", current_price},
        {"unrealized_pnl", unrealized_pnl},
        {"realized_pnl", realized_pnl},
        {"open_time", to_iso8601(open_time)},
        {"strategy_name", strategy_name},
        {"order_ids", order_ids}
    };
}

Position Position::from_json(const nlohmann::json& j) {
    return Position{
        j["id"].get<std::string>(),
        j["symbol"].get<std::string>(),
        order_side_from_string(j["side"].get<std::string>()),
        j["quantity"].get<double>(),
        j["avg_entry_price"].get<double>(),
        j["current_price"].get<double>(),
        j["unrealized_pnl"].get<double>(),
        j["realized_pnl"].get<double>(),
        parse_iso8601(j["open_time"].get<std::string>()),
        j["strategy_name"].get<std::string>(),
        j["order_ids"].get<std::vector<std::string>>()
    };
}

nlohmann::json Trade::to_json() const {
    return nlohmann::json{
        {"id", id},
        {"symbol", symbol},
        {"side", to_string(side)},
        {"entry_price", entry_price},
        {"exit_price", exit_price},
        {"quantity", quantity},
        {"pnl", pnl},
        {"commission", commission},
        {"entry_time", to_iso8601(entry_time)},
        {"exit_time", to_iso8601(exit_time)},
        {"strategy_name", strategy_name},
        {"entry_order_id", entry_order_id},
        {"exit_order_id", exit_order_id}
    };
}

Trade Trade::from_json(const nlohmann::json& j) {
    return Trade{
        j["id"].get<std::string>(),
        j["symbol"].get<std::string>(),
        order_side_from_string(j["side"].get<std::string>()),
        j["entry_price"].get<double>(),
        j["exit_price"].get<double>(),
        j["quantity"].get<double>(),
        j["pnl"].get<double>(),
        j["commission"].get<double>(),
        parse_iso8601(j["entry_time"].get<std::string>()),
        parse_iso8601(j["exit_time"].get<std::string>()),
        j["strategy_name"].get<std::string>(),
        j["entry_order_id"].get<std::string>(),
        j["exit_order_id"].get<std::string>()
    };
}

nlohmann::json AccountInfo::to_json() const {
    return nlohmann::json{
        {"account_id", account_id},
        {"balance", balance},
        {"equity", equity},
        {"margin_used", margin_used},
        {"margin_available", margin_available},
        {"unrealized_pnl", unrealized_pnl},
        {"realized_pnl", realized_pnl},
        {"currency", currency},
        {"last_updated", to_iso8601(last_updated)}
    };
}

AccountInfo AccountInfo::from_json(const nlohmann::json& j) {
    return AccountInfo{
        j["account_id"].get<std::string>(),
        j["balance"].get<double>(),
        j["equity"].get<double>(),
        j["margin_used"].get<double>(),
        j["margin_available"].get<double>(),
        j["unrealized_pnl"].get<double>(),
        j["realized_pnl"].get<double>(),
        j["currency"].get<std::string>(),
        parse_iso8601(j["last_updated"].get<std::string>())
    };
}

nlohmann::json OrderResult::to_json() const {
    return nlohmann::json{
        {"success", success},
        {"order_id", order_id},
        {"broker_order_id", broker_order_id},
        {"error_message", error_message},
        {"executed_price", executed_price},
        {"executed_quantity", executed_quantity},
        {"execution_time", to_iso8601(execution_time)}
    };
}

OrderResult OrderResult::from_json(const nlohmann::json& j) {
    return OrderResult{
        j["success"].get<bool>(),
        j["order_id"].get<std::string>(),
        j["broker_order_id"].get<std::string>(),
        j["error_message"].get<std::string>(),
        j["executed_price"].get<double>(),
        j["executed_quantity"].get<double>(),
        parse_iso8601(j["execution_time"].get<std::string>())
    };
}

// Configuration serialization implementations
nlohmann::json TradingConfig::to_json() const {
    return nlohmann::json{
        {"symbols", symbols},
        {"timeframes", [this]() {
            std::vector<std::string> tf_strings;
            for (auto tf : timeframes) {
                tf_strings.push_back(to_string(tf));
            }
            return tf_strings;
        }()},
        {"max_positions", max_positions},
        {"paper_trading", paper_trading},
        {"min_trade_size", min_trade_size},
        {"max_trade_size", max_trade_size}
    };
}

TradingConfig TradingConfig::from_json(const nlohmann::json& j) {
    TradingConfig config;
    config.symbols = j["symbols"].get<std::vector<std::string>>();
    
    auto tf_strings = j["timeframes"].get<std::vector<std::string>>();
    for (const auto& tf_str : tf_strings) {
        config.timeframes.push_back(timeframe_from_string(tf_str));
    }
    
    config.max_positions = j["max_positions"].get<int>();
    config.paper_trading = j["paper_trading"].get<bool>();
    config.min_trade_size = j["min_trade_size"].get<double>();
    config.max_trade_size = j["max_trade_size"].get<double>();
    
    return config;
}

nlohmann::json BrokerConfig::to_json() const {
    return nlohmann::json{
        {"type", type},
        {"api_key", api_key},
        {"account_id", account_id},
        {"environment", environment},
        {"base_url", base_url},
        {"timeout_ms", timeout_ms},
        {"max_retries", max_retries}
    };
}

BrokerConfig BrokerConfig::from_json(const nlohmann::json& j) {
    return BrokerConfig{
        j["type"].get<std::string>(),
        j["api_key"].get<std::string>(),
        j["account_id"].get<std::string>(),
        j["environment"].get<std::string>(),
        j["base_url"].get<std::string>(),
        j["timeout_ms"].get<int>(),
        j["max_retries"].get<int>()
    };
}

nlohmann::json RiskConfig::to_json() const {
    return nlohmann::json{
        {"max_daily_drawdown", max_daily_drawdown},
        {"max_position_size", max_position_size},
        {"kelly_fraction", kelly_fraction},
        {"correlation_limit", correlation_limit},
        {"var_confidence_level", var_confidence_level},
        {"var_lookback_days", var_lookback_days}
    };
}

RiskConfig RiskConfig::from_json(const nlohmann::json& j) {
    return RiskConfig{
        j["max_daily_drawdown"].get<double>(),
        j["max_position_size"].get<double>(),
        j["kelly_fraction"].get<double>(),
        j["correlation_limit"].get<double>(),
        j["var_confidence_level"].get<double>(),
        j["var_lookback_days"].get<int>()
    };
}

} // namespace forex_bot