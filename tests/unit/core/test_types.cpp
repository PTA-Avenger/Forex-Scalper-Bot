#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "core/types.h"
#include <chrono>

using namespace forex_bot;

class TypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup common test data
    }
    
    void TearDown() override {
        // Cleanup
    }
};

// Test Tick structure
TEST_F(TypesTest, TickConstruction) {
    auto now = std::chrono::system_clock::now();
    Tick tick("EUR/USD", 1.0950, 1.0952, now);
    
    EXPECT_EQ(tick.symbol, "EUR/USD");
    EXPECT_DOUBLE_EQ(tick.bid, 1.0950);
    EXPECT_DOUBLE_EQ(tick.ask, 1.0952);
    EXPECT_EQ(tick.timestamp, now);
    EXPECT_DOUBLE_EQ(tick.GetSpread(), 0.0002);
    EXPECT_DOUBLE_EQ(tick.GetMidPrice(), 1.0951);
}

TEST_F(TypesTest, TickValidation) {
    auto now = std::chrono::system_clock::now();
    
    // Valid tick
    Tick valid_tick("EUR/USD", 1.0950, 1.0952, now);
    EXPECT_TRUE(valid_tick.IsValid());
    
    // Invalid tick - bid > ask
    Tick invalid_tick("EUR/USD", 1.0952, 1.0950, now);
    EXPECT_FALSE(invalid_tick.IsValid());
    
    // Invalid tick - negative prices
    Tick negative_tick("EUR/USD", -1.0950, 1.0952, now);
    EXPECT_FALSE(negative_tick.IsValid());
}

// Test OHLCV Bar structure
TEST_F(TypesTest, OHLCVBarConstruction) {
    auto start_time = std::chrono::system_clock::now();
    auto end_time = start_time + std::chrono::minutes(1);
    
    OHLCVBar bar("EUR/USD", 1.0950, 1.0955, 1.0948, 1.0953, 1000, start_time, end_time);
    
    EXPECT_EQ(bar.symbol, "EUR/USD");
    EXPECT_DOUBLE_EQ(bar.open, 1.0950);
    EXPECT_DOUBLE_EQ(bar.high, 1.0955);
    EXPECT_DOUBLE_EQ(bar.low, 1.0948);
    EXPECT_DOUBLE_EQ(bar.close, 1.0953);
    EXPECT_EQ(bar.volume, 1000);
    EXPECT_EQ(bar.start_time, start_time);
    EXPECT_EQ(bar.end_time, end_time);
}

TEST_F(TypesTest, OHLCVBarValidation) {
    auto start_time = std::chrono::system_clock::now();
    auto end_time = start_time + std::chrono::minutes(1);
    
    // Valid bar
    OHLCVBar valid_bar("EUR/USD", 1.0950, 1.0955, 1.0948, 1.0953, 1000, start_time, end_time);
    EXPECT_TRUE(valid_bar.IsValid());
    
    // Invalid bar - high < low
    OHLCVBar invalid_high_low("EUR/USD", 1.0950, 1.0948, 1.0955, 1.0953, 1000, start_time, end_time);
    EXPECT_FALSE(invalid_high_low.IsValid());
    
    // Invalid bar - high < open
    OHLCVBar invalid_high_open("EUR/USD", 1.0955, 1.0950, 1.0948, 1.0953, 1000, start_time, end_time);
    EXPECT_FALSE(invalid_high_open.IsValid());
}

// Test Order structure
TEST_F(TypesTest, OrderConstruction) {
    Order order;
    order.id = "test_order_123";
    order.symbol = "EUR/USD";
    order.type = OrderType::BUY;
    order.size = 0.1;
    order.price = 1.0950;
    order.stop_loss = 1.0930;
    order.take_profit = 1.0970;
    order.status = OrderStatus::PENDING;
    
    EXPECT_EQ(order.id, "test_order_123");
    EXPECT_EQ(order.symbol, "EUR/USD");
    EXPECT_EQ(order.type, OrderType::BUY);
    EXPECT_DOUBLE_EQ(order.size, 0.1);
    EXPECT_DOUBLE_EQ(order.price, 1.0950);
    EXPECT_DOUBLE_EQ(order.stop_loss, 1.0930);
    EXPECT_DOUBLE_EQ(order.take_profit, 1.0970);
    EXPECT_EQ(order.status, OrderStatus::PENDING);
}

TEST_F(TypesTest, OrderValidation) {
    Order valid_order;
    valid_order.symbol = "EUR/USD";
    valid_order.type = OrderType::BUY;
    valid_order.size = 0.1;
    valid_order.price = 1.0950;
    
    EXPECT_TRUE(valid_order.IsValid());
    
    // Invalid order - zero size
    Order invalid_size = valid_order;
    invalid_size.size = 0.0;
    EXPECT_FALSE(invalid_size.IsValid());
    
    // Invalid order - negative price
    Order invalid_price = valid_order;
    invalid_price.price = -1.0950;
    EXPECT_FALSE(invalid_price.IsValid());
}

// Test Position structure
TEST_F(TypesTest, PositionConstruction) {
    Position position;
    position.symbol = "EUR/USD";
    position.type = PositionType::LONG;
    position.size = 0.1;
    position.entry_price = 1.0950;
    position.current_price = 1.0960;
    position.unrealized_pnl = 10.0;
    
    EXPECT_EQ(position.symbol, "EUR/USD");
    EXPECT_EQ(position.type, PositionType::LONG);
    EXPECT_DOUBLE_EQ(position.size, 0.1);
    EXPECT_DOUBLE_EQ(position.entry_price, 1.0950);
    EXPECT_DOUBLE_EQ(position.current_price, 1.0960);
    EXPECT_DOUBLE_EQ(position.unrealized_pnl, 10.0);
}

// Test TradingSignal structure
TEST_F(TypesTest, TradingSignalConstruction) {
    TradingSignal signal;
    signal.symbol = "EUR/USD";
    signal.type = SignalType::BUY;
    signal.strength = 0.8;
    signal.timeframe = "1m";
    signal.source = "EMA_CROSSOVER";
    signal.timestamp = std::chrono::system_clock::now();
    
    EXPECT_EQ(signal.symbol, "EUR/USD");
    EXPECT_EQ(signal.type, SignalType::BUY);
    EXPECT_DOUBLE_EQ(signal.strength, 0.8);
    EXPECT_EQ(signal.timeframe, "1m");
    EXPECT_EQ(signal.source, "EMA_CROSSOVER");
    EXPECT_TRUE(signal.IsValid());
}

TEST_F(TypesTest, TradingSignalValidation) {
    TradingSignal valid_signal;
    valid_signal.symbol = "EUR/USD";
    valid_signal.type = SignalType::BUY;
    valid_signal.strength = 0.8;
    
    EXPECT_TRUE(valid_signal.IsValid());
    
    // Invalid signal - strength out of range
    TradingSignal invalid_strength = valid_signal;
    invalid_strength.strength = 1.5;
    EXPECT_FALSE(invalid_strength.IsValid());
    
    // Invalid signal - empty symbol
    TradingSignal invalid_symbol = valid_signal;
    invalid_symbol.symbol = "";
    EXPECT_FALSE(invalid_symbol.IsValid());
}

// Test RiskMetrics structure
TEST_F(TypesTest, RiskMetricsConstruction) {
    RiskMetrics metrics;
    metrics.total_exposure = 1000.0;
    metrics.daily_pnl = 50.0;
    metrics.drawdown = 0.05;
    metrics.var_95 = 100.0;
    metrics.sharpe_ratio = 1.5;
    
    EXPECT_DOUBLE_EQ(metrics.total_exposure, 1000.0);
    EXPECT_DOUBLE_EQ(metrics.daily_pnl, 50.0);
    EXPECT_DOUBLE_EQ(metrics.drawdown, 0.05);
    EXPECT_DOUBLE_EQ(metrics.var_95, 100.0);
    EXPECT_DOUBLE_EQ(metrics.sharpe_ratio, 1.5);
}