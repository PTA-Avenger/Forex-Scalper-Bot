#include <benchmark/benchmark.h>
#include "core/types.h"
#include "core/engine.h"
#include <chrono>
#include <vector>
#include <random>

using namespace forex_bot;

class OrderExecutionBenchmark : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) override {
        // Initialize random number generator for realistic order data
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        price_dist = std::uniform_real_distribution<double>(1.0900, 1.1000);
        size_dist = std::uniform_real_distribution<double>(0.01, 1.0);
        
        // Pre-generate test orders
        orders.reserve(1000);
        for (int i = 0; i < 1000; ++i) {
            Order order;
            order.id = "benchmark_order_" + std::to_string(i);
            order.symbol = "EUR/USD";
            order.type = (i % 2 == 0) ? OrderType::BUY : OrderType::SELL;
            order.size = size_dist(rng);
            order.price = price_dist(rng);
            order.status = OrderStatus::PENDING;
            orders.push_back(order);
        }
    }
    
    void TearDown(const ::benchmark::State& state) override {
        orders.clear();
    }

protected:
    std::mt19937 rng;
    std::uniform_real_distribution<double> price_dist;
    std::uniform_real_distribution<double> size_dist;
    std::vector<Order> orders;
};

// Benchmark order validation performance
BENCHMARK_F(OrderExecutionBenchmark, OrderValidation)(benchmark::State& state) {
    size_t order_index = 0;
    
    for (auto _ : state) {
        const auto& order = orders[order_index % orders.size()];
        bool is_valid = order.IsValid();
        benchmark::DoNotOptimize(is_valid);
        order_index++;
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Benchmark order creation performance
BENCHMARK_F(OrderExecutionBenchmark, OrderCreation)(benchmark::State& state) {
    size_t order_index = 0;
    
    for (auto _ : state) {
        Order order;
        order.id = "benchmark_order_" + std::to_string(order_index);
        order.symbol = "EUR/USD";
        order.type = OrderType::BUY;
        order.size = 0.1;
        order.price = 1.0950;
        order.timestamp = std::chrono::system_clock::now();
        
        benchmark::DoNotOptimize(order);
        order_index++;
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Benchmark order serialization (for network transmission)
BENCHMARK_F(OrderExecutionBenchmark, OrderSerialization)(benchmark::State& state) {
    size_t order_index = 0;
    
    for (auto _ : state) {
        const auto& order = orders[order_index % orders.size()];
        
        // Simulate JSON serialization
        std::string json = "{";
        json += "\"id\":\"" + order.id + "\",";
        json += "\"symbol\":\"" + order.symbol + "\",";
        json += "\"type\":" + std::to_string(static_cast<int>(order.type)) + ",";
        json += "\"size\":" + std::to_string(order.size) + ",";
        json += "\"price\":" + std::to_string(order.price);
        json += "}";
        
        benchmark::DoNotOptimize(json);
        order_index++;
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Benchmark risk calculation for orders
BENCHMARK_F(OrderExecutionBenchmark, RiskCalculation)(benchmark::State& state) {
    size_t order_index = 0;
    const double account_balance = 10000.0;
    const double max_risk_per_trade = 0.02; // 2%
    
    for (auto _ : state) {
        const auto& order = orders[order_index % orders.size()];
        
        // Calculate position size based on risk
        double risk_amount = account_balance * max_risk_per_trade;
        double stop_distance = order.price * 0.001; // 10 pips
        double position_size = risk_amount / stop_distance;
        
        benchmark::DoNotOptimize(position_size);
        order_index++;
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Benchmark order queue operations
BENCHMARK_F(OrderExecutionBenchmark, OrderQueueOperations)(benchmark::State& state) {
    std::queue<Order> order_queue;
    size_t order_index = 0;
    
    for (auto _ : state) {
        // Add order to queue
        order_queue.push(orders[order_index % orders.size()]);
        
        // Process order if queue has items
        if (!order_queue.empty()) {
            Order processed_order = order_queue.front();
            order_queue.pop();
            benchmark::DoNotOptimize(processed_order);
        }
        
        order_index++;
    }
    
    state.SetItemsProcessed(state.iterations());
}

// Benchmark concurrent order processing
BENCHMARK_F(OrderExecutionBenchmark, ConcurrentOrderProcessing)(benchmark::State& state) {
    std::atomic<size_t> processed_count{0};
    const int num_threads = state.range(0);
    
    for (auto _ : state) {
        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Launch worker threads
        for (int i = 0; i < num_threads; ++i) {
            workers.emplace_back([&, i]() {
                size_t local_index = i;
                for (int j = 0; j < 100; ++j) {
                    const auto& order = orders[local_index % orders.size()];
                    
                    // Simulate order processing
                    bool is_valid = order.IsValid();
                    if (is_valid) {
                        processed_count.fetch_add(1, std::memory_order_relaxed);
                    }
                    
                    local_index += num_threads;
                }
            });
        }
        
        // Wait for all workers to complete
        for (auto& worker : workers) {
            worker.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        state.SetIterationTime(duration / 1e6); // Convert to seconds
    }
    
    state.SetItemsProcessed(processed_count.load());
    state.UseManualTime();
}

// Register benchmarks with different thread counts
BENCHMARK_REGISTER_F(OrderExecutionBenchmark, ConcurrentOrderProcessing)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();

// Set benchmark time limits to ensure realistic performance measurement
BENCHMARK_REGISTER_F(OrderExecutionBenchmark, OrderValidation)
    ->MinTime(1.0)->Iterations(1000000);

BENCHMARK_REGISTER_F(OrderExecutionBenchmark, OrderCreation)
    ->MinTime(1.0)->Iterations(1000000);

BENCHMARK_REGISTER_F(OrderExecutionBenchmark, OrderSerialization)
    ->MinTime(1.0)->Iterations(100000);

BENCHMARK_REGISTER_F(OrderExecutionBenchmark, RiskCalculation)
    ->MinTime(1.0)->Iterations(1000000);

BENCHMARK_REGISTER_F(OrderExecutionBenchmark, OrderQueueOperations)
    ->MinTime(1.0)->Iterations(1000000);