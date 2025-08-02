#include "core/engine.h"
#include "utils/logger.h"
#include "utils/config_manager.h"
#include <iostream>
#include <csignal>
#include <memory>
#include <thread>
#include <chrono>

namespace forex_bot {

// Global engine instance for signal handling
std::unique_ptr<TradingEngine> g_engine;
std::atomic<bool> g_shutdown_requested{false};

/**
 * @brief Signal handler for graceful shutdown
 */
void SignalHandler(int signal) {
    std::cout << "\nReceived signal " << signal << ". Initiating graceful shutdown..." << std::endl;
    g_shutdown_requested = true;
    
    if (g_engine) {
        g_engine->Stop();
    }
}

/**
 * @brief Print application banner
 */
void PrintBanner() {
    std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║                    Forex Scalping Bot                       ║
║                High-Performance Trading System              ║
║                     Version 1.0.0                          ║
╚══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

/**
 * @brief Print usage information
 */
void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  -c, --config PATH     Configuration file path (default: config/trading_config.json)\n"
              << "  -p, --paper           Enable paper trading mode\n"
              << "  -b, --backtest PATH   Run backtest with historical data\n"
              << "  -v, --verbose         Enable verbose logging\n"
              << "  -h, --help            Show this help message\n"
              << "  --version             Show version information\n\n"
              << "Examples:\n"
              << "  " << program_name << " -c config/live.json\n"
              << "  " << program_name << " --paper -v\n"
              << "  " << program_name << " --backtest data/historical.csv\n"
              << std::endl;
}

/**
 * @brief Print version information
 */
void PrintVersion() {
    std::cout << "Forex Scalping Bot v1.0.0\n"
              << "Built with C++20, Boost, and modern trading technologies\n"
              << "Copyright (c) 2024 Forex Bot Team\n"
              << std::endl;
}

/**
 * @brief Parse command line arguments
 */
struct CommandLineArgs {
    std::string config_path = "config/trading_config.json";
    std::string backtest_data_path;
    bool paper_trading = false;
    bool verbose = false;
    bool show_help = false;
    bool show_version = false;
    bool run_backtest = false;
};

CommandLineArgs ParseArguments(int argc, char* argv[]) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            args.show_help = true;
        } else if (arg == "--version") {
            args.show_version = true;
        } else if (arg == "-v" || arg == "--verbose") {
            args.verbose = true;
        } else if (arg == "-p" || arg == "--paper") {
            args.paper_trading = true;
        } else if (arg == "-c" || arg == "--config") {
            if (i + 1 < argc) {
                args.config_path = argv[++i];
            } else {
                throw std::invalid_argument("Config path required after " + arg);
            }
        } else if (arg == "-b" || arg == "--backtest") {
            if (i + 1 < argc) {
                args.backtest_data_path = argv[++i];
                args.run_backtest = true;
            } else {
                throw std::invalid_argument("Backtest data path required after " + arg);
            }
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    
    return args;
}

/**
 * @brief Initialize logging system
 */
void InitializeLogging(bool verbose) {
    auto logger = Logger::GetInstance();
    
    if (verbose) {
        logger->SetLogLevel(LogLevel::DEBUG);
    } else {
        logger->SetLogLevel(LogLevel::INFO);
    }
    
    logger->AddConsoleAppender();
    logger->AddFileAppender("logs/forex_bot.log");
    
    LOG_INFO("Logging system initialized");
}

/**
 * @brief Run the trading engine
 */
int RunTradingEngine(const CommandLineArgs& args) {
    try {
        LOG_INFO("Starting Forex Scalping Bot...");
        LOG_INFO("Configuration file: {}", args.config_path);
        
        // Create appropriate engine based on mode
        if (args.run_backtest) {
            LOG_INFO("Running in backtest mode with data: {}", args.backtest_data_path);
            g_engine = EngineFactory::CreateBacktestingEngine(args.config_path, args.backtest_data_path);
        } else if (args.paper_trading) {
            LOG_INFO("Running in paper trading mode");
            g_engine = EngineFactory::CreatePaperTradingEngine(args.config_path);
        } else {
            LOG_INFO("Running in live trading mode");
            g_engine = EngineFactory::CreateEngine(args.config_path);
        }
        
        if (!g_engine) {
            LOG_ERROR("Failed to create trading engine");
            return 1;
        }
        
        // Initialize the engine
        if (!g_engine->Initialize()) {
            LOG_ERROR("Failed to initialize trading engine");
            return 1;
        }
        
        LOG_INFO("Trading engine initialized successfully");
        
        // Start the engine
        if (!g_engine->Start()) {
            LOG_ERROR("Failed to start trading engine");
            return 1;
        }
        
        LOG_INFO("Trading engine started successfully");
        
        // Print system status
        auto health_status = g_engine->GetHealthStatus();
        LOG_INFO("System health status: {}", health_status.dump(2));
        
        // Main loop - wait for shutdown signal
        while (!g_shutdown_requested && g_engine->IsRunning()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // Periodically log statistics
            static int stats_counter = 0;
            if (++stats_counter >= 300) { // Every 5 minutes
                auto stats = g_engine->GetStatistics();
                LOG_INFO("System statistics: {}", stats.dump(2));
                stats_counter = 0;
            }
        }
        
        LOG_INFO("Shutting down trading engine...");
        g_engine->Stop();
        
        // Wait a bit for graceful shutdown
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        LOG_INFO("Trading engine stopped successfully");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Fatal error: {}", e.what());
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        LOG_ERROR("Unknown fatal error occurred");
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return 1;
    }
}

/**
 * @brief Run backtesting mode
 */
int RunBacktest(const CommandLineArgs& args) {
    try {
        LOG_INFO("Starting backtest...");
        
        auto engine = EngineFactory::CreateBacktestingEngine(args.config_path, args.backtest_data_path);
        if (!engine) {
            LOG_ERROR("Failed to create backtesting engine");
            return 1;
        }
        
        if (!engine->Initialize()) {
            LOG_ERROR("Failed to initialize backtesting engine");
            return 1;
        }
        
        if (!engine->Start()) {
            LOG_ERROR("Failed to start backtest");
            return 1;
        }
        
        // Wait for backtest to complete
        while (engine->IsRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Get and display results
        auto performance = engine->GetPerformanceMetrics();
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "BACKTEST RESULTS" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Total Return: " << std::fixed << std::setprecision(2) 
                  << performance.total_return * 100 << "%" << std::endl;
        std::cout << "Annualized Return: " << std::fixed << std::setprecision(2) 
                  << performance.annualized_return * 100 << "%" << std::endl;
        std::cout << "Sharpe Ratio: " << std::fixed << std::setprecision(3) 
                  << performance.sharpe_ratio << std::endl;
        std::cout << "Max Drawdown: " << std::fixed << std::setprecision(2) 
                  << performance.max_drawdown * 100 << "%" << std::endl;
        std::cout << "Win Rate: " << std::fixed << std::setprecision(1) 
                  << performance.win_rate * 100 << "%" << std::endl;
        std::cout << "Total Trades: " << performance.total_trades << std::endl;
        std::cout << "Profit Factor: " << std::fixed << std::setprecision(2) 
                  << performance.profit_factor << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        LOG_INFO("Backtest completed successfully");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Backtest error: {}", e.what());
        std::cerr << "Backtest error: " << e.what() << std::endl;
        return 1;
    }
}

} // namespace forex_bot

/**
 * @brief Main application entry point
 */
int main(int argc, char* argv[]) {
    using namespace forex_bot;
    
    try {
        // Parse command line arguments
        auto args = ParseArguments(argc, argv);
        
        if (args.show_help) {
            PrintUsage(argv[0]);
            return 0;
        }
        
        if (args.show_version) {
            PrintVersion();
            return 0;
        }
        
        // Print banner
        PrintBanner();
        
        // Initialize logging
        InitializeLogging(args.verbose);
        
        // Setup signal handlers for graceful shutdown
        std::signal(SIGINT, SignalHandler);
        std::signal(SIGTERM, SignalHandler);
        
        // Run appropriate mode
        if (args.run_backtest) {
            return RunBacktest(args);
        } else {
            return RunTradingEngine(args);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        PrintUsage(argv[0]);
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}