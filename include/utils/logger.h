#pragma once

// Try to use spdlog if available, otherwise fall back to standard library implementation
#ifdef SPDLOG_VERSION
    #include <spdlog/spdlog.h>
    #include <spdlog/sinks/stdout_color_sinks.h>
    #include <spdlog/sinks/basic_file_sink.h>
    #define USE_SPDLOG
#else
    // Check if spdlog headers exist
    #if __has_include(<spdlog/spdlog.h>)
        #include <spdlog/spdlog.h>
        #include <spdlog/sinks/stdout_color_sinks.h>
        #include <spdlog/sinks/basic_file_sink.h>
        #define USE_SPDLOG
    #else
        // Fall back to standard library implementation
        #include <iostream>
        #include <fstream>
        #include <chrono>
        #include <iomanip>
        #include <sstream>
        #include <mutex>
        #define USE_FALLBACK_LOGGER
    #endif
#endif

#include <memory>
#include <string>

namespace forex_bot {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL
};

#ifdef USE_SPDLOG
class Logger {
public:
    static Logger* GetInstance() {
        static Logger instance;
        return &instance;
    }

    void SetLogLevel(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG:
                spdlog::set_level(spdlog::level::debug);
                break;
            case LogLevel::INFO:
                spdlog::set_level(spdlog::level::info);
                break;
            case LogLevel::WARN:
                spdlog::set_level(spdlog::level::warn);
                break;
            case LogLevel::ERROR:
                spdlog::set_level(spdlog::level::err);
                break;
            case LogLevel::CRITICAL:
                spdlog::set_level(spdlog::level::critical);
                break;
        }
    }

    void AddConsoleAppender() {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        logger->sinks().push_back(console_sink);
    }

    void AddFileAppender(const std::string& filename) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(filename, true);
        logger->sinks().push_back(file_sink);
    }

    template<typename... Args>
    void Debug(const char* fmt, const Args&... args) {
        logger->debug(fmt, args...);
    }

    template<typename... Args>
    void Info(const char* fmt, const Args&... args) {
        logger->info(fmt, args...);
    }

    template<typename... Args>
    void Warn(const char* fmt, const Args&... args) {
        logger->warn(fmt, args...);
    }

    template<typename... Args>
    void Error(const char* fmt, const Args&... args) {
        logger->error(fmt, args...);
    }

    template<typename... Args>
    void Critical(const char* fmt, const Args&... args) {
        logger->critical(fmt, args...);
    }

private:
    Logger() {
        logger = spdlog::default_logger();
    }
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    std::shared_ptr<spdlog::logger> logger;
};

#else // USE_FALLBACK_LOGGER

class Logger {
public:
    static Logger* GetInstance() {
        static Logger instance;
        return &instance;
    }

    void SetLogLevel(LogLevel level) {
        current_level_ = level;
    }

    void AddConsoleAppender() {
        console_enabled_ = true;
    }

    void AddFileAppender(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mutex_);
        file_stream_ = std::make_unique<std::ofstream>(filename, std::ios::app);
        file_enabled_ = file_stream_->is_open();
    }

    template<typename... Args>
    void Debug(const char* fmt, const Args&... args) {
        if (current_level_ <= LogLevel::DEBUG) {
            Log("DEBUG", fmt, args...);
        }
    }

    template<typename... Args>
    void Info(const char* fmt, const Args&... args) {
        if (current_level_ <= LogLevel::INFO) {
            Log("INFO", fmt, args...);
        }
    }

    template<typename... Args>
    void Warn(const char* fmt, const Args&... args) {
        if (current_level_ <= LogLevel::WARN) {
            Log("WARN", fmt, args...);
        }
    }

    template<typename... Args>
    void Error(const char* fmt, const Args&... args) {
        if (current_level_ <= LogLevel::ERROR) {
            Log("ERROR", fmt, args...);
        }
    }

    template<typename... Args>
    void Critical(const char* fmt, const Args&... args) {
        if (current_level_ <= LogLevel::CRITICAL) {
            Log("CRITICAL", fmt, args...);
        }
    }

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    template<typename... Args>
    void Log(const char* level, const char* fmt, const Args&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        ss << " [" << level << "] ";
        
        // Simple format string replacement
        std::string message = FormatString(fmt, args...);
        ss << message;

        std::string log_line = ss.str();

        if (console_enabled_) {
            std::cout << log_line << std::endl;
        }

        if (file_enabled_ && file_stream_) {
            *file_stream_ << log_line << std::endl;
            file_stream_->flush();
        }
    }

    template<typename T>
    std::string FormatString(const char* fmt, const T& arg) {
        std::string result(fmt);
        size_t pos = result.find("{}");
        if (pos != std::string::npos) {
            std::stringstream ss;
            ss << arg;
            result.replace(pos, 2, ss.str());
        }
        return result;
    }

    template<typename T, typename... Args>
    std::string FormatString(const char* fmt, const T& first, const Args&... rest) {
        std::string result(fmt);
        size_t pos = result.find("{}");
        if (pos != std::string::npos) {
            std::stringstream ss;
            ss << first;
            result.replace(pos, 2, ss.str());
            return FormatString(result.c_str(), rest...);
        }
        return result;
    }

    std::string FormatString(const char* fmt) {
        return std::string(fmt);
    }

    LogLevel current_level_ = LogLevel::INFO;
    bool console_enabled_ = false;
    bool file_enabled_ = false;
    std::unique_ptr<std::ofstream> file_stream_;
    std::mutex mutex_;
};

#endif // USE_SPDLOG

} // namespace forex_bot

// Convenience macros
#define LOG_DEBUG(...)    forex_bot::Logger::GetInstance()->Debug(__VA_ARGS__)
#define LOG_INFO(...)     forex_bot::Logger::GetInstance()->Info(__VA_ARGS__)
#define LOG_WARN(...)     forex_bot::Logger::GetInstance()->Warn(__VA_ARGS__)
#define LOG_ERROR(...)    forex_bot::Logger::GetInstance()->Error(__VA_ARGS__)
#define LOG_CRITICAL(...) forex_bot::Logger::GetInstance()->Critical(__VA_ARGS__)
