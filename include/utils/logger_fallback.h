#pragma once

#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

namespace forex_bot {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR,
    CRITICAL
};

class FallbackLogger {
public:
    static FallbackLogger* GetInstance() {
        static FallbackLogger instance;
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
    FallbackLogger() = default;
    ~FallbackLogger() = default;
    FallbackLogger(const FallbackLogger&) = delete;
    FallbackLogger& operator=(const FallbackLogger&) = delete;

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
        
        // Simple format string replacement (basic implementation)
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

} // namespace forex_bot