#pragma once

#include <string>
#include <nlohmann/json.hpp>

namespace forex_bot {

class ConfigManager {
public:
    static ConfigManager* GetInstance() {
        static ConfigManager instance;
        return &instance;
    }

    bool LoadConfig(const std::string& config_path) {
        try {
            std::ifstream config_file(config_path);
            if (!config_file.is_open()) {
                return false;
            }
            config = nlohmann::json::parse(config_file);
            return true;
        } catch (...) {
            return false;
        }
    }

    template<typename T>
    T GetValue(const std::string& key, const T& default_value = T()) const {
        try {
            return config.value(key, default_value);
        } catch (...) {
            return default_value;
        }
    }

    bool HasKey(const std::string& key) const {
        return config.contains(key);
    }

    const nlohmann::json& GetConfig() const {
        return config;
    }

private:
    ConfigManager() = default;
    ~ConfigManager() = default;
    ConfigManager(const ConfigManager&) = delete;
    ConfigManager& operator=(const ConfigManager&) = delete;

    nlohmann::json config;
};

} // namespace forex_bot
