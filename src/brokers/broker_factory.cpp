#include "brokers/broker_interface.h"
#include "brokers/mt5_broker.h"
#include <stdexcept>

namespace forex_bot {

std::unique_ptr<BrokerInterface> CreateBroker(const BrokerConfig& config) {
    if (config.type == "mt5") {
        return std::make_unique<MT5Broker>(config);
    } else if (config.type == "oanda") {
        // TODO: Implement OANDA broker
        throw std::runtime_error("OANDA broker not implemented yet");
    } else if (config.type == "mt4") {
        // TODO: Implement MT4 broker
        throw std::runtime_error("MT4 broker not implemented yet");
    } else {
        throw std::runtime_error("Unknown broker type: " + config.type);
    }
}

} // namespace forex_bot