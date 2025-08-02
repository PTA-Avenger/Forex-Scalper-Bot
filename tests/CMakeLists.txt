# CMakeLists.txt for tests
cmake_minimum_required(VERSION 3.16)

# Find or fetch Google Test
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Find or fetch Google Benchmark for performance tests
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
FetchContent_MakeAvailable(benchmark)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Test source files
file(GLOB_RECURSE TEST_SOURCES
    "unit/*.cpp"
    "integration/*.cpp"
    "performance/*.cpp"
)

# Core source files (exclude main.cpp)
file(GLOB_RECURSE CORE_SOURCES 
    "${CMAKE_SOURCE_DIR}/src/core/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/brokers/*.cpp" 
    "${CMAKE_SOURCE_DIR}/src/strategies/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/risk/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/backtest/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/utils/*.cpp"
)

# Unit tests executable
add_executable(unit_tests
    unit/test_main.cpp
    unit/core/test_types.cpp
    unit/core/test_market_data.cpp
    unit/core/test_order_manager.cpp
    unit/strategies/test_ema_crossover.cpp
    unit/risk/test_risk_manager.cpp
    unit/utils/test_logger.cpp
    ${CORE_SOURCES}
)

target_link_libraries(unit_tests
    gtest_main
    gmock_main
    Boost::system
    Boost::thread
    Boost::chrono
    Boost::filesystem
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${HIREDIS_LIBRARIES}
    ${PostgreSQL_LIBRARIES}
    pthread
)

# Integration tests executable
add_executable(integration_tests
    integration/test_main.cpp
    integration/test_broker_integration.cpp
    integration/test_database_integration.cpp
    integration/test_redis_integration.cpp
    integration/test_api_endpoints.cpp
    ${CORE_SOURCES}
)

target_link_libraries(integration_tests
    gtest_main
    gmock_main
    Boost::system
    Boost::thread
    Boost::chrono
    Boost::filesystem
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${HIREDIS_LIBRARIES}
    ${PostgreSQL_LIBRARIES}
    pthread
)

# Performance tests executable
add_executable(performance_tests
    performance/test_main.cpp
    performance/benchmark_order_execution.cpp
    performance/benchmark_market_data.cpp
    performance/benchmark_strategy_execution.cpp
    ${CORE_SOURCES}
)

target_link_libraries(performance_tests
    benchmark::benchmark
    Boost::system
    Boost::thread
    Boost::chrono
    Boost::filesystem
    OpenSSL::SSL
    OpenSSL::Crypto
    CURL::libcurl
    nlohmann_json::nlohmann_json
    spdlog::spdlog
    ${HIREDIS_LIBRARIES}
    ${PostgreSQL_LIBRARIES}
    pthread
)

# Add tests to CTest
enable_testing()
add_test(NAME UnitTests COMMAND unit_tests)
add_test(NAME IntegrationTests COMMAND integration_tests)
add_test(NAME PerformanceTests COMMAND performance_tests --benchmark_min_time=1)

# Set test properties
set_tests_properties(UnitTests PROPERTIES TIMEOUT 60)
set_tests_properties(IntegrationTests PROPERTIES TIMEOUT 300)
set_tests_properties(PerformanceTests PROPERTIES TIMEOUT 600)

# Coverage target (if gcov is available)
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
    
    find_program(LCOV_PATH lcov)
    find_program(GENHTML_PATH genhtml)
    
    if(LCOV_PATH AND GENHTML_PATH)
        add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E make_directory coverage
            COMMAND ${LCOV_PATH} --directory . --capture --output-file coverage/coverage.info
            COMMAND ${LCOV_PATH} --remove coverage/coverage.info '/usr/*' --output-file coverage/coverage.info
            COMMAND ${GENHTML_PATH} coverage/coverage.info --output-directory coverage/html
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Generating code coverage report"
        )
    endif()
endif()