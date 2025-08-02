cmake_minimum_required(VERSION 3.16)
project(ForexScalpingBot VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -fsanitize=address")

# Include FetchContent module for downloading dependencies
include(FetchContent)

# Find required packages
find_package(Boost REQUIRED COMPONENTS system thread chrono filesystem)
find_package(OpenSSL REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)

# Find nlohmann/json
find_package(nlohmann_json REQUIRED)

# Find spdlog - with fallback to FetchContent
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found, fetching from GitHub...")
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.12.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# Find hiredis for Redis
pkg_check_modules(HIREDIS REQUIRED hiredis)

# Find PostgreSQL
find_package(PostgreSQL REQUIRED)

# Include directories
include_directories(include)
include_directories(${PostgreSQL_INCLUDE_DIRS})

# Source files
file(GLOB_RECURSE SOURCES 
    "src/core/*.cpp"
    "src/brokers/*.cpp" 
    "src/strategies/*.cpp"
    "src/risk/*.cpp"
    "src/backtest/*.cpp"
    "src/utils/*.cpp"
)

# Create executable
add_executable(${PROJECT_NAME} src/main.cpp ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
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

# Set target include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PostgreSQL_INCLUDE_DIRS}
)

# Compiler-specific options
target_compile_options(${PROJECT_NAME} PRIVATE ${HIREDIS_CFLAGS_OTHER})

# Enable testing
enable_testing()

# Add tests subdirectory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(DIRECTORY config/ DESTINATION etc/forex-bot)