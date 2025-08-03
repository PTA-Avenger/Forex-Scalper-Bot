#include <gtest/gtest.h>
#include <gmock/gmock.h>

int main(int argc, char** argv) {
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Initialize Google Mock
    ::testing::InitGoogleMock(&argc, argv);
    
    // Run all tests
    return RUN_ALL_TESTS();
}