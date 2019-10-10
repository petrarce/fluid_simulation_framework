cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)

# Local
set(SOURCE_FILES
    test_main.cpp
	sample_tests.cpp)
    
# Create a standalone executable for the geo module
add_executable(learnSPH_tests ${SOURCE_FILES})
add_dependencies(learnSPH_tests learnSPH)
target_link_libraries(learnSPH_tests learnSPH)
