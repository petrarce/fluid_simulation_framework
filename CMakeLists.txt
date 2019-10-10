cmake_minimum_required(VERSION 3.1)
project(learnSPH)
set(CMAKE_CXX_STANDARD 17)

if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") # Parallel compilation in Visual Studio
endif()

# Modules
add_subdirectory(learnSPH)
add_subdirectory(tests)
add_subdirectory(app)

# Copy res folder
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/res)
