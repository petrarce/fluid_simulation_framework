cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)

include_directories("${CMAKE_SOURCE_DIR}/CompactNSearch/include" "${CMAKE_SOURCE_DIR}/cereal/include")

set(SOURCE_FILES_test_time_integrator test_time_integrator.cpp)

add_executable(test_time_integrator ${SOURCE_FILES_test_time_integrator})
add_dependencies(test_time_integrator learnSPH CompactNSearch)
target_link_libraries(test_time_integrator learnSPH CompactNSearch)

set(SOURCE_FILES_test_marching_cubes test_marching_cubes.cpp)

add_executable(test_marching_cubes ${SOURCE_FILES_test_marching_cubes})
add_dependencies(test_marching_cubes learnSPH)
target_link_libraries(test_marching_cubes learnSPH)

set(SOURCE_FILES_test_cereal test_cereal.cpp)

add_executable(test_cereal ${SOURCE_FILES_test_cereal})
add_dependencies(test_cereal learnSPH)
target_link_libraries(test_cereal learnSPH)

set(SOURCE_FILES_test_surface_reconstruction test_surface_reconstruction.cpp)

add_executable(test_surface_reconstruction ${SOURCE_FILES_test_surface_reconstruction})
add_dependencies(test_surface_reconstruction learnSPH)
target_link_libraries(test_surface_reconstruction learnSPH)
