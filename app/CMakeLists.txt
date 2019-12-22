cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)

include_directories("${CMAKE_SOURCE_DIR}/CompactNSearch/include" "${CMAKE_SOURCE_DIR}/cereal/include")

set(SOURCE_FILES_test_sim_EXP test_sim_EXP.cpp)

add_executable(test_sim_EXP ${SOURCE_FILES_test_sim_EXP})
add_dependencies(test_sim_EXP learnSPH CompactNSearch)
target_link_libraries(test_sim_EXP learnSPH CompactNSearch)

set(SOURCE_FILES_test_sim_PBF test_sim_PBF.cpp)

add_executable(test_sim_PBF ${SOURCE_FILES_test_sim_PBF})
add_dependencies(test_sim_PBF learnSPH CompactNSearch)
target_link_libraries(test_sim_PBF learnSPH CompactNSearch)

set(SOURCE_FILES_test_marching_cubes test_marching_cubes.cpp)

add_executable(test_marching_cubes ${SOURCE_FILES_test_marching_cubes})
add_dependencies(test_marching_cubes learnSPH)
target_link_libraries(test_marching_cubes learnSPH)

set(SOURCE_FILES_test_recon test_recon.cpp)

add_executable(test_recon ${SOURCE_FILES_test_recon})
add_dependencies(test_recon learnSPH)
target_link_libraries(test_recon learnSPH)

set(SOURCE_FILES_test_border_cone test_border_cone.cpp)

add_executable(test_border_cone ${SOURCE_FILES_test_border_cone})
add_dependencies(test_border_cone learnSPH CompactNSearch)
target_link_libraries(test_border_cone learnSPH CompactNSearch)

set(SOURCE_FILES_test_wavefront_parser test_wavefront_parser.cpp)

add_executable(test_wavefront_parser ${SOURCE_FILES_test_wavefront_parser})
add_dependencies(test_wavefront_parser learnSPH CompactNSearch)
target_link_libraries(test_wavefront_parser learnSPH CompactNSearch)

set(SOURCE_FILES_test_sim test_sim.cpp)

add_executable(test_sim ${SOURCE_FILES_test_sim})
add_dependencies(test_sim learnSPH CompactNSearch)
target_link_libraries(test_sim learnSPH CompactNSearch boost_program_options boost_system boost_thread)
