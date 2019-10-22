cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)

# Local
set(SOURCE_FILES_learnSPH_app
    main.cpp)
# Create a standalone executable for the geo module
add_executable(learnSPH_app ${SOURCE_FILES_learnSPH_app})
add_dependencies(learnSPH_app learnSPH)
target_link_libraries(learnSPH_app learnSPH)

# Local
set(SOURCE_FILES_bf_vs_lib_NBS_compare
	compare_main.cpp)
add_executable(bf_vs_lib_NBS_compare ${SOURCE_FILES_bf_vs_lib_NBS_compare})
add_dependencies(bf_vs_lib_NBS_compare learnSPH CompactNSearch)
target_link_libraries(bf_vs_lib_NBS_compare learnSPH CompactNSearch)
include_directories("${CMAKE_SOURCE_DIR}/CompactNSearch/include/")


set(SOURCE_FILES_kernel_interpolation
	kernel_interpolation.cpp)
add_executable(kernel_interpolation ${SOURCE_FILES_kernel_interpolation})
add_dependencies(kernel_interpolation learnSPH CompactNSearch)
target_link_libraries(kernel_interpolation learnSPH CompactNSearch)
