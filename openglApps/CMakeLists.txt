cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXE_LINKER_FLAGS_INIT )

include_directories("${CMAKE_SOURCE_DIR}/openglApps/")

set(SOURCE_FILES_marching_cubes_vis marching_cubes_vis.cpp)
add_executable(marching_cubes_vis ${SOURCE_FILES_marching_cubes_vis})
add_dependencies(marching_cubes_vis glad learnSPH)
target_link_libraries(marching_cubes_vis glad glfw3 rt m dl X11 pthread xcb Xau Xdmcp learnSPH)

