# External dependencies
## OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

## Eigen
set( EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/Eigen )
if( NOT EIGEN3_INCLUDE_DIR )
    message( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
endif()

set(SOURCE_FILES
	vtk_writer.cpp
	${CMAKE_SOURCE_DIR}/include/vtk_writer.h
	kernel.cpp
	${CMAKE_SOURCE_DIR}/include/kernel.h
	NBsearch.cpp
	${CMAKE_SOURCE_DIR}/include/NBsearch.hpp
	${CMAKE_SOURCE_DIR}/include/types.hpp
	${CMAKE_SOURCE_DIR}/include/storage.h
	${CMAKE_SOURCE_DIR}/include/particle_sampler.h
	${CMAKE_SOURCE_DIR}/include/solver.h
	${CMAKE_SOURCE_DIR}/include/marching_cubes.h
	${CMAKE_SOURCE_DIR}/include/triangle_table.h
	${CMAKE_SOURCE_DIR}/include/marching_cubes_lut.hpp
	particle_sampler.cpp
	solver.cpp
	marching_cubes.cpp)

# Create a library version so it can be linked with other modules
add_library(learnSPH ${SOURCE_FILES})
target_include_directories(learnSPH PUBLIC ${EIGEN3_INCLUDE_DIR})
include_directories("${CMAKE_SOURCE_DIR}/learnSPH/"
					"${CMAKE_SOURCE_DIR}/CompactNSearch/include/")
add_dependencies(learnSPH CompactNSearch)
target_link_libraries(learnSPH CompactNSearch)
