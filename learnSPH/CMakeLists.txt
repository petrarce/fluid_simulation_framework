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
	core/kernel.cpp
	core/kernel.h
	core/particle_sampler.cpp
	core/particle_sampler.h
	core/storage.h
	core/vtk_writer.cpp
	core/vtk_writer.h
	simulation/solver.cpp
	simulation/solver.h
	surf_reconstr/marching_cubes.cpp
	surf_reconstr/marching_cubes.h
	surf_reconstr/look_up_tables.hpp
	${CMAKE_SOURCE_DIR}/include/types.hpp
)

# Create a library version so it can be linked with other modules
add_library(learnSPH ${SOURCE_FILES})
target_include_directories(learnSPH PUBLIC ${EIGEN3_INCLUDE_DIR})
include_directories("${CMAKE_SOURCE_DIR}/learnSPH/"
					"${CMAKE_SOURCE_DIR}/CompactNSearch/include/")
add_dependencies(learnSPH CompactNSearch)
target_link_libraries(learnSPH CompactNSearch)

