cmake_minimum_required(VERSION 3.1)
project(learnSPH)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_DOUBLE -fopenmp")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEBUG")
endif()

if (CMAKE_BUILD_TYPE MATCHES "Release")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPROFILE")
endif()


if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP") # Parallel compilation in Visual Studio
endif()

#remove nodebug for assertions to work out
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}" )
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL}" )
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" )


include_directories("${CMAKE_SOURCE_DIR}/include/"
					"${CMAKE_SOURCE_DIR}/extern/Eigen/"
					"${CMAKE_SOURCE_DIR}"
					"${CMAKE_SOURCE_DIR}/cereal/include/"
)


# Modules
add_subdirectory(learnSPH)
add_subdirectory(app)
add_subdirectory(CompactNSearch)

# Copy res folder
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/res)

