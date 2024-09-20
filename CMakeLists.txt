cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_COMPILER "g++")
#set(CMAKE_C_COMPILER "gcc")

project(VectorSetSearch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

#add_compile_options(
#        -Wall
#        -Wextra
#        # TODO(Josh): Fix eigen error and add back -werror
#        -Wno-unused-function
#        -Wno-unused-parameter
#        -pedantic
#        -Ofast
#        -g
#        -fno-omit-frame-pointer
#)

string(ASCII 27 Esc)
set(ColourReset "${Esc}[m")
set(Green "${Esc}[32m")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
    message("${Green}Release mode${ColourReset}")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message("${Green}Clang${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -DNDEBUG -openmp -march=native -fpic -ftree-vectorize ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -DNDEBUG -openmp -march=native -fpic -ftree-vectorize ${OpenMP_CXX_FLAGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message("${Green}GNU${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -lrt -DNDEBUG -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -lrt -DNDEBUG -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0 ${OpenMP_CXX_FLAGS})
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message("${Green}MSVC${ColourReset}")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize ${OpenMP_CXX_FLAGS}")
        add_definitions(-O3 -lrt -DNDEBUG -openmp -march=native -fpic -w -fopenmp -ftree-vectorize ${OpenMP_CXX_FLAGS})
    else ()
        message("${Green}Unknown compiler${ColourReset}")
    endif ()
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    #    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -fsanitize=address -lasan ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native ${OpenMP_CXX_FLAGS}")
    #    add_definitions(-DNDEBUG -march=native)
    message("${Green}Debug mode${ColourReset}")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native ${OpenMP_CXX_FLAGS}")
    message("${Green}Unknown mode${ColourReset}")
endif ()

find_package(OpenMP REQUIRED)

#pybind11_add_module(IPGraphSearchVQ src/app/IPGraphSearchVQ.cpp)
#target_link_libraries(IPGraphSearchVQ PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog ${OpenBLAS_LIB})
#target_include_directories(IPGraphSearchVQ PRIVATE ${OpenBLAS_INCLUDE_DIR})

#add_executable(t1 test.cpp)
#target_link_libraries(t1 PRIVATE spdlog::spdlog)

add_executable(version src/version.cpp)
target_link_libraries(version PRIVATE OpenMP::OpenMP_CXX)

add_executable(add src/add.cpp)
target_link_libraries(add PRIVATE OpenMP::OpenMP_CXX)

