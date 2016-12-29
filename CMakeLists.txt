cmake_minimum_required(VERSION 3.6)
project(mpi_matrix_multiplication)

set(CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

SET(CMAKE_C_COMPILER mpicc)
SET(CMAKE_CXX_COMPILER mpiCC)

set(SOURCE_FILES main.c)
add_executable(mpi_matrix_multiplication ${SOURCE_FILES})
