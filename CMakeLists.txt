cmake_minimum_required(VERSION 3.17)
project(SpMMDemo)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

set(DISTGRAPH_ROOT $ENV{DISTGRAPH_ROOT})

include_directories(
    $ENV{COMBLAS_ROOT}/install/include
    ${DISTGRAPH_ROOT}/cpp
    ${MPI_CXX_INCLUDE_PATH}
    $ENV{MKLROOT}/include
)

link_directories(
    $ENV{COMBLAS_ROOT}/install/lib
    ${DISTGRAPH_ROOT}/build
    $ENV{MKLROOT}/lib/intel64
)

add_executable(spmm_demo spmm_demo.cpp)

target_link_libraries(spmm_demo
    disembed_lib
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    CombBLAS
    mkl_intel_lp64
    mkl_core
    mkl_sequential
)
