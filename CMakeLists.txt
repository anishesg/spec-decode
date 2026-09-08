cmake_minimum_required(VERSION 3.20)
project(spec_decode LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Require sm_80+ (Ampere and newer)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -DDEBUG_BOUNDS")

# Static library with all device-side logic
add_library(spec_decode_lib STATIC
    src/reference.cu
    src/fused_verify.cu
    src/tree_verify.cu
)
target_include_directories(spec_decode_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)
set_target_properties(spec_decode_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Test executables
enable_testing()

add_executable(test_logsumexp tests/test_logsumexp.cu)
target_link_libraries(test_logsumexp spec_decode_lib CUDA::cudart)
target_include_directories(test_logsumexp PRIVATE src)
set_target_properties(test_logsumexp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_logsumexp COMMAND test_logsumexp)

add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness spec_decode_lib CUDA::cudart)
target_include_directories(test_correctness PRIVATE src)
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME test_correctness COMMAND test_correctness)

# Benchmark executable
add_executable(bench_latency benchmarks/bench_latency.cu)
target_link_libraries(bench_latency spec_decode_lib CUDA::cudart)
target_include_directories(bench_latency PRIVATE src)
set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
