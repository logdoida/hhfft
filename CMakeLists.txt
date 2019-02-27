# This Cmake file is intended for building HHFFT library
cmake_minimum_required(VERSION 2.8)
project(hhfft CXX)

set(HHFFT_VERSION_MAJOR 0)
set(HHFFT_VERSION_MINOR 2)

set(SRC_LIST
./src/architecture.cpp
./src/aligned_arrays.cpp
./src/utilities.cpp
./src/hhfft_1d_d.cpp
./src/1d_complex/hhfft_1d_complex_d.cpp
./src/1d_complex/hhfft_1d_complex_plain_d.cpp
./src/1d_complex/hhfft_1d_complex_sse2_d.cpp
./src/1d_complex/hhfft_1d_complex_f.cpp
./src/1d_complex/hhfft_1d_complex_plain_f.cpp
./src/1d_complex/hhfft_1d_complex_sse2_f.cpp
./src/hhfft_2d_d.cpp
./src/2d_complex/hhfft_2d_complex_d.cpp
./src/2d_complex/hhfft_2d_complex_plain_d.cpp
./src/2d_complex/hhfft_2d_complex_sse2_d.cpp
./src/2d_complex/hhfft_2d_complex_f.cpp
./src/2d_complex/hhfft_2d_complex_plain_f.cpp
./src/2d_complex/hhfft_2d_complex_sse2_f.cpp
./src/hhfft_1d_real_d.cpp
./src/1d_real/hhfft_1d_real_setter_d.cpp
./src/1d_real/hhfft_1d_real_plain_d.cpp
./src/1d_real/hhfft_1d_real_sse2_d.cpp
./src/1d_real/hhfft_1d_real_setter_f.cpp
./src/1d_real/hhfft_1d_real_plain_f.cpp
./src/1d_real/hhfft_1d_real_sse2_f.cpp
./src/hhfft_2d_real_d.cpp
./src/2d_real/hhfft_2d_real_setter_d.cpp
./src/2d_real/hhfft_2d_real_plain_d.cpp
./src/2d_real/hhfft_2d_real_sse2_d.cpp
./src/2d_real/hhfft_2d_real_setter_f.cpp
./src/2d_real/hhfft_2d_real_plain_f.cpp
./src/2d_real/hhfft_2d_real_sse2_f.cpp
./src/raders/raders_d.cpp
)

# All files are compiled with atleast these flags
add_compile_options(-std=c++11 -DNDEBUG -O3 -Wall)

# Check if it is possible to use -mavx or -mavx512f
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavx COMPILER_SUPPORTS_AVX)
check_cxx_compiler_flag(-mavx512f COMPILER_SUPPORTS_AVX512F)

# If compiler supports AVX, use it
if(COMPILER_SUPPORTS_AVX)
    set(SRC_LIST ${SRC_LIST} ./src/1d_complex/hhfft_1d_complex_avx_d.cpp)
    set_source_files_properties(./src/1d_complex/hhfft_1d_complex_avx_d.cpp PROPERTIES COMPILE_FLAGS -mavx)
    set(SRC_LIST ${SRC_LIST} ./src/2d_complex/hhfft_2d_complex_avx_d.cpp)
    set_source_files_properties(./src/2d_complex/hhfft_2d_complex_avx_d.cpp PROPERTIES COMPILE_FLAGS -mavx)
    set(SRC_LIST ${SRC_LIST} ./src/1d_real/hhfft_1d_real_avx_d.cpp)
    set_source_files_properties(./src/1d_real/hhfft_1d_real_avx_d.cpp PROPERTIES COMPILE_FLAGS -mavx)
    set(SRC_LIST ${SRC_LIST} ./src/2d_real/hhfft_2d_real_avx_d.cpp)
    set_source_files_properties(./src/2d_real/hhfft_2d_real_avx_d.cpp PROPERTIES COMPILE_FLAGS -mavx)
    set(SRC_LIST ${SRC_LIST} ./src/1d_complex/hhfft_1d_complex_avx_f.cpp)
    set_source_files_properties(./src/1d_complex/hhfft_1d_complex_avx_f.cpp PROPERTIES COMPILE_FLAGS -mavx)
    set(SRC_LIST ${SRC_LIST} ./src/2d_complex/hhfft_2d_complex_avx_f.cpp)
    set_source_files_properties(./src/2d_complex/hhfft_2d_complex_avx_f.cpp PROPERTIES COMPILE_FLAGS -mavx)
    set(SRC_LIST ${SRC_LIST} ./src/1d_real/hhfft_1d_real_avx_f.cpp)
    set_source_files_properties(./src/1d_real/hhfft_1d_real_avx_f.cpp PROPERTIES COMPILE_FLAGS -mavx)
    set(SRC_LIST ${SRC_LIST} ./src/2d_real/hhfft_2d_real_avx_f.cpp)
    set_source_files_properties(./src/2d_real/hhfft_2d_real_avx_f.cpp PROPERTIES COMPILE_FLAGS -mavx)
    add_definitions(-DHHFFT_COMPILED_WITH_AVX=1)
endif()

# If compiler supports AVX512F, use it
# TODO AVX512 not supported yet
#if(COMPILER_SUPPORTS_AVX512F)
#    add_definitions(-DHHFFT_COMPILED_WITH_AVX512F=1)
#endif()

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

# These linker flags are needed (sometimes?) to fix a strange bug
target_link_libraries(${PROJECT_NAME} -lgcc_s -lgcc)

# Test programs
include_directories(./src)
add_executable(hhfft_test1 ./test/test1.cpp)
target_link_libraries(hhfft_test1 ${PROJECT_NAME})

add_executable(hhfft_test2 ./test/test2.cpp)
target_link_libraries(hhfft_test2 ${PROJECT_NAME})

#Installation rules
install (TARGETS hhfft DESTINATION lib)
install (FILES
./src/hhfft_1d_d.h
./src/hhfft_1d_real_d.h
./src/hhfft_2d_d.h
./src/hhfft_2d_real_d.h
./src/architecture.h
./src/step_info.h
./src/aligned_arrays.h
./src/raders_d.h
DESTINATION include)
