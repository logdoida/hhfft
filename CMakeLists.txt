# This Cmake file is intended for building HHFFT library
cmake_minimum_required(VERSION 2.8)
project(hhfft CXX)

set(HHFFT_VERSION_MAJOR 0)
set(HHFFT_VERSION_MINOR 1)

set(SRC_LIST ./src/architecture.cpp ./src/hhfft_base.cpp ./src/hhfft_plain_d.cpp ./src/hhfft.cpp)

# All files are compiled with atleast these flags
add_compile_options(-std=c++11 -DNDEBUG -O3 -Wall)

# If it is possible to use -mavx or -mavx512f flags use them
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-mavx COMPILER_SUPPORTS_AVX)
check_cxx_compiler_flag(-mavx512f COMPILER_SUPPORTS_AVX512F)

# If compiler supports AVX, use it
if(COMPILER_SUPPORTS_AVX)
    set(SRC_LIST ${SRC_LIST} ./src/hhfft_plain_avx_d.cpp)
    set_source_files_properties(./src/hhfft_plain_avx_d.cpp PROPERTIES COMPILE_FLAGS -mavx)
#    set(SRC_LIST ${SRC_LIST} ./src/hhfft_avx_d.cpp)
#    set_source_files_properties(./src/hhfft_avx_d.cpp PROPERTIES COMPILE_FLAGS -mavx)
    add_definitions(-DHHFFT_COMPILED_WITH_AVX=1)
endif()

# If compiler supports AVX512F, use it
if(COMPILER_SUPPORTS_AVX512F)
    set(SRC_LIST ${SRC_LIST} ./src/hhfft_plain_avx512f_d.cpp)
    set_source_files_properties(./src/hhfft_plain_avx512f_d.cpp PROPERTIES COMPILE_FLAGS -mavx512f)
#   set(SRC_LIST ${SRC_LIST} ./src/hhfft_avx512f_d.cpp)
#   set_source_files_properties(./src/hhfft_avx512f_d.cpp PROPERTIES COMPILE_FLAGS -mavx512f)
    add_definitions(-DHHFFT_COMPILED_WITH_AVX512F=1)
endif()

add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

# These linker flags are needed (sometimes?) to fix a strange bug
target_link_libraries(${PROJECT_NAME} -lgcc_s -lgcc)

# Test programs
include_directories(./src)
add_executable(hhfft_test1 ./test/test1.cpp)
target_link_libraries(hhfft_test1 ${PROJECT_NAME})

add_executable(hhfft_test2 ./test/test2.cpp)
target_link_libraries(hhfft_test2 ${PROJECT_NAME})

#Instllation rules
install (TARGETS hhfft DESTINATION lib)
install (FILES ./src/hhfft.h ./src/hhfft_base.h DESTINATION include)
