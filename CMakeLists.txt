cmake_minimum_required(VERSION 3.18)
project(ctransformers C CXX)

set(CT_INSTRUCTIONS "avx2" CACHE STRING "avx2 | avx | basic")

option(CT_CUBLAS "Use cuBLAS" OFF)
option(CT_CUDA_FORCE_DMMV "use dmmv instead of mmvq CUDA kernels" OFF)
option(CT_CUDA_F16 "use 16 bit floats for some calculations" OFF)
set(CT_CUDA_MMQ_Y "64" CACHE STRING "y tile size for mmq CUDA kernels")
set(CT_CUDA_DMMV_X "32" CACHE STRING "x stride for dmmv CUDA kernels")
set(CT_CUDA_MMV_Y "1" CACHE STRING "y block size for mmv CUDA kernels")
set(CT_CUDA_KQUANTS_ITER "2" CACHE STRING "iters/thread per block for Q2_K/Q6_K")

option(CT_HIPBLAS "Use hipBLAS" OFF)
option(CT_METAL "Use Metal" OFF)

message(STATUS "CT_INSTRUCTIONS: ${CT_INSTRUCTIONS}")
message(STATUS "CT_CUBLAS: ${CT_CUBLAS}")
message(STATUS "CT_HIPBLAS: ${CT_HIPBLAS}")
message(STATUS "CT_METAL: ${CT_METAL}")

set(BUILD_SHARED_LIBS ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<0:>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<0:>)

# Compile Flags

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

if (NOT MSVC)
    set(c_flags
        -Wall
        -Wextra
        -Wpedantic
        -Wcast-qual
        -Wdouble-promotion
        -Wshadow
        -Wstrict-prototypes
        -Wpointer-arith
    )
    set(cxx_flags
        # TODO(marella): Add other warnings.
        # -Wall
        -Wextra
        -Wpedantic
        -Wcast-qual
        -Wno-unused-function
        -Wno-multichar
    )
endif()

add_compile_options(
    "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
    "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
)

# Architecture Flags

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (NOT MSVC)
        add_compile_options(-mcpu=native)
    endif()
else()
    message(STATUS "x86 detected")
    if (APPLE)
        # Universal binary.
        set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
    endif()

    if (MSVC)
        if (CT_INSTRUCTIONS STREQUAL "avx2")
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX2>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>)
        elseif (CT_INSTRUCTIONS STREQUAL "avx")
            add_compile_options($<$<COMPILE_LANGUAGE:C>:/arch:AVX>)
            add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
        endif()
    else()
        if (CT_INSTRUCTIONS STREQUAL "avx2")
            add_compile_options(-mfma -mavx2)
            add_compile_options(-mf16c -mavx)
        elseif (CT_INSTRUCTIONS STREQUAL "avx")
            add_compile_options(-mf16c -mavx)
        endif()
    endif()
endif()

if (CT_CUBLAS)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 52 == lowest CUDA 12 standard
        # 60 == f16 CUDA intrinsics
        # 61 == integer CUDA intrinsics
        # 70 == compute capability at which unrolling a loop in mul_mat_q kernels is faster
        if (CT_CUDA_F16)
            set(CMAKE_CUDA_ARCHITECTURES "60;61;70") # needed for f16 CUDA intrinsics
        else()
            set(CMAKE_CUDA_ARCHITECTURES "52;61;70") # lowest CUDA 12 standard + lowest for integer intrinsics
        endif()
    endif()
    message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Library

add_library(
    ctransformers SHARED
    models/llm.cc
    models/ggml/ggml.c
    models/ggml/ggml-alloc.c
    models/ggml/k_quants.c
    models/ggml/cmpnct_unicode.cpp
)

target_include_directories(ctransformers PRIVATE models)
target_link_libraries(ctransformers PRIVATE Threads::Threads)
set_target_properties(ctransformers PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(ctransformers PRIVATE GGML_USE_K_QUANTS)

if (APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")
        target_link_libraries(ctransformers PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(ctransformers PRIVATE GGML_USE_ACCELERATE)
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (CT_CUBLAS)
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        message(STATUS "cuBLAS found")
        enable_language(CUDA)

        target_sources(ctransformers PRIVATE models/ggml/ggml-cuda.cu)
        target_link_libraries(ctransformers PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)

        target_compile_definitions(ctransformers PRIVATE GGML_USE_CUBLAS)
        target_compile_definitions(ctransformers PRIVATE GGML_CUDA_MMQ_Y=${CT_CUDA_MMQ_Y})
        target_compile_definitions(ctransformers PRIVATE GGML_CUDA_DMMV_X=${CT_CUDA_DMMV_X})
        target_compile_definitions(ctransformers PRIVATE GGML_CUDA_MMV_Y=${CT_CUDA_MMV_Y})
        target_compile_definitions(ctransformers PRIVATE K_QUANTS_PER_ITERATION=${CT_CUDA_KQUANTS_ITER})
        if (CT_CUDA_FORCE_DMMV)
            target_compile_definitions(ctransformers PRIVATE GGML_CUDA_FORCE_DMMV)
        endif()
        if (CT_CUDA_F16)
            target_compile_definitions(ctransformers PRIVATE GGML_CUDA_F16)
        endif()
    else()
        message(WARNING "cuBLAS not found")
    endif()
endif()

if (CT_HIPBLAS)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)

    if (NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CC=/opt/rocm/llvm/bin/clang")
    endif()
    if (NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
        message(WARNING "Only LLVM is supported for HIP, hint: CXX=/opt/rocm/llvm/bin/clang++")
    endif()

    find_package(hip)
    find_package(hipblas)
    find_package(rocblas)

    if (${hipblas_FOUND} AND ${hip_FOUND})
        message(STATUS "HIP and hipBLAS found")
        set_source_files_properties(models/ggml/ggml-cuda.cu PROPERTIES LANGUAGE CXX)

        target_sources(ctransformers PRIVATE models/ggml/ggml-cuda.cu)
        target_link_libraries(ctransformers PRIVATE hip::device hip::host roc::rocblas roc::hipblas)

        target_compile_definitions(ctransformers PRIVATE GGML_USE_HIPBLAS)
        target_compile_definitions(ctransformers PRIVATE GGML_USE_CUBLAS)
        target_compile_definitions(ctransformers PRIVATE GGML_CUDA_MMQ_Y=${CT_CUDA_MMQ_Y})
        target_compile_definitions(ctransformers PRIVATE GGML_CUDA_DMMV_X=${CT_CUDA_DMMV_X})
        target_compile_definitions(ctransformers PRIVATE GGML_CUDA_MMV_Y=${CT_CUDA_MMV_Y})
        target_compile_definitions(ctransformers PRIVATE K_QUANTS_PER_ITERATION=${CT_CUDA_KQUANTS_ITER})
        target_compile_definitions(ctransformers PRIVATE CC_TURING=1000000000)
        if (CT_CUDA_FORCE_DMMV)
            target_compile_definitions(ctransformers PRIVATE GGML_CUDA_FORCE_DMMV)
        endif()
    else()
        message(WARNING "hipBLAS or HIP not found. Try setting CMAKE_PREFIX_PATH=/opt/rocm")
    endif()
endif()

if (CT_METAL)
    find_library(FOUNDATION_LIBRARY         Foundation              REQUIRED)
    find_library(METAL_FRAMEWORK            Metal                   REQUIRED)
    find_library(METALKIT_FRAMEWORK         MetalKit                REQUIRED)
    find_library(METALPERFORMANCE_FRAMEWORK MetalPerformanceShaders REQUIRED)

    target_sources(ctransformers PRIVATE models/ggml/ggml-metal.m)
    target_link_libraries(
        ctransformers PRIVATE
        ${FOUNDATION_LIBRARY}
        ${METAL_FRAMEWORK}
        ${METALKIT_FRAMEWORK}
        ${METALPERFORMANCE_FRAMEWORK}
    )

    target_compile_definitions(ctransformers PRIVATE GGML_USE_METAL)
    target_compile_definitions(ctransformers PRIVATE GGML_METAL_NDEBUG)

    set_target_properties(ctransformers PROPERTIES RESOURCE "${CMAKE_CURRENT_SOURCE_DIR}/models/ggml/ggml-metal.metal")
    configure_file(models/ggml/ggml-metal.metal lib/ggml-metal.metal COPYONLY)
endif()

# scikit-build

install(
    TARGETS ctransformers
    LIBRARY DESTINATION ctransformers/lib/local
    RUNTIME DESTINATION ctransformers/lib/local
    RESOURCE DESTINATION ctransformers/lib/local
)
