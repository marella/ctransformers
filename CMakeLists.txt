cmake_minimum_required(VERSION 3.17)
project(ctransformers C CXX)

set(CT_INSTRUCTIONS "avx2" CACHE STRING "avx2 | avx | basic")
message(STATUS "CT_INSTRUCTIONS: ${CT_INSTRUCTIONS}")

set(BUILD_SHARED_LIBS ON)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<0:>)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/$<0:>)

# Compile Flags

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
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

# Library

add_library(
    ctransformers SHARED
    models/llm.cc
    models/ggml/ggml.c
)

target_include_directories(ctransformers PRIVATE models)
target_link_libraries(ctransformers PRIVATE Threads::Threads)
set_target_properties(ctransformers PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

# scikit-build

install(
    TARGETS ctransformers
    LIBRARY DESTINATION ctransformers/lib/local
    RUNTIME DESTINATION ctransformers/lib/local
)