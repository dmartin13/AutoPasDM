cmake_minimum_required(VERSION 3.18)

project(distributed_autopas LANGUAGES C CXX)

# MPI is the communication backend of the current prototype. Finding it at the
# project root makes MPI::MPI_CXX visible to all bundled examples as well.
find_package(MPI REQUIRED COMPONENTS CXX)

# Build examples by default for a standalone DistributedAutoPas checkout, but
# not when the project is embedded into another superbuild.
set(_dap_build_examples_default OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_dap_build_examples_default ON)
endif()
option(
    DISTRIBUTED_AUTOPAS_BUILD_EXAMPLES
    "Build DistributedAutoPas examples"
    ${_dap_build_examples_default}
)

# Build tests by default for a standalone checkout, but keep them disabled when
# DistributedAutoPas is embedded into another project unless explicitly requested.
set(_dap_build_tests_default OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(_dap_build_tests_default ON)
endif()
option(
    DISTRIBUTED_AUTOPAS_BUILD_TESTS
    "Build DistributedAutoPas tests"
    ${_dap_build_tests_default}
)

# Optional compile-time performance counters. Both options are disabled
# by default so production builds contain neither clock reads nor timer storage.
option(
    DISTRIBUTED_AUTOPAS_ENABLE_TIMERS
    "Enable coarse DistributedAutoPas performance timers"
    OFF
)
option(
    DISTRIBUTED_AUTOPAS_ENABLE_DETAILED_TIMERS
    "Enable detailed communication timers and counters (implies coarse timers)"
    OFF
)

set(_dap_enable_timers ${DISTRIBUTED_AUTOPAS_ENABLE_TIMERS})
if(DISTRIBUTED_AUTOPAS_ENABLE_DETAILED_TIMERS)
    set(_dap_enable_timers ON)
endif()

# The bundled md-flexible example currently supports the single-site particle
# model only. This value has to be known while AutoPas' applicationLibrary is
# configured.
if(DISTRIBUTED_AUTOPAS_BUILD_EXAMPLES)
    set(MD_FLEXIBLE_MODE "SINGLESITE" CACHE STRING "md-flexible particle mode")
    if(NOT MD_FLEXIBLE_MODE STREQUAL "SINGLESITE")
        message(FATAL_ERROR
            "The bundled DistributedAutoPas md-flexible example currently supports MD_FLEXIBLE_MODE=SINGLESITE only."
        )
    endif()
endif()

# DistributedAutoPas can either be built standalone or as part of a larger
# CMake build that has already configured AutoPas.
if(NOT TARGET autopas)
    get_filename_component(_dap_workspace_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
    set(
        AUTOPAS_SOURCE_DIR
        "${_dap_workspace_dir}/AutoPas"
        CACHE PATH
        "Path to the AutoPas source directory"
    )

    if(NOT EXISTS "${AUTOPAS_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "AutoPas was not found at '${AUTOPAS_SOURCE_DIR}'. "
            "Please set AUTOPAS_SOURCE_DIR to your AutoPas checkout."
        )
    endif()

    # DistributedAutoPas does not need AutoPas' own examples/tests. Energy
    # measurements are disabled for the prototype so PMT/RAPL does not require
    # access to energy_uj on development systems.
    set(AUTOPAS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(AUTOPAS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(AUTOPAS_ENABLE_ENERGY_MEASUREMENTS OFF CACHE BOOL "" FORCE)

    add_subdirectory(
        "${AUTOPAS_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/autopas"
    )
endif()

add_library(
    distributed_autopas
    src/DomainDecomposition.cpp
    src/Runtime.cpp
)

target_include_directories(
    distributed_autopas
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(
    distributed_autopas
    PUBLIC
        autopas
        MPI::MPI_CXX
)

target_compile_features(distributed_autopas PUBLIC cxx_std_20)

if(_dap_enable_timers)
    target_compile_definitions(distributed_autopas PUBLIC DAP_ENABLE_TIMERS)
endif()
if(DISTRIBUTED_AUTOPAS_ENABLE_DETAILED_TIMERS)
    target_compile_definitions(distributed_autopas PUBLIC DAP_ENABLE_DETAILED_TIMERS)
endif()

if(DISTRIBUTED_AUTOPAS_BUILD_TESTS)
    enable_testing()

    # Reuse a GoogleTest target provided by a parent project when available.
    # Otherwise prefer an installed package and finally fetch a pinned release.
    if(TARGET gtest AND NOT TARGET GTest::gtest)
        add_library(GTest::gtest ALIAS gtest)
    endif()
    if(TARGET gtest_main AND NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()

    if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
        find_package(GTest CONFIG QUIET)
    endif()

    if(NOT TARGET GTest::gtest OR NOT TARGET GTest::gtest_main)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
        )
        FetchContent_MakeAvailable(googletest)
    endif()

    add_subdirectory(tests)
endif()

if(DISTRIBUTED_AUTOPAS_BUILD_EXAMPLES)
    if(NOT DEFINED AUTOPAS_SOURCE_DIR OR NOT EXISTS "${AUTOPAS_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "Building the bundled examples requires AUTOPAS_SOURCE_DIR to point to the AutoPas source tree."
        )
    endif()
    add_subdirectory(examples)
endif()

find_program(CLANG_FORMAT_EXE NAMES clang-format-14 clang-format)

if(CLANG_FORMAT_EXE)
    file(
        GLOB_RECURSE CLANG_FORMAT_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
    )

    # Use a project-specific target name so this is safe when AutoPas and
    # DistributedAutoPas live in the same superbuild.
    add_custom_target(
        distributed_autopas_clangformat
        COMMAND ${CLANG_FORMAT_EXE} -i ${CLANG_FORMAT_FILES}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Running clang-format for DistributedAutoPas"
    )
else()
    message(STATUS "clang-format-14 not found; formatting target is disabled")
endif()
