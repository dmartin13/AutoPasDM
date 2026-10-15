# md-flexible adapted as a bundled AutoPasDM example.
# AutoPas and AutoPasDM are configured by the repository root.

list(APPEND CMAKE_MODULE_PATH "${AUTOPAS_SOURCE_DIR}/cmake/modules")

include(autopas_yaml-cpp)
include(autopas_all)

option(MD_FLEXIBLE_USE_MPI "Enable MPI support for the AutoPasDM md-flexible example." ON)
option(MD_FLEXIBLE_FUNCTOR_AUTOVEC "Compile AutoVec Functor." ON)
option(MD_FLEXIBLE_FUNCTOR_AVX "If instruction set is available, compile AVX Functor." OFF)
option(MD_FLEXIBLE_FUNCTOR_SVE "If instruction set is available, compile SVE Functor." OFF)
option(MD_FLEXIBLE_FUNCTOR_HWY "Compile Highway Functor." ON)
option(MD_FLEXIBLE_FUNCTOR_ATM_AUTOVEC "Compile auto vectorized Axilrod-Teller-Muto Functor." OFF)
option(MD_FLEXIBLE_CALC_GLOBALS "Compile md-flexible functors with global calculations." OFF)

file(
    GLOB_RECURSE MDFlex_SRC
    "src/*.cpp"
    "src/*.h"
)

add_executable(md-flexible ${MDFlex_SRC})
target_include_directories(md-flexible PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
    md-flexible
    PUBLIC
        autopas
        autopasdm
        molecularDynamicsLibrary
        yaml-cpp
        ${ALL_LIB}
        $<$<BOOL:${MD_FLEXIBLE_USE_MPI}>:MPI::MPI_CXX>
)

target_compile_definitions(
    md-flexible
    PUBLIC
        $<$<BOOL:${MD_FLEXIBLE_FUNCTOR_AUTOVEC}>:MD_FLEXIBLE_FUNCTOR_AUTOVEC>
        $<$<BOOL:${MD_FLEXIBLE_FUNCTOR_AVX}>:MD_FLEXIBLE_FUNCTOR_AVX>
        $<$<BOOL:${MD_FLEXIBLE_FUNCTOR_SVE}>:MD_FLEXIBLE_FUNCTOR_SVE>
        $<$<BOOL:${MD_FLEXIBLE_FUNCTOR_HWY}>:MD_FLEXIBLE_FUNCTOR_HWY>
        $<$<BOOL:${MD_FLEXIBLE_FUNCTOR_ATM_AUTOVEC}>:MD_FLEXIBLE_FUNCTOR_ATM_AUTOVEC>
        $<$<BOOL:${MD_FLEXIBLE_CALC_GLOBALS}>:MD_FLEXIBLE_CALC_GLOBALS>
)

# Copy input and helper files next to the executable as in the original
# md-flexible build.
file(
    GLOB_RECURSE MDFLEX_AUX_FILES
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "*.sh"
    "*.gp"
    "*.py"
    "*.yaml"
    "*.rule"
    "*.frule"
)

foreach(aux_file ${MDFLEX_AUX_FILES})
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/${aux_file}"
        "${CMAKE_CURRENT_BINARY_DIR}/${aux_file}"
        COPYONLY
    )
endforeach()

# Small integration example: md-flexible's MoleculeLJ + md-flexible's ordinary
# AutoPas LJFunctor running through AutoPasDM. No functor adapter.
add_executable(autopasdm-lj-example examples/AutoPasDMLJExample.cpp)
target_include_directories(autopasdm-lj-example PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
    autopasdm-lj-example
    PRIVATE
        autopasdm
        molecularDynamicsLibrary
)
