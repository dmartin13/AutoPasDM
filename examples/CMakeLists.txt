add_executable(minimal_mpi minimal_mpi.cpp)
target_link_libraries(minimal_mpi PRIVATE autopasdm)

add_subdirectory(lj)

option(
    AUTOPASDM_BUILD_MDFLEX_EXAMPLE
    "Build the md-flexible AutoPasDM example"
    ON
)

if(AUTOPASDM_BUILD_MDFLEX_EXAMPLE)
    add_subdirectory(md-flexible)
endif()
