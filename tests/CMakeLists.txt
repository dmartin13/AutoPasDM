add_executable(
    autopasdm_tests
    DomainDecompositionTest.cpp
    PerformanceCountersTest.cpp
    InvertedPressureTest.cpp
    ReflectiveBoundaryTest.cpp
)

target_link_libraries(
    autopasdm_tests
    PRIVATE
        autopasdm
        GTest::gtest_main
)

target_include_directories(
    autopasdm_tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../examples/md-flexible"
)

target_compile_features(autopasdm_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(
    autopasdm_tests
    TEST_PREFIX "AutoPasDM/"
    DISCOVERY_MODE PRE_TEST
)

function(add_autopasdm_mpi_gtest target test_name number_of_ranks)
    add_test(
        NAME "AutoPasDM/${test_name}"
        COMMAND
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} ${number_of_ranks}
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${target}>
            "--gtest_filter=${test_name}"
            ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties("AutoPasDM/${test_name}" PROPERTIES PROCESSORS ${number_of_ranks})
endfunction()


add_executable(
    autopasdm_load_balancing_tests
    InvertedPressureLoadBalancerTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    autopasdm_load_balancing_tests
    PRIVATE
        autopasdm
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(autopasdm_load_balancing_tests PRIVATE cxx_std_20)

add_autopasdm_mpi_gtest(
    autopasdm_load_balancing_tests
    InvertedPressureLoadBalancerTest.AveragesWorkAcrossPlanesAndMovesSharedBoundaryConsistently
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_load_balancing_tests
    InvertedPressureLoadBalancerTest.EnforcesMinimumWidthBeforeApplyingDampedShift
    2
)

add_executable(
    autopasdm_communication_tests
    CommunicationTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    autopasdm_communication_tests
    PRIVATE
        autopasdm
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(autopasdm_communication_tests PRIVATE cxx_std_20)

add_autopasdm_mpi_gtest(
    autopasdm_communication_tests
    CommunicationTest.ExchangesParticlesWithLeftAndRightNeighbors
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_communication_tests
    CommunicationTest.SupportsSplitBeginFinishNeighborExchange
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_communication_tests
    CommunicationTest.SplitExchangeSupportsMissingNeighbors
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_communication_tests
    CommunicationTest.SupportsEmptyNeighborBuffers
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_communication_tests
    CommunicationTest.ExchangesParticlesWithArbitraryDestinationRanks
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_communication_tests
    CommunicationTest.RejectsWrongNumberOfDestinationBuffers
    4
)

add_executable(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    autopasdm_particle_migration_tests
    PRIVATE
        autopasdm
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(autopasdm_particle_migration_tests PRIVATE cxx_std_20)

if(AUTOPASDM_ENABLE_DETAILED_TIMERS)
    add_autopasdm_mpi_gtest(
        autopasdm_particle_migration_tests
        ParticleMigrationTest.SkipsNeighborExchangeWhenNoRankHasEmigrants
        4
    )
endif()

add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.KeepsParticlesThatRemainInLocalDomain
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.MigratesToLeftNeighbor
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.MigratesToRightNeighbor
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.MigratesAcrossPeriodicLeftBoundary
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.MigratesAcrossPeriodicRightBoundary
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.RejectsMovementAcrossMoreThanOneSubdomain
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.MigratesAcrossYFaceInThreeDimensions
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.MigratesAcrossZFaceInThreeDimensions
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.RoutesEdgeMigrationThroughTwoDimensions
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.RoutesCornerMigrationThroughThreeDimensions
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.DropsParticlesAtNoneBoundary
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.WrapsPeriodicYBoundaryInThreeDimensions
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.RejectsParticleCrossingReflectiveBoundary
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_particle_migration_tests
    ParticleMigrationTest.RejectsReflectiveCrossingWithoutSubdivisionInThatDimension
    4
)

add_executable(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    autopasdm_halo_exchange_tests
    PRIVATE
        autopasdm
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(autopasdm_halo_exchange_tests PRIVATE cxx_std_20)

add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.ExchangesLeftAndRightBoundaryParticles
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.IgnoresParticlesOutsideHaloWidth
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.ShiftsPeriodicLeftHaloBelowGlobalDomain
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.ShiftsPeriodicRightHaloAboveGlobalDomain
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.PropagatesFaceEdgeAndCornerHalosInThreeDimensions
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.ShiftsPeriodicCornerHaloAcrossAllDimensions
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.CreatesPeriodicHalosLocallyForUndecomposedDimension
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.NoneBoundarySuppressesGlobalHaloButKeepsInternalHalo
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_halo_exchange_tests
    HaloExchangeTest.ReflectiveBoundarySuppressesGlobalHaloButKeepsInternalHalo
    4
)



add_executable(
    autopasdm_runtime_tests
    RuntimeTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    autopasdm_runtime_tests
    PRIVATE
        autopasdm
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(autopasdm_runtime_tests PRIVATE cxx_std_20)

add_autopasdm_mpi_gtest(
    autopasdm_runtime_tests
    RuntimeTest.ReportsRankSizeAndRoot
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_runtime_tests
    RuntimeTest.BroadcastsStringFromNonzeroRoot
    4
)

add_executable(
    autopasdm_public_api_tests
    AutoPasDMTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    autopasdm_public_api_tests
    PRIVATE
        autopasdm
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(autopasdm_public_api_tests PRIVATE cxx_std_20)

add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.AddsReplicatedInputFromRootExactlyOnce
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.RedistributesContributionsFromAllRanksToTheirOwners
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.UsesThreeDimensionalProcessGridForInitialOwnership
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.ResizesLocalBoxAndMigratesParticles
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.AppliesInvertedPressureLoadBalancingAndMigratesParticles
    2
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.RedistributesParticlesUsingAdaptiveTargetRanks
    2
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.AppliesRepeatedInvertedPressureLoadBalancingFromCurrentDomain
    2
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.BalancesMigratesAndComputesAcrossAdaptiveThreeDimensionalCorner
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.ReducesDoubleValuesGlobally
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.AppliesKernelOnlyToOwnedParticlesInRegion
    1
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.ComputesInteractionAcrossThreeDimensionalCorner
    8
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.ComputesPeriodicForcesWithSingleRank
    1
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.ComputesSamePeriodicForcesAcrossFourRanks
    4
)
add_autopasdm_mpi_gtest(
    autopasdm_public_api_tests
    AutoPasDMTest.ComputesForcesAfterExplicitPreparation
    4
)
