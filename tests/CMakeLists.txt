add_executable(
    distributed_autopas_tests
    DomainDecompositionTest.cpp
    PerformanceCountersTest.cpp
    InvertedPressureTest.cpp
    ReflectiveBoundaryTest.cpp
)

target_link_libraries(
    distributed_autopas_tests
    PRIVATE
        distributed_autopas
        GTest::gtest_main
)

target_include_directories(
    distributed_autopas_tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../examples/md-flexible"
)

target_compile_features(distributed_autopas_tests PRIVATE cxx_std_20)

include(GoogleTest)
gtest_discover_tests(
    distributed_autopas_tests
    TEST_PREFIX "DistributedAutoPas/"
    DISCOVERY_MODE PRE_TEST
)

function(add_dap_mpi_gtest target test_name number_of_ranks)
    add_test(
        NAME "DistributedAutoPas/${test_name}"
        COMMAND
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} ${number_of_ranks}
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:${target}>
            "--gtest_filter=${test_name}"
            ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties("DistributedAutoPas/${test_name}" PROPERTIES PROCESSORS ${number_of_ranks})
endfunction()


add_executable(
    distributed_autopas_load_balancing_tests
    InvertedPressureLoadBalancerTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    distributed_autopas_load_balancing_tests
    PRIVATE
        distributed_autopas
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(distributed_autopas_load_balancing_tests PRIVATE cxx_std_20)

add_dap_mpi_gtest(
    distributed_autopas_load_balancing_tests
    InvertedPressureLoadBalancerTest.AveragesWorkAcrossPlanesAndMovesSharedBoundaryConsistently
    8
)
add_dap_mpi_gtest(
    distributed_autopas_load_balancing_tests
    InvertedPressureLoadBalancerTest.EnforcesMinimumWidthBeforeApplyingDampedShift
    2
)

add_executable(
    distributed_autopas_communication_tests
    CommunicationTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    distributed_autopas_communication_tests
    PRIVATE
        distributed_autopas
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(distributed_autopas_communication_tests PRIVATE cxx_std_20)

add_dap_mpi_gtest(
    distributed_autopas_communication_tests
    CommunicationTest.ExchangesParticlesWithLeftAndRightNeighbors
    4
)
add_dap_mpi_gtest(
    distributed_autopas_communication_tests
    CommunicationTest.SupportsSplitBeginFinishNeighborExchange
    4
)
add_dap_mpi_gtest(
    distributed_autopas_communication_tests
    CommunicationTest.SplitExchangeSupportsMissingNeighbors
    4
)
add_dap_mpi_gtest(
    distributed_autopas_communication_tests
    CommunicationTest.SupportsEmptyNeighborBuffers
    4
)
add_dap_mpi_gtest(
    distributed_autopas_communication_tests
    CommunicationTest.ExchangesParticlesWithArbitraryDestinationRanks
    4
)
add_dap_mpi_gtest(
    distributed_autopas_communication_tests
    CommunicationTest.RejectsWrongNumberOfDestinationBuffers
    4
)

add_executable(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    distributed_autopas_particle_migration_tests
    PRIVATE
        distributed_autopas
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(distributed_autopas_particle_migration_tests PRIVATE cxx_std_20)

add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.KeepsParticlesThatRemainInLocalDomain
    4
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.MigratesToLeftNeighbor
    4
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.MigratesToRightNeighbor
    4
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.MigratesAcrossPeriodicLeftBoundary
    4
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.MigratesAcrossPeriodicRightBoundary
    4
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.RejectsMovementAcrossMoreThanOneSubdomain
    4
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.MigratesAcrossYFaceInThreeDimensions
    8
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.MigratesAcrossZFaceInThreeDimensions
    8
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.RoutesEdgeMigrationThroughTwoDimensions
    8
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.RoutesCornerMigrationThroughThreeDimensions
    8
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.DropsParticlesAtNoneBoundary
    4
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.WrapsPeriodicYBoundaryInThreeDimensions
    8
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.RejectsParticleCrossingReflectiveBoundary
    4
)
add_dap_mpi_gtest(
    distributed_autopas_particle_migration_tests
    ParticleMigrationTest.RejectsReflectiveCrossingWithoutSubdivisionInThatDimension
    4
)

add_executable(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    distributed_autopas_halo_exchange_tests
    PRIVATE
        distributed_autopas
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(distributed_autopas_halo_exchange_tests PRIVATE cxx_std_20)

add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.ExchangesLeftAndRightBoundaryParticles
    4
)
add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.IgnoresParticlesOutsideHaloWidth
    4
)
add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.ShiftsPeriodicLeftHaloBelowGlobalDomain
    4
)
add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.ShiftsPeriodicRightHaloAboveGlobalDomain
    4
)
add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.PropagatesFaceEdgeAndCornerHalosInThreeDimensions
    8
)
add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.ShiftsPeriodicCornerHaloAcrossAllDimensions
    8
)
add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.CreatesPeriodicHalosLocallyForUndecomposedDimension
    4
)
add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.NoneBoundarySuppressesGlobalHaloButKeepsInternalHalo
    8
)
add_dap_mpi_gtest(
    distributed_autopas_halo_exchange_tests
    HaloExchangeTest.ReflectiveBoundarySuppressesGlobalHaloButKeepsInternalHalo
    4
)



add_executable(
    distributed_autopas_runtime_tests
    RuntimeTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    distributed_autopas_runtime_tests
    PRIVATE
        distributed_autopas
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(distributed_autopas_runtime_tests PRIVATE cxx_std_20)

add_dap_mpi_gtest(
    distributed_autopas_runtime_tests
    RuntimeTest.ReportsRankSizeAndRoot
    4
)
add_dap_mpi_gtest(
    distributed_autopas_runtime_tests
    RuntimeTest.BroadcastsStringFromNonzeroRoot
    4
)

add_executable(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.cpp
    MPITestMain.cpp
)

target_link_libraries(
    distributed_autopas_public_api_tests
    PRIVATE
        distributed_autopas
        GTest::gtest
        MPI::MPI_CXX
)

target_compile_features(distributed_autopas_public_api_tests PRIVATE cxx_std_20)

add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.AddsReplicatedInputFromRootExactlyOnce
    4
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.RedistributesContributionsFromAllRanksToTheirOwners
    4
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.UsesThreeDimensionalProcessGridForInitialOwnership
    8
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.ResizesLocalBoxAndMigratesParticles
    4
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.AppliesInvertedPressureLoadBalancingAndMigratesParticles
    2
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.RedistributesParticlesUsingAdaptiveTargetRanks
    2
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.AppliesRepeatedInvertedPressureLoadBalancingFromCurrentDomain
    2
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.BalancesMigratesAndComputesAcrossAdaptiveThreeDimensionalCorner
    8
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.ReducesDoubleValuesGlobally
    4
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.AppliesKernelOnlyToOwnedParticlesInRegion
    1
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.ComputesInteractionAcrossThreeDimensionalCorner
    8
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.ComputesPeriodicForcesWithSingleRank
    1
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.ComputesSamePeriodicForcesAcrossFourRanks
    4
)
add_dap_mpi_gtest(
    distributed_autopas_public_api_tests
    DistributedAutoPasTest.ComputesForcesAfterExplicitPreparation
    4
)
