# ============================================================
# ROOT
# ============================================================
include_directories(${ROOT_INCLUDE_DIRS})
link_libraries(${ROOT_LIBRARIES})

# ============================================================
# OpenMP
# ============================================================
find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

# ============================================================
# GSL (required by ToyPolarizationCPU)
# ============================================================
find_package(GSL REQUIRED)

set(ROOT_PLOT_LIBS
    ${ROOT_Core_LIBRARY}
    ${ROOT_RIO_LIBRARY}
    ${ROOT_Hist_LIBRARY}
    ${ROOT_Graf_LIBRARY}
    ${ROOT_Gpad_LIBRARY}
    ${ROOT_MathCore_LIBRARY}
    ${ROOT_Matrix_LIBRARY}
    ${ROOT_Postscript_LIBRARY}
    ${ROOT_Graf3d_LIBRARY}
    ${ROOT_Thread_LIBRARY}
)

# ============================================================
# CUDA
# ============================================================
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 120)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)



# ============================================================
# Executable targets
# ============================================================
add_executable(theoryfit Theoryfit.cpp)
add_executable(TheoryCosthetaGraph TheoryCosthetaGraph.cpp)
add_executable(AnalysePolarisationFromFake AnalysePolarisationFromFake.cpp)
add_executable(Meg HardMEGWay.cpp)
add_executable(MCPolarization MonteCarloPolarization.cpp)
add_executable(MontecarloAFBvsx MontecarloAFBvsx.cpp)
add_executable(RemoveBias RemoveBias.cpp)
add_executable(ToyPolarizationCPU ToyPolarizationCPU.cpp)
add_executable(Toy_withBias Toy_withBias.cpp)
# CSV utility that derives K and target N from the fit/GPU summaries.
add_executable(AverageMuon AverageMuon.cpp)
add_executable(AverageMuon_withBias AverageMuon_withBias.cpp)
add_executable(VarianceTestGPU_root VariancetestGPU_root.cpp)
add_executable(VarianceTestGPU_root_withBias VariancetestGPU_root_withBias.cpp)
# Build the reference average-cos(theta)-vs-x executable.
add_executable(reference_cos reference_cos.cpp)
# Enable C++17 filesystem support for reference_cos.cpp.
target_compile_features(reference_cos PRIVATE cxx_std_17)
target_compile_features(VarianceTestGPU_root PRIVATE cxx_std_17)
target_compile_features(VarianceTestGPU_root_withBias PRIVATE cxx_std_17)
target_link_libraries(Meg PRIVATE ${ROOT_PLOT_LIBS})
target_link_libraries(ToyPolarizationCPU PRIVATE ${ROOT_PLOT_LIBS} GSL::gsl GSL::gslcblas)
target_link_libraries(Toy_withBias PRIVATE ${ROOT_PLOT_LIBS} GSL::gsl GSL::gslcblas)
target_link_libraries(VarianceTestGPU_root PRIVATE ${ROOT_PLOT_LIBS})
target_link_libraries(VarianceTestGPU_root_withBias PRIVATE ${ROOT_PLOT_LIBS})

# ============================================================
# GPU targets
# ============================================================
add_executable(MonteCarloMuonGPU MontecarloGPU.cu)
set_target_properties(MonteCarloMuonGPU PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
)
# Link the CUDA runtime and cuRAND for `curand_*` APIs.
target_link_libraries(MonteCarloMuonGPU PRIVATE
    CUDA::cudart
    CUDA::curand
)

# 3-layer GPU executable
add_executable(GPULayer3Acceptance GPULayer3Acceptance.cu)
set_target_properties(GPULayer3Acceptance PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
)
target_link_libraries(GPULayer3Acceptance PRIVATE
    CUDA::cudart
    CUDA::curand
)

# 3-layer Monte Carlo GPU executable
add_executable(3LayerMonteCarlo 3LayerMonteCarlo.cu)
set_target_properties(3LayerMonteCarlo PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
)
set(THREELAYER_MC_OUTDIR "/home/tom/Mu3e/Photos/MC Muon photos/MC GPU 3layer")
target_compile_definitions(3LayerMonteCarlo PRIVATE
    THREELAYER_MC_OUTDIR=\"${THREELAYER_MC_OUTDIR}\"
)
target_link_libraries(3LayerMonteCarlo PRIVATE
    CUDA::cudart
    CUDA::curand
)

# 3-layer Monte Carlo GPU executable (reference)
add_executable(3LayerMonteCarlo_Reference 3LayerMonteCarlo_Reference.cu)
set_target_properties(3LayerMonteCarlo_Reference PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
)
target_link_libraries(3LayerMonteCarlo_Reference PRIVATE
    CUDA::cudart
    CUDA::curand
)

# 3-layer P_fit variance test (GPU)
add_executable(VarianceTestGPU VariancetestGPU.cu)
set_target_properties(VarianceTestGPU PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
)
target_link_libraries(VarianceTestGPU PRIVATE
    CUDA::cudart
    CUDA::curand
)

# 3-layer P_fit variance test with bias (GPU)
add_executable(VarianceTestGPU_withBias variancetest_with_bias.cu)
set_target_properties(VarianceTestGPU_withBias PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
)
target_link_libraries(VarianceTestGPU_withBias PRIVATE
    CUDA::cudart
    CUDA::curand
)

# (ROOT + OpenMP are already linked globally via link_libraries above)
