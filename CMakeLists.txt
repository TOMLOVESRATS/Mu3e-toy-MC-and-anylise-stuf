cmake_minimum_required(VERSION 3.20)
project(Mu3e-toy-MC-and-anylise Mu3e C CXX CUDA)

# ============================================================
# Project-wide C++ standard
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# ROOT
# ============================================================
find_package(ROOT REQUIRED COMPONENTS Core RIO Hist Graf Gpad MathCore Matrix Postscript Graf3d Thread)
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

# ============================================================
# CUDA
# ============================================================
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.1/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 120)
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# ============================================================
# ROOT plotting libraries
# ============================================================
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
# CPU Executables (from main branch)
# ============================================================
add_executable(theoryfit Theoryfit.cpp)

add_executable(AnalysePolarisationFromFake AnalysePolarisationFromFake.cpp)

add_executable(Meg HardMEGWay.cpp)
target_link_libraries(Meg PRIVATE ${ROOT_PLOT_LIBS})

add_executable(MontecarloAFBvsx MontecarloAFBvsx.cpp)

add_executable(ToyPolarizationCPU ToyPolarizationCPU.cpp)
target_link_libraries(ToyPolarizationCPU PRIVATE ${ROOT_PLOT_LIBS} GSL::gsl GSL::gslcblas)

add_executable(AverageMuon AverageMuon.cpp)

add_executable(VarianceTestGPU_root VariancetestGPU_root.cpp)
target_compile_features(VarianceTestGPU_root PRIVATE cxx_std_17)
target_link_libraries(VarianceTestGPU_root PRIVATE ${ROOT_PLOT_LIBS})

add_executable(reference_cos reference_cos.cpp)
target_compile_features(reference_cos PRIVATE cxx_std_17)

add_executable(MichelPolarization MichelPolarizationTheoryEventscopy.cpp)

# ============================================================
# GPU Executables (CUDA - from GPU-coding branch)
# ============================================================
add_executable(MonteCarloMuonGPU MontecarloGPU.cu)
set_target_properties(MonteCarloMuonGPU PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    CUDA_ARCHITECTURES 120
)
target_link_libraries(MonteCarloMuonGPU PRIVATE
    CUDA::cudart
    CUDA::curand
    ${ROOT_LIBRARIES}
)

add_executable(GPULayer3Acceptance GPULayer3Acceptance.cu)
set_target_properties(GPULayer3Acceptance PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    CUDA_ARCHITECTURES 120
)
target_link_libraries(GPULayer3Acceptance PRIVATE
    CUDA::cudart
    CUDA::curand
)

add_executable(3LayerMonteCarlo 3LayerMonteCarlo.cu)
set_target_properties(3LayerMonteCarlo PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    CUDA_ARCHITECTURES 120
)
set(THREELAYER_MC_OUTDIR "/home/tom/Mu3e/Photos/MC Muon photos/MC GPU 3layer")
target_compile_definitions(3LayerMonteCarlo PRIVATE
    THREELAYER_MC_OUTDIR="${THREELAYER_MC_OUTDIR}"
)
target_link_libraries(3LayerMonteCarlo PRIVATE
    CUDA::cudart
    CUDA::curand
)

add_executable(3LayerMonteCarlo_Reference 3LayerMonteCarlo_Reference.cu)
set_target_properties(3LayerMonteCarlo_Reference PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    CUDA_ARCHITECTURES 120
)
target_link_libraries(3LayerMonteCarlo_Reference PRIVATE
    CUDA::cudart
    CUDA::curand
)

add_executable(VarianceTestGPU VariancetestGPU.cu)
set_target_properties(VarianceTestGPU PROPERTIES
    BUILD_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    INSTALL_RPATH "/usr/local/cuda/targets/x86_64-linux/lib"
    CUDA_ARCHITECTURES 120
)
target_link_libraries(VarianceTestGPU PRIVATE
    CUDA::cudart
    CUDA::curand
    ${ROOT_LIBRARIES}
)
