# Mu3e Toy MC and Analysis Suite

This repository contains Monte Carlo simulations and analysis tools for the Mu3e experiment, focusing on muon polarization measurements and related physics studies. The project includes both CPU-based analysis tools and GPU-accelerated computation code.

## Repository Structure

### Branches
- **main**: CPU-based analysis tools and Monte Carlo simulations
- **GPU-coding**: CUDA-based GPU accelerated computations

## File Descriptions

### CPU Analysis Tools (Main Branch)

#### `Theoryfit.cpp`
- **Purpose**: Theoretical fitting of polarization data
- **Branch**: main
- **Executable**: `theoryfit`
- **Usage**: `./theoryfit`
- **Dependencies**: ROOT

#### `AnalysePolarisationFromFake.cpp`
- **Purpose**: Analyzes polarization from simulated/fake data
- **Branch**: main
- **Executable**: `AnalysePolarisationFromFake`
- **Usage**: `./AnalysePolarisationFromFake`
- **Configuration**: Edit `baseDir` variable (line 61) to point to your ROOT data files
- **Dependencies**: ROOT, MichelPolarizationTheoryEventscopy.cpp

#### `HardMEGWay.cpp`
- **Purpose**: MEG analysis approach for polarization studies
- **Branch**: main
- **Executable**: `Meg`
- **Usage**: `./Meg`
- **Dependencies**: ROOT plotting libraries

#### `MontecarloAFBvsx.cpp`
- **Purpose**: Monte Carlo simulation of asymmetry forward-backward (AFB) as function of x
- **Branch**: main
- **Executable**: `MontecarloAFBvsx`
- **Usage**: `./MontecarloAFBvsx`
- **Dependencies**: ROOT

#### `ToyPolarizationCPU.cpp`
- **Purpose**: Toy model for CPU-based polarization calculations
- **Branch**: main
- **Executable**: `ToyPolarizationCPU`
- **Usage**: `./ToyPolarizationCPU`
- **Dependencies**: ROOT, GSL (GNU Scientific Library)

#### `AverageMuon.cpp`
- **Purpose**: CSV utility that computes average muon properties and derives K and target N from fit/GPU summaries
- **Branch**: main
- **Executable**: `AverageMuon`
- **Usage**: `./AverageMuon <input_file>`
- **Dependencies**: ROOT

#### `VariancetestGPU_root.cpp`
- **Purpose**: CPU-based variance testing for GPU comparison (uses ROOT analysis)
- **Branch**: main
- **Executable**: `VarianceTestGPU_root`
- **Usage**: `./VarianceTestGPU_root <input_file>`
- **Dependencies**: ROOT, C++17

#### `reference_cos.cpp`
- **Purpose**: Reference implementation for computing average cos(theta) vs x
- **Branch**: main
- **Executable**: `reference_cos`
- **Usage**: `./reference_cos <root_file> <histogram_name>`
- **Dependencies**: ROOT, C++17 filesystem support

#### `MichelPolarizationTheoryEventscopy.cpp`
- **Purpose**: Michel polarization theory event generation and processing
- **Branch**: main
- **Executable**: `MichelPolarization`
- **Usage**: `./MichelPolarization`
- **Note**: Often run before other analysis tools as it generates input data
- **Dependencies**: ROOT

#### `AverageCostheta.h`
- **Purpose**: Header file containing utility functions for computing average cos(theta) vs x
- **Branch**: main
- **Function**: `computeAvgCosThetaVsXFromFile(filename, histogram_name, tag)`
- **Usage**: Include in your C++ code: `#include "AverageCostheta.h"`
- **Note**: Use as a header-only library; do NOT add to CMake builds if using as include-only
- **Dependencies**: ROOT

---

### GPU Acceleration Tools (GPU-coding Branch)

#### `MontecarloGPU.cu`
- **Purpose**: GPU-accelerated Monte Carlo muon simulations using CUDA
- **Branch**: GPU-coding
- **Executable**: `MonteCarloMuonGPU`
- **Usage**: `./MonteCarloMuonGPU [options]`
- **Performance**: Significantly faster than CPU version for large sample sizes
- **Dependencies**: CUDA Toolkit, cuRAND, ROOT

#### `GPULayer3Acceptance.cu`
- **Purpose**: GPU-accelerated 3-layer detector acceptance calculations
- **Branch**: GPU-coding
- **Executable**: `GPULayer3Acceptance`
- **Usage**: `./GPULayer3Acceptance`
- **Dependencies**: CUDA Toolkit, cuRAND

#### `3LayerMonteCarlo.cu`
- **Purpose**: GPU-accelerated 3-layer Monte Carlo simulation with output to file
- **Branch**: GPU-coding
- **Executable**: `3LayerMonteCarlo`
- **Usage**: `./3LayerMonteCarlo`
- **Output**: Saves results to `/home/tom/Mu3e/Photos/MC Muon photos/MC GPU 3layer`
- **Note**: Update output directory path in CMakeLists.txt if needed
- **Dependencies**: CUDA Toolkit, cuRAND

#### `3LayerMonteCarlo_Reference.cu`
- **Purpose**: Reference implementation of 3-layer GPU Monte Carlo for comparison/validation
- **Branch**: GPU-coding
- **Executable**: `3LayerMonteCarlo_Reference`
- **Usage**: `./3LayerMonteCarlo_Reference`
- **Dependencies**: CUDA Toolkit, cuRAND

#### `VariancetestGPU.cu`
- **Purpose**: GPU-based variance testing for polarization fitting
- **Branch**: GPU-coding
- **Executable**: `VarianceTestGPU`
- **Usage**: `./VarianceTestGPU`
- **Comparison**: Compare results with CPU version (VarianceTestGPU_root)
- **Dependencies**: CUDA Toolkit, cuRAND, ROOT

---

## Building the Project

### Prerequisites
- CMake 3.20 or higher
- C++ compiler with C++17 support
- ROOT 6.x or higher
- OpenMP
- GSL (GNU Scientific Library)
- CUDA Toolkit 13.1+ (for GPU targets)

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/TOMLOVESRATS/Mu3e-toy-MC-and-anylise-stuf.git
cd Mu3e-toy-MC-and-anylise-stuf

# Create build directory
mkdir build
cd build

# Configure CMake (for main branch - CPU only)
cmake ..

# Or configure for GPU support (checkout GPU-coding branch)
git checkout GPU-coding
cmake ..

# Compile
make -j$(nproc)
```

### Build Specific Targets

```bash
# Build only CPU tools
make theoryfit AnalysePolarisationFromFake Meg MontecarloAFBvsx ToyPolarizationCPU

# Build only GPU tools
make MonteCarloMuonGPU GPULayer3Acceptance 3LayerMonteCarlo VarianceTestGPU
```

---

## Running the Executables

### CPU-Based Executables

#### Theory Fitting
```bash
./build/theoryfit
```

#### Polarization Analysis
```bash
./build/AnalysePolarisationFromFake
```
**Setup**: Edit the `baseDir` variable in `AnalysePolarisationFromFake.cpp` (around line 61) to point to your ROOT data files location.

#### MEG Analysis
```bash
./build/Meg
```

#### Monte Carlo AFB vs X
```bash
./build/MontecarloAFBvsx
```

#### CPU Toy Polarization
```bash
./build/ToyPolarizationCPU
```

#### Average Muon Analysis
```bash
./build/AverageMuon <input_csv_file>
```

#### Variance Test (CPU)
```bash
./build/VarianceTestGPU_root <input_file>
```

#### Reference Cos(theta) Calculator
```bash
./build/reference_cos <root_file> <histogram_name>
```

#### Michel Polarization
```bash
./build/MichelPolarization
```

### GPU-Based Executables (GPU-coding branch)

First, check out the GPU branch:
```bash
git checkout GPU-coding
cd build && cmake .. && make
```

#### GPU Monte Carlo Muon
```bash
./build/MonteCarloMuonGPU
```

#### GPU Layer 3 Acceptance
```bash
./build/GPULayer3Acceptance
```

#### GPU 3-Layer Monte Carlo
```bash
./build/3LayerMonteCarlo
```
Output saved to: `/home/tom/Mu3e/Photos/MC Muon photos/MC GPU 3layer`

#### GPU 3-Layer Reference
```bash
./build/3LayerMonteCarlo_Reference
```

#### GPU Variance Test
```bash
./build/VarianceTestGPU
```

---

## Workflow Example

### Complete Analysis Pipeline

1. **Generate Theory Events**
   ```bash
   ./build/MichelPolarization
   ```

2. **Analyze Fake/Simulated Data**
   ```bash
   ./build/AnalysePolarisationFromFake
   ```

3. **Compute Average Cos(theta)**
   ```bash
   ./build/reference_cos data/output.root histogram_name
   ```

4. **Run Variance Tests**
   ```bash
   ./build/VarianceTestGPU_root data/output.root
   ```

5. **Compare with GPU (if available)**
   ```bash
   git checkout GPU-coding
   cd ../build && cmake .. && make
   ./build/VarianceTestGPU data/output.root
   ```

---

## Using AverageCostheta as a Header Library

The `AverageCostheta.h` file provides utility functions for analysis. To use it in your own code:

```cpp
#include "AverageCostheta.h"

int main() {
    // Compute average cos(theta) vs x from ROOT file
    computeAvgCosThetaVsXFromFile("mydata.root", "my_histogram", "tag");
    return 0;
}
```

**Note**: Do NOT add this to CMakeLists.txt if using as a header-only library.

---

## Performance Notes

- **CPU tools**: Suitable for small to medium datasets, easier to debug
- **GPU tools**: Dramatically faster for large simulations (100x+ speedup typical)
- **Variance tests**: Run both CPU and GPU versions to validate GPU correctness

---

## Environment Variables

Update these paths in CMakeLists.txt or set as environment variables:
- `CUDA_PATH`: Path to CUDA installation (default: `/usr/local/cuda-13.1`)
- `ROOT_CONFIG`: Point to ROOT installation
- `THREELAYER_MC_OUTDIR`: Output directory for 3-layer MC results

---

## Troubleshooting

### CMake Build Fails
- Ensure ROOT is properly installed: `root-config --version`
- Check CUDA installation: `nvcc --version`
- Verify GSL is installed: `gsl-config --version`

### CUDA Compilation Errors
- Check CUDA architecture compatibility in CMakeLists.txt (currently set to 120)
- Update to match your GPU: `cmake -DCMAKE_CUDA_ARCHITECTURES=<your_arch> ..`

### ROOT File Issues
- Verify ROOT files exist and are readable
- Update `baseDir` paths in source files as needed

---

## License

[Add your license information here]

## Contact

[Add contact information]