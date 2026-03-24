//====================================================
// 3LayerMonteCarlo.cu
// - Monte Carlo in (x, cos#theta) with Michel spectrum + polarisation P
// - Apply Mu3e 3-layer helix acceptance (conf18-style SHAPE toy model logic)
// - Keep x and cos#theta axes for future code
// - Fill histogram of ACCEPTED EVENTS (NOT an acceptance map)
//====================================================

// ROOT includes
#include "TFile.h"
#include "TH2.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TLatex.h"

// CUDA
#include <cuda_runtime.h>
#include <curand_kernel.h>

// C++
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

//==============================================================
// Parameters
//==============================================================
constexpr int numberOfBinsX        = 100;
constexpr int numberOfBinsCosTheta = 100;

// theta range
static const double thetaMinPlot = M_PI/2 - 1.3;
static const double thetaMaxPlot = M_PI/2 + 1.3;

// cos(theta) range
static const double cosThetaMin =
    std::min(std::cos(thetaMinPlot), std::cos(thetaMaxPlot));
static const double cosThetaMax =
    std::max(std::cos(thetaMinPlot), std::cos(thetaMaxPlot));

// energy range
constexpr double E_MAX    = 52.83;
constexpr double E_MIN    = 10.0;
constexpr double xMinPlot = E_MIN/E_MAX;
constexpr double xMaxPlot = 1.0;

// PDG constant
constexpr double x0 = 9.67e-3;

struct Vertex { float x, y, z; };

constexpr float TARGET_ALPHA_MM = 19.1033f;
constexpr float TARGET_BETA = 0.378397f;

//==============================================================
// Important part: target shape and inverse sampling
// This section defines the target z-shape and the data needed to sample it.
//==============================================================

struct ZCdfTable {
    int Nz = 0;
    std::vector<float> zEdges;
    std::vector<float> cdfZ;
};

// Load the target z-distribution and build the inverse-sampling CDF.
static ZCdfTable LoadZCdfTable(const std::string& rootPath,
                               const std::string& histName = "target_z_dist")
{
    ZCdfTable tab;

    TFile f(rootPath.c_str(), "READ");
    if (f.IsZombie()) {
        throw std::runtime_error("Cannot open target ROOT file: " + rootPath);
    }

    TH1* h = dynamic_cast<TH1*>(f.Get(histName.c_str()));
    if (!h) {
        throw std::runtime_error("Cannot find TH1 '" + histName + "' in " + rootPath);
    }

    tab.Nz = h->GetXaxis()->GetNbins();
    tab.zEdges.resize(tab.Nz + 1);
    tab.cdfZ.resize(tab.Nz + 1, 0.0f);

    for (int iz = 0; iz <= tab.Nz; ++iz) {
        tab.zEdges[iz] = (float)h->GetXaxis()->GetBinLowEdge(iz + 1);
    }

    double total = 0.0;
    for (int iz = 1; iz <= tab.Nz; ++iz) {
        total += h->GetBinContent(iz);
    }

    if (total <= 0.0) {
        throw std::runtime_error("target_z_dist has zero total entries.");
    }

    double acc = 0.0;
    tab.cdfZ[0] = 0.0f;
    for (int iz = 1; iz <= tab.Nz; ++iz) {
        acc += h->GetBinContent(iz);
        tab.cdfZ[iz] = (float)(acc / total);
    }
    tab.cdfZ[tab.Nz] = 1.0f;

    return tab;
}

//==============================================================
// 3-layer acceptance table 7.1
//==============================================================
constexpr float B_T_f   = 1.0f;

constexpr float r1_mm_f = 23.3f;
constexpr float r2_mm_f = 29.8f;
constexpr float r3_mm_f = 73.9f;

constexpr float L1_mm_f = 124.7f;
constexpr float L2_mm_f = 124.7f;
constexpr float L3_mm_f = 351.9f;

constexpr float M_E_MEV_f = 0.51099895f;

//==============================================================
// Constant memory for gpu to read
//==============================================================
__device__ __constant__ double d_cosThetaMin;
__device__ __constant__ double d_cosThetaMax;
__device__ __constant__ double d_xMinPlot;
__device__ __constant__ double d_xMaxPlot;
__device__ __constant__ float d_cosThetaMin_f;
__device__ __constant__ float d_cosThetaMax_f;
__device__ __constant__ float d_xMinPlot_f;
__device__ __constant__ float d_xMaxPlot_f;

//==============================================================
// CUDA error check macro
//==============================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";      \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

//==============================================================
// Convenience only
// Progress reporting and small helper formatting do not change the physics result.
//==============================================================

//==============================================================
// CPU helper: simple progress bar
//==============================================================
static void PrintProgress(unsigned long long done,
                          unsigned long long total,
                          double elapsedSec)
{
    const int barWidth = 40;

    double frac = (total > 0) ? (double)done / (double)total : 0.0;
    if (frac > 1.0) frac = 1.0;

    int filled = (int)(barWidth * frac);

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i)
        std::cout << (i < filled ? '=' : ' ');
    std::cout << "] "
              << std::fixed << std::setprecision(1)
              << (100.0 * frac) << "%  "
              << done << "/" << total
              << "  elapsed " << std::setprecision(2) << elapsedSec << " s"
              << std::flush;

    if (done >= total) std::cout << "\n";
}

//==============================================================
// Convenience helpers
// Small utility helpers used by the main Monte Carlo sections.
//==============================================================
__device__ __forceinline__ double clampDouble(double x, double lo, double hi)
{
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}
// Root bins are 1..N, so map a floating-point value to that indexing.
__device__ __forceinline__ int valueToBin_f(float v, float vmin, float vmax, int nbins)
{
    if (v < vmin || v >= vmax) return -1;
    float f = (v - vmin) / (vmax - vmin);
    int bin0 = (int)(f * nbins);
    return bin0 + 1; // ROOT-style 1..nbins
}

//==============================================================
// Important part: inverse sampling
// Source kept here:
// https://blog.traverseresearch.nl/fast-cdf-generation-on-the-gpu-for-light-picking-5c50b97c552b
//==============================================================
__device__ __forceinline__ int upper_bound_cdf(const float* cdf, int n, float u)
{
    int lo = 0;
    int hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < u) lo = mid;
        else hi = mid;
    }
    return hi;
}

__device__ __forceinline__ Vertex sample_target_vertex_from_zcdf_rgeom(
    curandStatePhilox4_32_10_t* st,
    const float* zEdges,
    const float* cdfZ,
    int Nz)
{
    const float uZ = curand_uniform(st);
    int iz = upper_bound_cdf(cdfZ, Nz, uZ);
    if (iz < 1) iz = 1;
    if (iz > Nz) iz = Nz;
    const float z0 = zEdges[iz - 1] + (zEdges[iz] - zEdges[iz - 1]) * curand_uniform(st);
    float rmax = TARGET_ALPHA_MM - TARGET_BETA * fabsf(z0);
    if (rmax < 0.0f) rmax = 0.0f;
    const float r0 = rmax * sqrtf(curand_uniform(st));

    const float phi = 2.0f * (float)M_PI * curand_uniform(st);

    Vertex v;
    v.x = r0 * cosf(phi);
    v.y = r0 * sinf(phi);
    v.z = z0;
    return v;
}

//==============================================================
// Important part: Michel rate
// This is the physics formula used for the event weight before acceptance.
//==============================================================
__device__ __forceinline__ double michel_device(double xp, double cosTheta,
                                                double rho, double eta,
                                                double epsilon, double delta,
                                                double P)
{
    if (xp <= 0.0 || xp > 1.0) return 0.0;

    double value =
        xp * xp *
        (3.0 * (1.0 - xp)
         + 2.0 * rho * (4.0 * xp / 3.0 - 1.0)
         + 3.0 * eta * x0 * (1.0 - xp) / xp
         + P * epsilon * cosTheta *
               (1.0 - xp + (2.0 / 3.0) * delta * (4.0 * xp - 3.0)));

    return (value > 0.0) ? value : 0.0;
}

//==============================================================
// Important part: acceptance
// This section implements the 3-layer helix acceptance test.
//==============================================================
__device__ __forceinline__ bool hit_at_radius_device(float pT_MeV, float pz_MeV,
                                                     float r_mm, float zHalf_mm,
                                                     float z0_mm, bool recurl)
{
    // Guard
    if (pT_MeV <= 0.0f) return false;

    // Curvature radius in mm
    const float rho_mm = pT_MeV / (0.3f * B_T_f);
    if (rho_mm <= 0.0f) return false;

    // Geometric reach
    if (r_mm > 2.0f * rho_mm) return false;

    // arg = r / (2 rho) in [0,1]
    float arg = r_mm / (2.0f * rho_mm);
    arg = fminf(1.0f, fmaxf(0.0f, arg));

    // dphi to reach radius r
    const float dphi1 = 2.0f * asinf(arg);
    const float dphi  = recurl ? (2.0f * (float)M_PI - dphi1) : dphi1;

    // Helix z advance in mm
    const float z_mm = rho_mm * dphi * (pz_MeV / pT_MeV);

    return fabsf(z0_mm + z_mm) <= zHalf_mm;
}

__device__ __forceinline__ float norm2pi(float a)
{
    const float twoPi = 2.0f * (float)M_PI;
    a = fmodf(a, twoPi);
    if (a < 0.0f) a += twoPi;
    return a;
}

__device__ __forceinline__ bool hit_at_radius_device_offset(
    float pT_MeV,
    float pz_MeV,
    float r_layer_mm,
    float zHalf_mm,
    float x0_mm,
    float y0_mm,
    float z0_mm,
    float phi_p,
    bool recurl)
{
    const float pT_GeV = pT_MeV * 1e-3f;
    const float rho_m = pT_GeV / (0.3f * B_T_f);
    if (rho_m <= 0.0f) return false;

    const float R_m = r_layer_mm * 1e-3f;
    const float x0_m = x0_mm * 1e-3f;
    const float y0_m = y0_mm * 1e-3f;

    const float xc = x0_m + rho_m * sinf(phi_p);
    const float yc = y0_m - rho_m * cosf(phi_p);

    const float D = sqrtf(xc * xc + yc * yc);
    if (D < 1e-9f) {
        return hit_at_radius_device(pT_MeV, pz_MeV, r_layer_mm, zHalf_mm, z0_mm, recurl);
    }

    const float A = xc * xc + yc * yc + rho_m * rho_m - R_m * R_m;
    const float B = 2.0f * rho_m * xc;
    const float C = 2.0f * rho_m * yc;

    const float M = sqrtf(B * B + C * C);
    if (M < 1e-12f) return false;

    const float s = -A / M;
    if (s < -1.0f || s > 1.0f) return false;

    const float psi0 = atan2f(C, B);
    const float d = acosf(fminf(1.0f, fmaxf(-1.0f, s)));

    const float psiA = psi0 + d;
    const float psiB = psi0 - d;
    const float psi_start = atan2f(y0_m - yc, x0_m - xc);

    const float dA = norm2pi(psiA - psi_start);
    const float dB = norm2pi(psiB - psi_start);

    float dpsi_short = fminf(dA, dB);
    if (dpsi_short < 1e-8f) dpsi_short = fmaxf(dA, dB);

    const float dpsi = recurl ? (2.0f * (float)M_PI - dpsi_short) : dpsi_short;
    const float z_m = rho_m * dpsi * (pz_MeV / pT_MeV);
    const float z_mm = z_m * 1e3f;

    return fabsf(z0_mm + z_mm) <= zHalf_mm;
}

//==============================================================
// RNG init kernel
//==============================================================
__global__ void init_rng(curandStatePhilox4_32_10_t* states,
                         int nStates,
                         unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nStates) return;
    curand_init(seed, tid, 0ULL, &states[tid]);
}

//==============================================================
// Important part: main Monte Carlo generation
// This kernel combines the Michel rate, rejection sampling, target shape,
// and acceptance into the accepted-event histogram.
//==============================================================
__global__ void generate_histogram_chunk(
    unsigned int* __restrict__ d_finalHist,      // size: histSize (10k)
    unsigned long long* acceptedCount,
    unsigned long long* trialsCount,
    long long N,
    curandStatePhilox4_32_10_t* states,
    int nStates,
    const float* d_zEdges,
    const float* d_cdfZ,
    int Nz,
    float maxAllowedRate,
    int nTrialsPerThread,
    double par_rho,
    double par_eta,
    double par_epsilon,
    double par_delta,
    double P
)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nStates) return;

    const int histSize = numberOfBinsX * numberOfBinsCosTheta;

    // -------- shared histogram (dynamic shared mem) --------
    extern __shared__ unsigned int shist[];

    // init shared hist
    for (int k = threadIdx.x; k < histSize; k += blockDim.x) {
        shist[k] = 0U;
    }
    __syncthreads();

    curandStatePhilox4_32_10_t local = states[tid];

    unsigned long long localTrials = 0ULL;
    unsigned long long stopSeen    = 0ULL;

    for (int t = 0; t < nTrialsPerThread; ++t) {

        ++localTrials;

        // stop check every 64 iterations
        if ((t & 63) == 0) {
            stopSeen = atomicAdd(acceptedCount, 0ULL);
            if (stopSeen >= (unsigned long long)N) break;
        }

        float4 u4 = curand_uniform4(&local);

        float randomX = d_xMinPlot_f + (d_xMaxPlot_f - d_xMinPlot_f) * u4.x;
        if (randomX >= d_xMaxPlot_f)
            randomX = nextafterf(d_xMaxPlot_f, d_xMinPlot_f);

        const float randomCosTheta =
            d_cosThetaMin_f + (d_cosThetaMax_f - d_cosThetaMin_f) * u4.y;

        // Michel rate
        const double rate = michel_device((double)randomX, (double)randomCosTheta,
                                          par_rho, par_eta, par_epsilon, par_delta, P);
        if (rate <= 0.0) continue;

        // rejection
        const double r = (double)maxAllowedRate * (double)u4.z;
        if (r >= rate) continue;

        // ---------------- smearing before helix acceptance ----------------
        float sigma = 0.0512f * randomX - 0.0039f;
        if (sigma < 0.0f) sigma = 0.0f;

        float xSmeared = randomX;
        if (sigma > 0.0f) {
            float2 n2 = curand_normal2(&local);
            float trial = randomX + sigma * n2.x;
            if (trial >= d_xMinPlot_f && trial < d_xMaxPlot_f) {
                xSmeared = trial;
            }
        }

        // ---------------- helix acceptance ----------------
        const float zH1 = 0.5f * L1_mm_f;
        const float zH2 = 0.5f * L2_mm_f;
        const float zH3 = 0.5f * L3_mm_f;
        const Vertex vtx = sample_target_vertex_from_zcdf_rgeom(&local, d_zEdges, d_cdfZ, Nz);
        const float x0_mm = vtx.x;
        const float y0_mm = vtx.y;
        const float z0_mm = vtx.z;
        const float phi_p = 2.0f * (float)M_PI * u4.w;

        const float E = xSmeared * (float)E_MAX;
        if (E <= M_E_MEV_f) continue;

        const float p = sqrtf(E*E - M_E_MEV_f*M_E_MEV_f); // MeV/c
        const float u = fminf(1.0f, fmaxf(-1.0f, randomCosTheta));
        const float sinT = sqrtf(fmaxf(0.0f, 1.0f - u*u));

        const float pz = p * u;
        const float pT = p * sinT;
        if (pT <= 1e-9f) continue;

        const bool o1 = hit_at_radius_device_offset(pT, pz, r1_mm_f, zH1, x0_mm, y0_mm, z0_mm, phi_p, false);
        const bool o2 = hit_at_radius_device_offset(pT, pz, r2_mm_f, zH2, x0_mm, y0_mm, z0_mm, phi_p, false);
        const bool o3 = hit_at_radius_device_offset(pT, pz, r3_mm_f, zH3, x0_mm, y0_mm, z0_mm, phi_p, false);
        const bool r2 = hit_at_radius_device_offset(pT, pz, r3_mm_f, zH3, x0_mm, y0_mm, z0_mm, phi_p, true);
        if (!(o1 && o2 && o3 && r2)) continue;

        const int fillBinX = valueToBin_f(xSmeared, d_xMinPlot_f, d_xMaxPlot_f, numberOfBinsX);
        const int fillBinY = valueToBin_f(randomCosTheta, d_cosThetaMin_f, d_cosThetaMax_f, numberOfBinsCosTheta);
        if (fillBinX < 1 || fillBinY < 1) continue;

        // reserve accepted slot (global)
        const unsigned long long idx = atomicAdd(acceptedCount, 1ULL);
        if (idx >= (unsigned long long)N) break;

        const int histIndex = (fillBinX - 1) + numberOfBinsX * (fillBinY - 1);

        // FAST: shared-memory atomic
        atomicAdd(&shist[histIndex], 1U);
    }

    atomicAdd(trialsCount, localTrials);
    states[tid] = local;

    __syncthreads();

    // flush shared -> global (10k atomics per block, NOT per accepted event)
    for (int k = threadIdx.x; k < histSize; k += blockDim.x) {
        const unsigned int v = shist[k];
        if (v) atomicAdd(&d_finalHist[k], v);
    }
}
//==============================================================
// Formatting helper
//==============================================================
static std::string FormatPTag(double P)
{
    std::ostringstream tag;
    tag << std::fixed << std::setprecision(1) << P;
    std::string s = tag.str();
    for (char &c : s) if (c == '.') c = 'p';
    if (!s.empty() && s[0] == '-') s[0] = 'm';
    return s;
}

//==============================================================
// Important part: rejection method
// Find the maximum rate used as the rejection-sampling envelope.
//==============================================================
double findMaximumRate_host(const double* par, double P)
{
    double rho     = par[0];
    double eta     = par[1];
    double epsilon = par[2];
    double delta   = par[3];

    double maxRate = 0.0;

    for (int ix = 1; ix <= numberOfBinsX; ++ix) {
        double xC = xMinPlot + (ix - 0.5) * (xMaxPlot - xMinPlot) / numberOfBinsX;

        for (int iy = 1; iy <= numberOfBinsCosTheta; ++iy) {
            double cC = cosThetaMin + (iy - 0.5) * (cosThetaMax - cosThetaMin) / numberOfBinsCosTheta;

            double val =
                xC * xC *
                (3.0 * (1.0 - xC)
                 + 2.0 * rho * (4.0 * xC / 3.0 - 1.0)
                 + 3.0 * eta * x0 * (1.0 - xC) / xC
                 + P * epsilon * cC *
                       (1.0 - xC + (2.0 / 3.0) * delta * (4.0 * xC - 3.0)));

            if (val > maxRate) maxRate = val;
        }
    }
    return maxRate;
}

//==============================================================
// Run one P on GPU
//==============================================================
void RunOneP_GPU(long long N,
                 const double* par,
                 double P,
                 const std::string& outDir,
                 unsigned long long seedBase)
{
    const std::string tag      = FormatPTag(P);
    const std::string rootFile = outDir + "/MC_P_" + tag + ".root";
    const std::string pdfFile  = outDir + "/MC_P_" + tag + ".pdf";
    const std::string hname    = "HistogramRes_P_" + tag;
    // Important part: load the target-shape distribution used for inverse sampling.
    ZCdfTable tab = LoadZCdfTable("/home/tom/Mu3e/Photos/targtet shape/tom_distr_target.root", "target_z_dist");

    // Important part: build the rejection cap for the Michel sampling step.
    double maxAllowedRate = findMaximumRate_host(par, P);
    float maxAllowedRate_f = (float)maxAllowedRate;
    std::cout << "[P=" << P << "] maxAllowedRate=" << maxAllowedRate << "\n";

    const int histSize = numberOfBinsX * numberOfBinsCosTheta;

    unsigned long long* d_accepted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_accepted, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_accepted, 0, sizeof(unsigned long long)));

    unsigned long long* d_trials = nullptr;
    CUDA_CHECK(cudaMalloc(&d_trials, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_trials, 0, sizeof(unsigned long long)));

    const int threadsPerBlock = 256;
    const int blocks          = 1360;
    const int nStates         = blocks * threadsPerBlock;

    unsigned int* d_final = nullptr;
    CUDA_CHECK(cudaMalloc(&d_final, (size_t)histSize * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_final, 0, (size_t)histSize * sizeof(unsigned int)));
    float* d_zEdges = nullptr;
    float* d_cdfZ   = nullptr;
    int Nz = tab.Nz;

    CUDA_CHECK(cudaMalloc(&d_zEdges, (size_t)(tab.Nz + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cdfZ,   (size_t)(tab.Nz + 1) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_zEdges, tab.zEdges.data(),
                          (size_t)(tab.Nz + 1) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cdfZ, tab.cdfZ.data(),
                          (size_t)(tab.Nz + 1) * sizeof(float), cudaMemcpyHostToDevice));

    const size_t sharedBytes = (size_t)histSize * sizeof(unsigned int);

    curandStatePhilox4_32_10_t* d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, (size_t)nStates * sizeof(curandStatePhilox4_32_10_t)));

    unsigned long long seed = seedBase;
    seed ^= (unsigned long long)((P + 1.234567) * 1e9);
    seed += 0x9e3779b97f4a7c15ULL;

    init_rng<<<blocks, threadsPerBlock>>>(d_states, nStates, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int nTrialsPerThread = 4096;
    const int reportEvery      = 50;

    unsigned long long h_accepted = 0ULL;
    unsigned long long h_trials   = 0ULL;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    double kernelSecondsTotal = 0.0;

    int launchCount = 0;
    auto t0 = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaEventRecord(ev0));
    while (true) {
        generate_histogram_chunk<<<blocks, threadsPerBlock, sharedBytes>>>(
            d_final, d_accepted, d_trials, N,
            d_states, nStates,
            d_zEdges, d_cdfZ, Nz,
            maxAllowedRate_f,
            nTrialsPerThread,
            par[0], par[1], par[2], par[3], P
        );
        CUDA_CHECK(cudaGetLastError());

        ++launchCount;
        if (launchCount % reportEvery == 0) {
            CUDA_CHECK(cudaMemcpy(&h_accepted, d_accepted,
                                  sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "[P=" << std::fixed << std::setprecision(1) << P << "] ";
            PrintProgress(h_accepted, (unsigned long long)N, elapsed);
            if (h_accepted >= (unsigned long long)N) break;
        }
    }

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    kernelSecondsTotal = 1e-3 * ms;

    CUDA_CHECK(cudaMemcpy(&h_accepted, d_accepted,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_trials, d_trials,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    std::cout << "[P=" << std::fixed << std::setprecision(1) << P << "] "
              << "trials = " << h_trials
              << ", accepted = " << h_accepted;
    if (h_accepted > 0ULL) {
        std::cout << ", trials/accepted = " << std::setprecision(3)
                  << (double)h_trials / (double)h_accepted;
    } else {
        std::cout << ", trials/accepted = NaN";
    }
    std::cout << "\n";

    std::cout << "[P=" << std::fixed << std::setprecision(1) << P << "] "
              << "kernel-only time = " << std::setprecision(3) << kernelSecondsTotal << " s";
    if (kernelSecondsTotal > 0.0) {
        std::cout << ", rate = " << std::setprecision(1)
                  << (double)h_accepted / kernelSecondsTotal << " accepted/s";
    } else {
        std::cout << ", rate = NaN accepted/s";
    }
    std::cout << "\n";

    std::vector<unsigned int> h_hist(histSize, 0U);
    CUDA_CHECK(cudaMemcpy(h_hist.data(), d_final,
                          (size_t)histSize * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    //==========================================================
    // Convenience and graph output
    // This plotting/output block is for inspection and presentation.
    // It does not change the Monte Carlo result.
    // Parts of this graph/output setup were organised with LLM help.
    //==========================================================
    TH2F hist(hname.c_str(),
              "Monte Carlo (GPU, striped) after acceptance",
              numberOfBinsX, xMinPlot, xMaxPlot,
              numberOfBinsCosTheta, cosThetaMin, cosThetaMax);

    for (int iy0 = 0; iy0 < numberOfBinsCosTheta; ++iy0) {
        for (int ix0 = 0; ix0 < numberOfBinsX; ++ix0) {
            const int idx = ix0 + numberOfBinsX * iy0;
            hist.SetBinContent(ix0 + 1, iy0 + 1, (double)h_hist[idx]);
        }
    }

    hist.GetXaxis()->SetTitle("x = E_{e} / E_{max}");
    hist.GetYaxis()->SetTitle("cos#theta");
    hist.GetZaxis()->SetTitle("Events");
    hist.GetXaxis()->CenterTitle();
    hist.GetYaxis()->CenterTitle();
    hist.GetZaxis()->CenterTitle();
    hist.GetXaxis()->SetTitleOffset(1.2);
    hist.GetYaxis()->SetTitleOffset(1.3);
    hist.GetZaxis()->SetTitleOffset(1.4);
    hist.GetYaxis()->SetRangeUser(-1.0, 1.0);
    hist.GetXaxis()->SetRangeUser(0.0, 1.1);

    TFile f(rootFile.c_str(), "RECREATE");
    hist.Write();

    TCanvas c("c", "GPU Monte Carlo", 900, 700);
    hist.SetStats(false);
    hist.Draw("COLZ");

    TLatex latex;
    latex.SetNDC(true);
    latex.SetTextSize(0.04);
    latex.DrawLatex(0.15, 0.92, Form("Muon polarisation P = %.1f", P));
    c.SaveAs(pdfFile.c_str());

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    cudaFree(d_final);
    cudaFree(d_states);
    cudaFree(d_trials);
    cudaFree(d_accepted);
    cudaFree(d_zEdges);
    cudaFree(d_cdfZ);
}

//==============================================================
// Scan all P values
//==============================================================
void ScanPolarization_GPU(long long N,
                          const double* par,
                          const std::string& outDir,
                          unsigned long long seedBase)
{
    gSystem->mkdir(outDir.c_str(), true);

    for (int i = 0; i <= 20; ++i) {
        double P = -1.0 + 0.1 * i;
        RunOneP_GPU(N, par, P, outDir, seedBase);
    }
}

//==============================================================
// Main
//==============================================================
int main(int argc, char** argv)
{
    gROOT->SetBatch(kTRUE);

    //==========================================================
    // Convenience only
    // Command-line parsing and run configuration do not change the physics.
    //==========================================================
    // CLI settings
    // - --n    : number of accepted events
    // - --out  : output directory
    // - --seed : base seed for RNG
    //==========================================================
    long long Naccepted = 100000000000LL; // 1e11
    std::string outDir = "/home/tom/Mu3e/Photos/MC 3 layer photos/Reference graph";
    unsigned long long seedBase = 123456789ULL;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        if ((a == "--n" || a == "--nmu" || a == "--N") && i + 1 < argc) {
            Naccepted = std::stoll(argv[++i]);
        }/* */
        else if ((a == "--out" || a == "--outdir") && i + 1 < argc) {
            outDir = argv[++i];
        }
        else if (a == "--seed" && i + 1 < argc) {
            seedBase = std::stoull(argv[++i]);
        }
        else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [--n <Naccepted>] [--out <outDir>] [--seed <seed>]\n"
                << "  --n     number of accepted events (default 1e9)\n"
                << "  --out   output directory\n"
                << "  --seed  base RNG seed (default 123456789)\n";
            return 0;
        }
    }

    // Copy constants to GPU once
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMin, &cosThetaMin, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMax, &cosThetaMax, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMinPlot,    &xMinPlot,    sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMaxPlot,    &xMaxPlot,    sizeof(double)));
    float cosThetaMin_f = (float)cosThetaMin;
    float cosThetaMax_f = (float)cosThetaMax;
    float xMinPlot_f    = (float)xMinPlot;
    float xMaxPlot_f    = (float)xMaxPlot;
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMin_f, &cosThetaMin_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cosThetaMax_f, &cosThetaMax_f, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMinPlot_f,    &xMinPlot_f,    sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xMaxPlot_f,    &xMaxPlot_f,    sizeof(float)));

    double parSM[4] = {0.75, 0.0, 1.0, 0.75};

    gSystem->mkdir(outDir.c_str(), true);

    std::cout << "Naccepted = " << Naccepted << "\n";
    std::cout << "outDir    = " << outDir << "\n";
    std::cout << "seedBase  = " << seedBase << "\n";    

    ScanPolarization_GPU(Naccepted, parSM, outDir, seedBase);
    return 0;
}
