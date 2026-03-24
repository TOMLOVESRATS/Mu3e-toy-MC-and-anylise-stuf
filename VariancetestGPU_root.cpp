#include "TCanvas.h"
#include "TFile.h"
#include "TF1.h"
#include "TH1D.h"
#include "TH2.h"
#include "TKey.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TSystem.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// File guide
//
// Important analysis / fit section:
// 1) AfbModel(...)
// 2) MakeAfbFBVsX(...)
// 3) the main fit loop where fP fits P with fixed global K
//
// What this file does:
// - scans the produced MC ROOT files for many runs
// - builds A_FB(x) from each 2D histogram
// - loads the reference <cos(theta)>(x) template and zero-bias A_FB template
// - fits P for each run with a fixed global K
// - writes per-run fit results, summary text, and check plots
//
// Convenience-only section:
// - filename parsing, recursive discovery, CLI parsing, text output,
//   and directory handling.
//
// Graph / presentation-only section:
// - the PDF plots and cosmetics are only for visual checking/report output.
//
// LLM-assisted / readability-only section:
// - the convenience/plot organization and readability cleanup are the parts
//   marked below as non-core.
//
// Existing reference-file usage and directory paths are intentionally kept.
// ============================================================

// One discovered ROOT run input file.
struct RunInput {
    std::string path;
    std::string pTag;
    double pUsed = std::numeric_limits<double>::quiet_NaN();
    int run = -1;
};

// Bundle of A_FB/F/B histograms from one 2D MC histogram.
struct AfbComponents {
    TH1D* hAfb = nullptr;
    TH1D* hF = nullptr;
    TH1D* hB = nullptr;
};

// Global avg-cos template used by AfbModel in ROOT TF1 fits.
static TH1D* gAvgCos = nullptr;
static TH1D* gAfbZeroBias = nullptr;

// Zero-padded run tag: 7 -> 0007.
static std::string FormatRunTag(int run)
{
    std::ostringstream ss;
    ss << std::setfill('0') << std::setw(4) << run;
    return ss.str();
}

// Filename tag format used by VariancetestGPU.cu (e.g. -0.9 -> m0p9).
static std::string FormatPTag(double P)
{
    std::ostringstream tag;
    tag << std::fixed << std::setprecision(1) << P;
    std::string s = tag.str();
    for (char& c : s) if (c == '.') c = 'p';
    if (!s.empty() && s[0] == '-') s[0] = 'm';
    return s;
}

// HardMEGWay-style fit model: A_FB(x) = K * x * P * <cos(theta)>(x).
static double AfbModel(double* xx, double* par)
{
    if (!gAvgCos) return 0.0;
    const double E = xx[0];
    const int bin = gAvgCos->FindBin(E);
    if (bin < 1 || bin > gAvgCos->GetNbinsX()) return 0.0;
    const double avgCos = gAvgCos->GetBinContent(bin);
    // par[0] = P, par[1] = K
    return 8.224 * E * par[0] * avgCos;
}

// get precomputed avg-cos template from reference file
static TH1D* LoadAvgCosTemplate(const std::string& avgCosDir, const std::string& pTag)
{
    const std::string filePath = avgCosDir + "/AvgCostheta_vs_x_P_" + pTag + ".root";
    TFile* f = TFile::Open(filePath.c_str(), "READ");
    if (!f || f->IsZombie()) {
        if (f) { f->Close(); delete f; }
        return nullptr;
    }

    TH1D* h = dynamic_cast<TH1D*>(f->Get("AvgCosTheta_vs_x"));
    if (!h) {
        h = dynamic_cast<TH1D*>(f->Get(("hAvgCosTheta_vs_x_P_" + pTag).c_str()));
    }

    TH1D* out = nullptr;
    if (h) {
        out = dynamic_cast<TH1D*>(h->Clone(("AvgCos_template_" + pTag).c_str()));
        if (out) out->SetDirectory(nullptr);
    }

    f->Close();
    delete f;
    return out;
}

static TH1D* LoadAfbTemplate(const std::string& afbFile)
{
    TFile* f = TFile::Open(afbFile.c_str(), "READ");
    if (!f || f->IsZombie()) {
        if (f) { f->Close(); delete f; }
        return nullptr;
    }

    TH1D* h = dynamic_cast<TH1D*>(f->Get("Afb_vs_x"));
    TH1D* out = nullptr;
    if (h) {
        out = dynamic_cast<TH1D*>(h->Clone("Afb_zero_bias_template"));
        if (out) out->SetDirectory(nullptr);
    }

    f->Close();
    delete f;
    return out;
}

static void SubtractAfbZeroBias(TH1D* hAfb)
{
    if (!hAfb || !gAfbZeroBias) return;
    if (hAfb->GetNbinsX() != gAfbZeroBias->GetNbinsX()) return;
    hAfb->Add(gAfbZeroBias, -1.0);
}

// ------------------------------------------------------------
// Convenience + file-discovery section
// LLM-assisted/readability-only: these helpers are plumbing, not the fit.
// ------------------------------------------------------------

// Parse p-tag from filename fragment, e.g. m0p5 -> -0.5.
static bool ParsePTag(const std::string& pTag, double& p)
{
    if (pTag.empty()) return false;
    std::string s = pTag;
    if (s[0] == 'm') s[0] = '-';
    for (char& c : s) if (c == 'p') c = '.';
    try {
        p = std::stod(s);
    } catch (...) {
        return false;
    }
    return true;
}

static bool IsWithinDir(const fs::path& path, const fs::path& dir)
{
    const fs::path normPath = path.lexically_normal();
    const fs::path normDir = dir.lexically_normal();

    auto pIt = normPath.begin();
    auto dIt = normDir.begin();
    for (; dIt != normDir.end(); ++dIt, ++pIt) {
        if (pIt == normPath.end() || *pIt != *dIt) return false;
    }
    return true;
}

// Parse run input metadata from filename: MC_P_<tag>_run_<NNNN>.root
static bool ParseRunInputFromFilename(const std::string& filename, RunInput& out)
{
    const std::string prefix = "MC_P_";
    const std::string mid = "_run_";
    const std::string suffix = ".root";

    if (filename.size() <= prefix.size() + mid.size() + suffix.size()) return false;
    if (filename.rfind(prefix, 0) != 0) return false;
    if (filename.size() < suffix.size() ||
        filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return false;
    }

    const size_t posMid = filename.find(mid, prefix.size());
    if (posMid == std::string::npos) return false;

    const std::string pTag = filename.substr(prefix.size(), posMid - prefix.size());
    const std::string runPart = filename.substr(
        posMid + mid.size(),
        filename.size() - (posMid + mid.size()) - suffix.size());
    if (runPart.empty()) return false;

    double p = std::numeric_limits<double>::quiet_NaN();
    if (!ParsePTag(pTag, p)) return false;

    int run = -1;
    try {
        run = std::stoi(runPart);
    } catch (...) {
        return false;
    }

    out.pTag = pTag;
    out.pUsed = p;
    out.run = run;
    return true;
}

// Discover all run ROOT files from a directory, optionally filtering by P.
static bool DiscoverRunInputs(const std::string& mcDir,
                              bool filterByP,
                              double pFilter,
                              std::vector<RunInput>& runs)
{
    runs.clear();
    std::error_code ec;
    if (!fs::exists(mcDir, ec) || !fs::is_directory(mcDir, ec)) return false;

    std::map<std::string, fs::path> canonicalDirsByTag;
    const fs::path mcRoot = fs::path(mcDir);
    if (filterByP) {
        const std::string pTag = FormatPTag(pFilter);
        const fs::path canonicalDir = mcRoot / ("P_" + pTag);
        if (fs::exists(canonicalDir, ec) && fs::is_directory(canonicalDir, ec)) {
            canonicalDirsByTag[pTag] = canonicalDir;
        }
    } else {
        for (const auto& entry : fs::directory_iterator(mcRoot, ec)) {
            if (ec) break;
            if (!entry.is_directory()) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind("P_", 0) != 0 || name.size() <= 2) continue;
            canonicalDirsByTag[name.substr(2)] = entry.path();
        }
    }

    for (const auto& entry : fs::recursive_directory_iterator(mcDir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string filename = entry.path().filename().string();
        RunInput r;
        if (!ParseRunInputFromFilename(filename, r)) continue;
        if (filterByP && std::abs(r.pUsed - pFilter) > 1e-12) continue;
        const auto it = canonicalDirsByTag.find(r.pTag);
        if (it != canonicalDirsByTag.end() && !IsWithinDir(entry.path(), it->second)) continue;
        r.path = entry.path().string();
        runs.push_back(r);
    }

    std::sort(runs.begin(), runs.end(),
              [](const RunInput& a, const RunInput& b) {
                  if (a.pUsed != b.pUsed) return a.pUsed < b.pUsed;
                  if (a.run != b.run) return a.run < b.run;
                  return a.path < b.path;
              });
    return true;
}

// Load TH2 from file by preferred name, otherwise first TH2 found.
static TH2* LoadTH2FromFile(const std::string& filePath, const std::string& preferredHistName)
{
    TFile* f =TFile::Open(filePath.c_str(), "READ");
    if (!f || f->IsZombie()) {
        if (f) { f->Close(); delete f; }
        return nullptr;
    }

    TH2* h2 = nullptr;
    if (!preferredHistName.empty()) {
        h2 = dynamic_cast<TH2*>(f->Get(preferredHistName.c_str()));
    }
    if (!h2) {
        TIter next(f->GetListOfKeys());
        while (TKey* key = dynamic_cast<TKey*>(next())) {
            TObject* obj = f->Get(key->GetName());
            TH2* cand = dynamic_cast<TH2*>(obj);
            if (cand) {
                h2 = cand;
                break;
            }
        }
    }

    TH2* clone = nullptr;
    if (h2) {
        clone = dynamic_cast<TH2*>(h2->Clone());
        if (clone) clone->SetDirectory(nullptr);
    }

    f->Close();
    delete f;
    return clone;
}

// ------------------------------------------------------------
// Important analysis section: build A_FB(x), F(x), and B(x)
// ------------------------------------------------------------

// Build Afb/F/B from a 2D histogram in (x, cos(theta))\.
static AfbComponents MakeAfbFBVsX(const TH2* h2, const std::string& baseName)
{
    AfbComponents out;
    if (!h2) return out;

    const int nx = h2->GetNbinsX();
    const int ny = h2->GetNbinsY();
    // Convenience-only allocation and naming for the output histograms.
    out.hAfb = new TH1D((baseName + "_Afb").c_str(), "A_{FB}(x); x=E/E_{end}; A_{FB}",
                        nx, h2->GetXaxis()->GetXmin(), h2->GetXaxis()->GetXmax());
    out.hF = new TH1D((baseName + "_F").c_str(), "Forward counts F(x); x=E/E_{end}; F",
                      nx, h2->GetXaxis()->GetXmin(), h2->GetXaxis()->GetXmax());
    out.hB = new TH1D((baseName + "_B").c_str(), "Backward counts B(x); x=E/E_{end}; B",
                      nx, h2->GetXaxis()->GetXmin(), h2->GetXaxis()->GetXmax());
    out.hAfb->SetDirectory(nullptr);
    out.hF->SetDirectory(nullptr);
    out.hB->SetDirectory(nullptr);
    for (int ix = 1; ix <= nx; ++ix) {
        double F = 0.0;
        double B = 0.0;
        for (int iy = 1; iy <= ny; ++iy) {
            const double c = h2->GetYaxis()->GetBinCenter(iy);
            const double w = h2->GetBinContent(ix, iy);
            if (w <= 0.0) continue;
            if (c > 0.0) F += w;
            else if (c < 0.0) B += w;
        }

        const double N = F + B;
        double A = 0.0;
        double eA = 0.0;
        if (N > 0.0) {
            A = (F - B) / N;
            const double FB = F * B;
            if (FB > 0.0) eA = 2.0 * std::sqrt(FB) / std::pow(N, 1.5);
        }

        out.hAfb->SetBinContent(ix, A);
        out.hAfb->SetBinError(ix, eA);
        out.hF->SetBinContent(ix, F);
        out.hF->SetBinError(ix, (F > 0.0) ? std::sqrt(F) : 0.0);
        out.hB->SetBinContent(ix, B);
        out.hB->SetBinError(ix, (B > 0.0) ? std::sqrt(B) : 0.0);
    }

    out.hAfb->SetStats(false);
    out.hF->SetStats(false);
    out.hB->SetStats(false);
    return out;
}

// Compute mean and sigma (population std).

static void ComputeMeanSigma(const std::vector<double>& vals, double& mean, double& sigma)
{
    mean = std::numeric_limits<double>::quiet_NaN();
    sigma = std::numeric_limits<double>::quiet_NaN();
    if (vals.empty()) return;
    double sum = 0.0;
    double sumsq = 0.0;
    for (double v : vals) {
        sum += v;
        sumsq += v * v;
    }
    mean = sum / (double)vals.size();
    double var = (sumsq / (double)vals.size()) - mean * mean;
    if (var < 0.0) var = 0.0;
    sigma = std::sqrt(var);
}


// ------------------------------------------------------------
// Main workflow section: read runs, build A_FB(x), and fit P
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    gROOT->SetBatch(kTRUE);

    // Edit these defaults here if you want to drive runs from the source file
    // instead of passing CLI arguments.
    std::string mcDir = "/home/tom/Mu3e/Photos/MC 3 layer photos/Variancce test";
    std::string avgCosDir = "/home/tom/Mu3e/Photos/MC 3 layer photos/cos refer";
    std::string afbZeroDir = "/home/tom/Mu3e/Photos/MC 3 layer photos/3-layer afbvsx";
    std::string outDir = "/home/tom/Mu3e/Photos/MC 3 layer photos/Variancce test/Root analyse";
    double Ptrue = std::numeric_limits<double>::quiet_NaN();
    bool filterRunsByP = false; // set true to process only the Ptrue runs

    // Convenience-only CLI parsing.
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--mc-dir") && i + 1 < argc) {
            mcDir = argv[++i];
        } else if ((a == "--avgcos-dir") && i + 1 < argc) {
            avgCosDir = argv[++i];
        } else if ((a == "--afb-zero-dir") && i + 1 < argc) {
            afbZeroDir = argv[++i];
        } else if ((a == "--out" || a == "--outdir") && i + 1 < argc) {
            outDir = argv[++i];
        } else if ((a == "--p" || a == "--P") && i + 1 < argc) {
            Ptrue = std::stod(argv[++i]);
            filterRunsByP = true;
        } else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: " << argv[0] << " [--mc-dir <dir>] [--avgcos-dir <dir>] [--afb-zero-dir <dir>] [--out <outDir>] [--p <Ptrue>]\n"
                << "  --mc-dir  VariancetestGPU output dir (scans MC_P_*_run_*.root recursively)\n"
                << "  --avgcos-dir  directory with AvgCostheta_vs_x_P_*.root files\n"
                << "  --afb-zero-dir  directory with raw Afb_vs_x_P_0p0.root for post-build subtraction\n"
                << "  --out     output directory\n"
                << "  --p       optional P filter\n"
                << "  Default behavior can also be edited directly in this .cpp file.\n";
            return 0;
        }
    }

    gSystem->mkdir(outDir.c_str(), true);

    std::vector<RunInput> runs;
    // Convenience-only scan and text-output setup.
    if (!DiscoverRunInputs(mcDir, filterRunsByP, Ptrue, runs)) {
        std::cerr << "ERROR: failed to scan ROOT directory: " << mcDir << "\n";
        return 1;
    }
    if (runs.empty()) {
        std::cerr << "ERROR: no matching ROOT runs found in " << mcDir << "\n";
        return 1;
    }
    std::cout << "Discovered " << runs.size() << " run files from recursive scan\n";

    const std::string outRuns = outDir + "/pfit_runs_rootfit.txt";
    const std::string outSummary = outDir + "/pfit_summary_rootfit.txt";

    std::ofstream out(outRuns);
    if (!out) {
        std::cerr << "ERROR: cannot write " << outRuns << "\n";
        return 1;
    }
    out << "run,p_used,p_fit,p_fit_err,fit_status,mc_file\n";

    // IMPORTANT: this loop is the real fit workflow.
    std::vector<double> pvals;
    std::map<double, std::vector<double>> pvalsByP;
    std::map<double, int> totalRunsByP;
    bool savedExamplePlots = false;
    std::string outAfbToy;
    std::string outAfbFit;
    const double Kglobal = 8.224;
    bool printedFixedK = false;
    const std::string afbZeroFile = afbZeroDir + "/Afb_vs_x_P_0p0.root";
    gAfbZeroBias = LoadAfbTemplate(afbZeroFile);
    if (!gAfbZeroBias) {
        std::cerr << "WARN: failed to load raw P=0 Afb baseline from: " << afbZeroFile
                  << "\n      Proceeding without post-build Afb subtraction.\n";
    }

    for (const auto& r : runs) {
        const double pUsed = r.pUsed;
        totalRunsByP[pUsed] += 1;
        const std::string pTag = r.pTag;
        const std::string runTag = FormatRunTag(r.run);
        const std::string mcFile = r.path;
        const std::string histName = "HistogramRes_P_" + pTag + "_run_" + runTag;

        TH2* h2 = LoadTH2FromFile(mcFile, histName);
        if (!h2) {
            out << r.run << "," << std::fixed << std::setprecision(6) << pUsed
                << ",NaN,NaN,3," << mcFile << "\n";
            continue;
        }

        AfbComponents comp = MakeAfbFBVsX(h2, "run_" + std::to_string(r.run));
        delete h2;
        SubtractAfbZeroBias(comp.hAfb);

        if (!comp.hAfb) {
            out << r.run << "," << std::fixed << std::setprecision(6) << pUsed
                << ",NaN,NaN,4," << mcFile << "\n";
            delete comp.hAfb;
            delete comp.hF;
            delete comp.hB;
            continue;
        }

        TH1D* hAvg = LoadAvgCosTemplate(avgCosDir, pTag);
        if (!hAvg) {
            out << r.run << "," << std::fixed << std::setprecision(6) << pUsed
                << ",NaN,NaN,5," << mcFile << "\n";
            delete comp.hAfb;
            delete comp.hF;
            delete comp.hB;
            continue;
        }
        if (std::abs(pUsed) <= 1e-12) {
            out << r.run << "," << std::fixed << std::setprecision(6) << pUsed
                << ",NaN,NaN,6," << mcFile << "\n";
            delete comp.hAfb;
            delete comp.hF;
            delete comp.hB;
            delete hAvg;
            continue;
        }
        hAvg->Scale(1.0 / pUsed);

        const double xMinRaw = comp.hAfb->GetXaxis()->GetXmin();
        const double xMax = comp.hAfb->GetXaxis()->GetXmax();
        const double fitXMin = std::max(0.4, xMinRaw);

        gAvgCos = hAvg;
        if (!printedFixedK) {
            std::cout << "Using fixed global K = " << Kglobal << "\n";
            printedFixedK = true;
        }

        TF1 fP("fP", AfbModel, fitXMin, xMax, 2);
        fP.FixParameter(1, Kglobal);
        fP.SetParameter(0, 0.0);
        const int fitStatus = comp.hAfb->Fit(&fP, "WQRN");
        const double pFit = fP.GetParameter(0);
        const double pErr = fP.GetParError(0);

        out << r.run << "," << std::fixed << std::setprecision(6) << pUsed
            << "," << std::setprecision(8) << pFit
            << "," << pErr << "," << fitStatus << "," << mcFile << "\n";

        if (fitStatus == 0 && std::isfinite(pFit)) {
            pvals.push_back(pFit);
            pvalsByP[pUsed].push_back(pFit);

            // Graph / presentation-only section:
            // plotting is only for visual check / report-friendly output.
            // This is part of the readability/reporting cleanup, not the fit.
            if (!savedExamplePlots) {
                outAfbToy = outDir + "/Afb_oneToy_vs_x.pdf";
                outAfbFit = outDir + "/Afb_fit_overlay.pdf";

                TCanvas cAtoy("c_afb_onetoy", "Afb toy", 900, 650);
                comp.hAfb->SetMarkerStyle(20);
                comp.hAfb->SetMarkerSize(0.8);
                comp.hAfb->SetTitle(Form("One run A_{FB}(x) (run %d); x; A_{FB}", r.run));
                comp.hAfb->Draw("E1");
                cAtoy.SaveAs(outAfbToy.c_str());

                fP.SetLineColor(2);
                fP.SetLineWidth(2);
                TCanvas cFit("c_afb_fit_overlay", "Fit overlay", 900, 650);
                comp.hAfb->SetTitle(Form("A_{FB}(x) fit overlay (run %d); x; A_{FB}", r.run));
                comp.hAfb->Draw("E1");
                fP.Draw("SAME");
                cFit.SaveAs(outAfbFit.c_str());

                savedExamplePlots = true;
            }
        }

        gAvgCos = nullptr;

        delete comp.hAfb;
        delete comp.hF;
        delete comp.hB;
        delete hAvg;
    }

    delete gAfbZeroBias;
    gAfbZeroBias = nullptr;
    out.close();


    double meanAll = 0.0;
    double sigmaAll = 0.0;
    ComputeMeanSigma(pvals, meanAll, sigmaAll);
    // Convenience + graph/report output section.
    std::ofstream summary(outSummary);
    if (summary) {
        summary << "Ptrue,mean_pfit,sigma_pfit,valid_runs,total_runs\n";
        for (const auto& kv : totalRunsByP) {
            const double p = kv.first;
            const int totalCount = kv.second;
            const auto it = pvalsByP.find(p);
            const std::vector<double> emptyVals;
            const std::vector<double>& vals = (it != pvalsByP.end()) ? it->second : emptyVals;
            double mean = std::numeric_limits<double>::quiet_NaN();
            double sigma = std::numeric_limits<double>::quiet_NaN();
            ComputeMeanSigma(vals, mean, sigma);
            const int validCount = (int)vals.size();
            summary << std::fixed << std::setprecision(8)
                    << p << "," << mean << "," << sigma << ","
                    << validCount << "," << totalCount << "\n";
        }
    }

    // mostly presentation output
    std::string outPhatHistAll;
    std::vector<std::string> outPhatHistsByP;
    if (!pvals.empty()) {
        double vmin = *std::min_element(pvals.begin(), pvals.end());
        double vmax = *std::max_element(pvals.begin(), pvals.end());
        const double pad = (vmax > vmin) ? 0.1 * (vmax - vmin) : 0.1;
        TH1D h("h_phat_valid", "#hat{P} distribution over valid runs; #hat{P}; Entries", 60, vmin - pad, vmax + pad);
        for (double v : pvals) h.Fill(v);
        h.SetTitle(Form("#hat{P} over valid runs (N_{valid}=%zu); #hat{P}; Entries", pvals.size()));

        TCanvas c1("c1", "P_fit distribution", 900, 650);
        h.SetStats(false);
        h.Draw("HIST");
        TLatex lat;
        lat.SetNDC(true);
        lat.SetTextSize(0.04);
        lat.DrawLatex(0.15, 0.92, Form("mean = %.6f", meanAll));
        lat.DrawLatex(0.15, 0.86, Form("sigma = %.6f", sigmaAll));
        outPhatHistAll = outDir + "/Phat_hist_all_valid_runs_rootfit.pdf";
        c1.SaveAs(outPhatHistAll.c_str());
    }

    for (const auto& kv : pvalsByP) {
        const double p = kv.first;
        const auto& vals = kv.second;
        if (vals.empty()) continue;
        double mean = 0.0;
        double sigma = 0.0;
        ComputeMeanSigma(vals, mean, sigma);

        double vmin = *std::min_element(vals.begin(), vals.end());
        double vmax = *std::max_element(vals.begin(), vals.end());
        const double pad = (vmax > vmin) ? 0.1 * (vmax - vmin) : 0.1;
        const std::string pTag = FormatPTag(p);
        TH1D h(("h_phat_valid_" + pTag).c_str(),
               Form("#hat{P} distribution (P=%.1f); #hat{P}; Entries", p),
               60, vmin - pad, vmax + pad);
        for (double v : vals) h.Fill(v);

        TCanvas c("c_phat_per_p", "P_fit distribution per P", 900, 650);
        h.SetStats(false);
        h.Draw("HIST");
        TLatex lat;
        lat.SetNDC(true);
        lat.SetTextSize(0.04);
        lat.DrawLatex(0.15, 0.92, Form("P_{true} = %.3f", p));
        lat.DrawLatex(0.15, 0.86, Form("mean = %.6f", mean));
        lat.DrawLatex(0.15, 0.80, Form("sigma = %.6f", sigma));
        const std::string outOne = outDir + "/Phat_hist_P_" + pTag + "_rootfit.pdf";
        c.SaveAs(outOne.c_str());
        outPhatHistsByP.push_back(outOne);
    }

    std::cout << "Wrote:\n";
    std::cout << "  " << outRuns << "\n";
    std::cout << "  " << outSummary << "\n";
    if (savedExamplePlots) {
        std::cout << "  " << outAfbToy << "\n";
        std::cout << "  " << outAfbFit << "\n";
    }
    if (!outPhatHistAll.empty()) {
        std::cout << "  " << outPhatHistAll << "\n";
    }
    for (const auto& p : outPhatHistsByP) {
        std::cout << "  " << p << "\n";
    }
    return 0;
}
