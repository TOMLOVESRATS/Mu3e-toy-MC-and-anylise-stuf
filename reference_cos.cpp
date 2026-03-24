#include "TCanvas.h"
#include "TFile.h"
#include "TH1D.h"
#include "TH2.h"
#include "TKey.h"
#include "TLatex.h"
#include "TROOT.h"
#include "TSystem.h"

#include "AverageCostheta.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// File guide
//
// Important analysis section:
// - ProcessOneFile(...) is the main workflow in this file.
// - AverageCostheta::AvgcosthetavsX(...) is the core calculation used here.
//
// What this file does:
// - reads already-produced MC histograms
// - computes the reference <cos(theta)>(x) histogram for each input file
// - writes the reference ROOT histogram and a PDF plot
// - prepares the templates used later by the fit code
//
// Convenience-only section:
// - filename parsing, P-tag decoding, file discovery, CLI parsing, and mkdir.
//
// Graph / presentation-only section:
// - plot styling, canvas creation, PDF export, and user-facing log output.
//
// LLM-assisted / readability-only section:
// - the section labels and the convenience/presentation annotations below are
//   documentation/readability work, not part of the physics calculation.
//
// Existing "Reference graph" paths are intentionally kept in place.
// ============================================================

// ------------------------------------------------------------
// Convenience section: filename parsing and file discovery
// ------------------------------------------------------------
static bool ParsePTagFromFilename(const std::string& filename, std::string& pTag)
{
    const std::string prefix = "MC_P_";
    const std::string suffix = ".root";
    if (filename.rfind(prefix, 0) != 0) return false;
    if (filename.size() <= prefix.size() + suffix.size()) return false;
    if (filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) return false;

    pTag = filename.substr(prefix.size(), filename.size() - prefix.size() - suffix.size());
    return !pTag.empty();
}

static double DecodePTag(const std::string& pTag, bool& ok)
{
    ok = false;
    std::string s = pTag;
    if (!s.empty() && s[0] == 'm') s[0] = '-';
    for (char& c : s) {
        if (c == 'p') c = '.';
    }
    try {
        const double p = std::stod(s);
        ok = true;
        return p;
    } catch (...) {
        return 0.0;
    }
}

static TH2* LoadTH2FromFile(const std::string& filePath, const std::string& preferredHistName)
{
    TFile* f = TFile::Open(filePath.c_str(), "READ");
    if (!f || f->IsZombie()) {
        if (f) {
            f->Close();
            delete f;
        }
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

static std::vector<fs::path> DiscoverInputFiles(const std::string& inDir)
{
    std::vector<fs::path> files;
    std::error_code ec;
    if (!fs::exists(inDir, ec) || !fs::is_directory(inDir, ec)) return files;

    for (const auto& entry : fs::directory_iterator(inDir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;
        const std::string name = entry.path().filename().string();
        std::string pTag;
        if (!ParsePTagFromFilename(name, pTag)) continue;
        files.push_back(entry.path());
    }

    std::sort(files.begin(), files.end());
    return files;
}

// ------------------------------------------------------------
// Important workflow section: build reference <cos(theta)>(x)
// ------------------------------------------------------------
static bool ProcessOneFile(const fs::path& inFile, const std::string& outDir)
{
    const std::string filename = inFile.filename().string();
    std::string pTag;
    if (!ParsePTagFromFilename(filename, pTag)) {
        std::cerr << "Skipping unsupported file name: " << filename << "\n";
        return false;
    }

    const std::string preferredHistName = "HistogramRes_P_" + pTag;
    TH2* h2 = LoadTH2FromFile(inFile.string(), preferredHistName);
    if (!h2) {
        std::cerr << "ERROR: could not load TH2 from " << inFile << "\n";
        return false;
    }

    TH1D* hAvg = AverageCostheta::AvgcosthetavsX(h2, "P_" + pTag);
    delete h2;
    if (!hAvg) {
        std::cerr << "ERROR: failed computing <cos(theta)> for " << inFile << "\n";
        return false;
    }
    hAvg->SetDirectory(nullptr);

    bool okP = false;
    const double pValue = DecodePTag(pTag, okP);
    if (okP) {
        hAvg->SetTitle(Form("<cos#theta>(x) for P=%.1f; x=E/E_{max}; <cos#theta>(x)", pValue));
    } else {
        hAvg->SetTitle("<cos#theta>(x); x=E/E_{max}; <cos#theta>(x)");
    }
    hAvg->SetMarkerStyle(20);
    hAvg->SetMarkerSize(0.75);
    hAvg->SetLineWidth(2);
    hAvg->SetStats(false);

    const std::string outRoot = outDir + "/AvgCostheta_vs_x_P_" + pTag + ".root";
    const std::string outPdf = outDir + "/AvgCostheta_vs_x_P_" + pTag + ".pdf";

    bool wroteRoot = false;
    {
        TFile fOut(outRoot.c_str(), "RECREATE");
        if (!fOut.IsZombie()) {
            if (hAvg->Write("AvgCosTheta_vs_x") > 0) wroteRoot = true;
            fOut.Close();
        }
    }

    // Graph / presentation-only section:
    // the canvas, labels, and PDF output below are for visual reference only.
    TCanvas c(("c_avgcos_" + pTag).c_str(), "<cos(theta)> vs x", 900, 650);
    hAvg->Draw("P");
    TLatex lat;
    lat.SetNDC(true);
    lat.SetTextSize(0.04);
    if (okP) lat.DrawLatex(0.15, 0.92, Form("Reference MC: P = %.1f", pValue));
    c.SaveAs(outPdf.c_str());
    const bool wrotePdf = fs::exists(outPdf);

    if (!wroteRoot) {
        std::cerr << "ERROR: failed to write ROOT file: " << outRoot << "\n";
    }
    if (!wrotePdf) {
        std::cerr << "ERROR: failed to write PDF file: " << outPdf << "\n";
    }

    if (wroteRoot) std::cout << "Wrote: " << outRoot << "\n";
    if (wrotePdf) std::cout << "Wrote: " << outPdf << "\n";
    delete hAvg;
    return wroteRoot && wrotePdf;
}

// ------------------------------------------------------------
// Convenience workflow section: batch over all reference files
// ------------------------------------------------------------
int main(int argc, char** argv)
{
    gROOT->SetBatch(kTRUE);

    std::string inDir = "/home/tom/Mu3e/Photos/MC 3 layer photos/Reference graph";
    std::string outDir = "/home/tom/Mu3e/Photos/MC 3 layer photos/Reference graph/Avg costheta";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--in" || a == "--input") && i + 1 < argc) {
            inDir = argv[++i];
        } else if ((a == "--out" || a == "--outdir") && i + 1 < argc) {
            outDir = argv[++i];
        } else if (a == "--help" || a == "-h") {
            std::cout << "Usage: " << argv[0] << " [--in <inputDir>] [--out <outputDir>]\n";
            return 0;
        }
    }

    gSystem->mkdir(outDir.c_str(), true);

    const std::vector<fs::path> files = DiscoverInputFiles(inDir);
    if (files.empty()) {
        std::cerr << "ERROR: no MC_P_*.root files found in " << inDir << "\n";
        return 1;
    }

    int okCount = 0;
    for (const auto& file : files) {
        if (ProcessOneFile(file, outDir)) ++okCount;
    }

    std::cout << "Processed " << okCount << " / " << files.size() << " files\n";
    return (okCount == static_cast<int>(files.size())) ? 0 : 2;
}
