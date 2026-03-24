#include "TCanvas.h"
#include "TFile.h"
#include "TF1.h"
#include "TH1D.h"
#include "TH2.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TSystem.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

const std::string kMcGpuDir =
    "/home/tom/Mu3e/Photos/MC Muon photos/MC GPU event";
const std::string kAfbInputDir =
    "/home/tom/Mu3e/Photos/MC Muon photos/MC afbvsx";
const std::string kOutputDir =
    "/home/tom/Mu3e/Photos/MC Muon photos/Theoryfit against MC";

struct FitSummary {
    double truePolarization = 0.0;
    double fittedPolarization = 0.0;
    double fittedPolarizationError = 0.0;
    double lineSlope = 0.0;
    double lineSlopeError = 0.0;
    double totalEvents = 0.0;
};

// ============================================================
// Presentation-only helpers
// These keep names and output tidy but do not change the fit result.
// ============================================================
std::string formatPolarizationTag(double polarization)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << polarization;
    std::string s = oss.str();
    for (char &c : s) {
        if (c == '.') c = 'p';
    }
    if (!s.empty() && s[0] == '-') s[0] = 'm';
    return s;
}

std::string formatSignedValue(double value, int precision)
{
    std::ostringstream oss;
    oss << std::showpos << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

std::string buildMcGpuPath(double polarization)
{
    return kMcGpuDir + "/MC_P_" + formatPolarizationTag(polarization) + ".root";
}

std::string buildMcGpuHistName(double polarization)
{
    return "HistogramRes_P_" + formatPolarizationTag(polarization);
}

std::string buildAfbPath(double polarization)
{
    return kAfbInputDir + "/Afb_vs_x_P_" + formatPolarizationTag(polarization) + ".root";
}

std::string buildOutputStem(double polarization)
{
    std::ostringstream oss;
    oss << kOutputDir
        << "/Afb_fit_mc_P_"
        << std::showpos
        << std::fixed
        << std::setprecision(1)
        << polarization;
    return oss.str();
}

void styleAfbHistogram(TH1D& histogram)
{
    histogram.SetMarkerStyle(20);
    histogram.SetMarkerSize(0.9);
    histogram.SetLineWidth(2);
}

void printFitSummary(const FitSummary& summary)
{
    std::cout << "--------------------------------------------------\n";
    std::cout << "Theory fit summary\n";
    std::cout << "  P_true       : " << formatSignedValue(summary.truePolarization, 2) << '\n';
    std::cout << "  P_fit        : " << formatSignedValue(summary.fittedPolarization, 2)
              << " +/- " << std::fixed << std::setprecision(2)
              << summary.fittedPolarizationError << '\n';
    std::cout << "  line slope   : " << formatSignedValue(summary.lineSlope, 4)
              << " +/- " << std::fixed << std::setprecision(4)
              << summary.lineSlopeError << '\n';

    if (std::abs(summary.truePolarization) > 1e-12) {
        const double ratio = summary.fittedPolarization / summary.truePolarization;
        std::cout << "  P_fit / P_true: " << std::fixed << std::setprecision(4)
                  << ratio << '\n';
    } else {
        std::cout << "  P_fit / P_true: undefined for P_true = 0\n";
    }

    std::cout << "  MC events    : " << std::fixed << std::setprecision(0)
              << summary.totalEvents << '\n';
    std::cout << "--------------------------------------------------\n";
}

std::unique_ptr<TLegend> makeLegend(TH1D& histogram,
                                    TF1& fitFunction,
                                    const FitSummary& summary)
{
    auto legend = std::make_unique<TLegend>(0.58, 0.15, 0.88, 0.35);
    legend->SetBorderSize(0);
    legend->SetFillStyle(0);
    legend->AddEntry(&histogram,
                     Form("MC A_{FB}(x) (P_{true} = %+.2f)", summary.truePolarization),
                     "lep");
    legend->AddEntry(&fitFunction,
                     Form("Fitted P = %+.2f #pm %.2f",
                          summary.fittedPolarization,
                          summary.fittedPolarizationError),
                     "l");
    return legend;
}

void saveCanvasOutputs(TCanvas& canvas, double polarization)
{
    const std::string outputStem = buildOutputStem(polarization);
    canvas.SaveAs((outputStem + ".pdf").c_str());
    canvas.SaveAs((outputStem + ".root").c_str());
}

// ============================================================
// Input loading
// This is setup code around the real calculation.
// ============================================================
std::unique_ptr<TH2> loadMcGpuHistogram(double polarization)
{
    const std::string path = buildMcGpuPath(polarization);
    if (::gSystem->AccessPathName(path.c_str())) {
        std::cerr << "[WARN] MC GPU file not found: " << path << '\n';
        return nullptr;
    }

    std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "READ"));
    if (!file || file->IsZombie()) {
        std::cerr << "[WARN] Cannot open MC GPU file: " << path << '\n';
        return nullptr;
    }

    TH2* h = dynamic_cast<TH2*>(file->Get(buildMcGpuHistName(polarization).c_str()));
    if (!h) {
        std::cerr << "[WARN] Histogram '"
                  << buildMcGpuHistName(polarization)
                  << "' not found in: " << path << '\n';
        return nullptr;
    }

    TH2* hClone = dynamic_cast<TH2*>(h->Clone());
    hClone->SetDirectory(nullptr);
    return std::unique_ptr<TH2>(hClone);
}

std::unique_ptr<TH1D> loadAfbHistogram(double polarization)
{
    const std::string path = buildAfbPath(polarization);
    if (::gSystem->AccessPathName(path.c_str())) {
        std::cerr << "[WARN] Afb file not found: " << path << '\n';
        return nullptr;
    }

    std::unique_ptr<TFile> file(TFile::Open(path.c_str(), "READ"));
    if (!file || file->IsZombie()) {
        std::cerr << "[WARN] Cannot open Afb file: " << path << '\n';
        return nullptr;
    }

    TH1D* h = dynamic_cast<TH1D*>(file->Get("Afb_vs_x"));
    if (!h) {
        const std::string altName = "Afb_vs_x_P_" + formatPolarizationTag(polarization);
        h = dynamic_cast<TH1D*>(file->Get(altName.c_str()));
    }
    if (!h) {
        std::cerr << "[WARN] Histogram 'Afb_vs_x' not found in: " << path << '\n';
        return nullptr;
    }

    TH1D* hClone = dynamic_cast<TH1D*>(h->Clone());
    hClone->SetDirectory(nullptr);
    return std::unique_ptr<TH1D>(hClone);
}

// ============================================================
// Important physics step
// Convert the 2D MC histogram into A_FB(x) by counting forward events
// (cos(theta) > 0) and backward events (cos(theta) < 0) in each x bin.
// ============================================================
std::unique_ptr<TH1D> makeAfbFrom2D(const TH2* h2, const std::string& name = "hAfb_mc")
{
    if (!h2) {
        std::cerr << "[ERROR] makeAfbFrom2D: null h2\n";
        return nullptr;
    }

    const int xBins = h2->GetNbinsX();
    const int thetaBins = h2->GetNbinsY();
    const double xMin = h2->GetXaxis()->GetXmin();
    const double xMax = h2->GetXaxis()->GetXmax();

    auto hAfb = std::make_unique<TH1D>(name.c_str(),
                                       "A_{FB}(x) from MC; x; A_{FB}",
                                       xBins,
                                       xMin,
                                       xMax);
    hAfb->SetDirectory(nullptr);

    for (int ix = 1; ix <= xBins; ++ix) {
        double N_f = 0.0;
        double N_b = 0.0;

        for (int it = 1; it <= thetaBins; ++it) {
            const double costh = h2->GetYaxis()->GetBinCenter(it);
            const double content = h2->GetBinContent(ix, it);
            if (content <= 0.0) continue;

            if (costh > 0.0) {
                N_f += content;
            } else if (costh < 0.0) {
                N_b += content;
            }
        }

        const double Ntot = N_f + N_b;
        double afb = 0.0;
        double err = 0.0;

        if (Ntot > 0.0) {
            afb = (N_f - N_b) / Ntot;
            err = std::sqrt((1.0 - afb * afb) / Ntot);
        }

        hAfb->SetBinContent(ix, afb);
        hAfb->SetBinError(ix, err);
    }

    return hAfb;
}

}  // namespace

// ============================================================
// Core fit model
// This is the theory expression used to extract P from A_FB(x).
// ============================================================
Double_t Afbfitfunction(Double_t *x, Double_t *par)
{
    const double X = x[0];
    const double P = par[0];

    // Physics formula used in the fit:
    // A_FB(x) = P * (2x - 1) / (2 * (3 - 2x))
    const double denom = 2 * (3.0 - 2.0 * X);

    return P * (2.0 * X - 1.0) / denom;
}

// ============================================================
// Main workflow
// 1. Load the MC inputs.
// 2. Build or load A_FB(x).
// 3. Fit the theory function.
// 4. Save a readable plot and terminal summary.
// ============================================================
void FitTheoryAfb(double Ptrue)
{
    gStyle->SetOptStat(0);

    std::unique_ptr<TH2> h2 = loadMcGpuHistogram(Ptrue);
    if (!h2) {
        std::cerr << "[ERROR] Could not load MC GPU histogram for P = "
                  << Ptrue << '\n';
        return;
    }

    const double totalEvents = h2->Integral();

    std::unique_ptr<TH1D> hAfb = loadAfbHistogram(Ptrue);
    if (!hAfb) {
        std::cerr << "[WARN] Falling back to Afb(x) built from the 2D histogram for P = "
                  << Ptrue << '\n';
        hAfb = makeAfbFrom2D(h2.get(), Form("hAfb_mc_P_%+.1f", Ptrue));
    }
    if (!hAfb) {
        std::cerr << "[ERROR] Could not build Afb histogram for P = " << Ptrue << '\n';
        return;
    }

    const double xMin = hAfb->GetXaxis()->GetXmin();
    const double xMax = hAfb->GetXaxis()->GetXmax();

    auto theoryFit = std::make_unique<TF1>(
        Form("fafb_P_%+.1f", Ptrue), Afbfitfunction, xMin, xMax, 1);
    theoryFit->SetParName(0, "P");
    theoryFit->SetParameter(0, Ptrue);

    auto lineFit = std::make_unique<TF1>(
        Form("fline_P_%+.1f", Ptrue), "[0]*x", xMin, xMax);

    auto canvas = std::make_unique<TCanvas>(
        Form("cAfbFit_P_%+.1f", Ptrue), "A_{FB}(x) fit", 800, 600);

    styleAfbHistogram(*hAfb);
    hAfb->Draw("E1");

    hAfb->Fit(lineFit.get(), "R");
    hAfb->Fit(theoryFit.get(), "R");

    const FitSummary summary{
        Ptrue,
        theoryFit->GetParameter(0),
        theoryFit->GetParError(0),
        lineFit->GetParameter(0),
        lineFit->GetParError(0),
        totalEvents
    };

    std::unique_ptr<TLegend> legend = makeLegend(*hAfb, *theoryFit, summary);

    printFitSummary(summary);
    legend->Draw();
    saveCanvasOutputs(*canvas, Ptrue);
}

// Program entry point: create the output folder and run the fit for each P value.
int main()
{
    gSystem->mkdir(kOutputDir.c_str(), true);

    for (int k = -10; k <= 10; ++k) {
        const double P = k / 10.0;
        FitTheoryAfb(P);
    }

    return 0;
}
