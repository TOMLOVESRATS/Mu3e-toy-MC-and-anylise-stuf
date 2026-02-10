#include "TFile.h"
#include "TH1D.h"
#include "TH2.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TString.h"
#include "TSystem.h"
#include "TAxis.h"

#include "AverageCostheta.h"

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>

// ------------------------------------------------------------
// helpers
// ------------------------------------------------------------
static std::string formatP_6(double P)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << P;   // matches your file naming style
    return oss.str();
}

static std::string formatP_showpos_6(double P)
{
    std::ostringstream oss;
    oss << std::showpos << std::fixed << std::setprecision(6) << P;
    return oss.str();
}

static std::string formatP_1_showpos(double P)
{
    std::ostringstream oss;
    oss << std::showpos << std::fixed << std::setprecision(1) << P; // for plot labels
    return oss.str();
}

static std::string formatP_tag_1(double P)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << P;
    std::string s = oss.str();
    for (char &c : s) {
        if (c == '.') c = 'p';
    }
    if (!s.empty() && s[0] == '-') s[0] = 'm';
    return s;
}

// ------------------------------------------------------------
// load Afb(x) histogram from MC afbvsx file
// ------------------------------------------------------------
TH1D* loadSymmetryAfbVsX(const std::string& afbDir, double Pvalue)
{
    const std::string filename =
        afbDir + "/Afb_vs_x_P_" + formatP_tag_1(Pvalue) + ".root";

    TFile* f = TFile::Open(filename.c_str(), "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "ERROR: cannot open file: " << filename << "\n";
        if (f) { f->Close(); delete f; }
        return nullptr;
    }

    TH1D* h = dynamic_cast<TH1D*>(f->Get("Afb_vs_x"));
    if (!h) {
        const std::string altName = "Afb_vs_x_P_" + formatP_tag_1(Pvalue);
        h = dynamic_cast<TH1D*>(f->Get(altName.c_str()));
    }
    if (!h) {
        std::cerr << "ERROR: histogram 'Afb_vs_x' not found in: " << filename << "\n";
        f->Close();
        delete f;
        return nullptr;
    }

    TH1D* hClone = dynamic_cast<TH1D*>(h->Clone(Form("Afb_vs_x_clone_P_%s", formatP_1_showpos(Pvalue).c_str())));
    hClone->SetDirectory(nullptr);

    f->Close();
    delete f;
    return hClone;
}


// ------------------------------------------------------------
// (1) model fit Afb(x) = 3/(2c) * P * <cosθ(E)>
// ------------------------------------------------------------
static TH1D* gAvgCos = nullptr;

double AfbModel(double* xx, double* par)
{
    if (!gAvgCos) return 0.0;

    const double E = xx[0];
    const int bin = gAvgCos->FindBin(E); 
    //making anything outside the bin to return 0
    if (bin < 1 || bin > gAvgCos->GetNbinsX()) return 0.0;  
    const double avgCos = gAvgCos->GetBinContent(bin);
    return 3.0/(2.0*0.05)*par[0]* avgCos;

}
void FitP_from_Avgcos()
{
    const std::string baseMcGpuDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/Constant cut";
    const std::string baseAfbDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/Constant cut/Constant cut afb";
    const std::string baseOutDir =
        "/home/tom/Mu3e/Photos/MC 3 layer photos/Constant cut/constant cut Meg fit";

    const std::string mcGpuDir = baseMcGpuDir;
    const std::string afbDir = baseAfbDir;
    const std::string outDir = baseOutDir;
    std::vector<double> pTrueVals;
    std::vector<double> pRatioVals;
    std::vector<double> pRatioErrVals;

    gSystem->mkdir(outDir.c_str(), true);

    for (int i = 0; i <= 20; ++i) {
        double Ptrue = 1.0 - 0.1 * i;
        TH1D* hAfb = loadSymmetryAfbVsX(afbDir, Ptrue);
        if (!hAfb) continue;

        const std::string tag = formatP_tag_1(Ptrue);
        const std::string mcFile =
            mcGpuDir + "/MC_P_" + tag + "_ConstCut.root";

        gAvgCos = AverageCostheta::computeAvgCosThetaVsXFromFile(
            mcFile, "HistogramRes_P_" + tag+"_ConstCut", Form("P_%+.1f", Ptrue));
        if (!gAvgCos) {
            std::cerr << "ERROR: avg cos(theta) histogram not available for P=" << Ptrue << "\n";
            delete hAfb;
            continue;
        }

        if (std::abs(Ptrue) <= 1e-12) {
            delete gAvgCos;
            gAvgCos = nullptr;
            delete hAfb;
            continue;
        }
        gAvgCos->Scale(1.0 / Ptrue);

        const double xMin = hAfb->GetXaxis()->GetXmin();
        const double xMax = hAfb->GetXaxis()->GetXmax();
        const double xMinFit = 0.6;
        TF1 fP("fP", AfbModel, std::max(xMin, xMinFit), xMax, 1);
        fP.SetParameter(0, 0.4);
        int fitStatus = hAfb->Fit(&fP, "QR");
        if (fitStatus != 0) {
            std::cerr << "WARN: fit failed for P=" << Ptrue << " status=" << fitStatus << "\n";
        }

        const double Pfit = fP.GetParameter(0);
        const double PfitErr = fP.GetParError(0);
        pTrueVals.push_back(Ptrue);
        pRatioVals.push_back(Pfit / Ptrue);
        pRatioErrVals.push_back(PfitErr / std::abs(Ptrue));

        hAfb->SetMarkerStyle(20);
        hAfb->SetMarkerSize(0.8);
        hAfb->SetLineWidth(1);

        fP.SetLineWidth(2);
        fP.SetLineColor(kRed + 1);

        TCanvas cAfb(Form("cAfb_%s", tag.c_str()), "Afb vs x with fit", 900, 650);
        hAfb->SetTitle(Form("A_{FB}(x) fit, P_{true}=%+.1f; x=E/E_{end}; A_{FB}", Ptrue));
        hAfb->Draw("E1");
        fP.Draw("SAME");
        TLegend leg(0.52, 0.73, 0.88, 0.88);
        leg.SetBorderSize(0);
        leg.SetFillStyle(0);
        leg.AddEntry(hAfb, Form("A_{FB}(x), P_{true}=%+.1f", Ptrue), "lep");
        leg.AddEntry(&fP, Form("fit: P_{fit}=%+.3f #pm %.3f", Pfit, PfitErr), "l");
        leg.Draw();

        const std::string pShow = formatP_1_showpos(Ptrue);
        const std::string outPdf = outDir + "/AfbVsX_FitP_P_" + pShow + ".pdf";
        const std::string outRoot = outDir + "/AfbVsX_FitP_P_" + pShow + ".root";
        cAfb.SaveAs(outPdf.c_str());

        TFile fOut(outRoot.c_str(), "RECREATE");
        hAfb->Write("Afb_vs_x");
        fP.Write("fit_AfbModel");
        fOut.Close();

        delete gAvgCos;
        gAvgCos = nullptr;
        delete hAfb;
    }

    if (!pTrueVals.empty()) {
        TGraphErrors gRatio((int)pTrueVals.size());
        gRatio.SetName("g_pfit_over_ptrue_vs_ptrue");
        gRatio.SetTitle("P_{fit}/P_{true} vs P_{true}; P_{true}; P_{fit}/P_{true}");
        gRatio.SetMarkerStyle(20);
        for (int i = 0; i < (int)pTrueVals.size(); ++i) {
            gRatio.SetPoint(i, pTrueVals[i], pRatioVals[i]);
            gRatio.SetPointError(i, 0.0, pRatioErrVals[i]);
        }

        TCanvas cRatio("c_pfit_over_ptrue_vs_ptrue", "Pfit/Ptrue vs Ptrue", 900, 650);
        gRatio.Draw("APE");
        cRatio.SaveAs(Form("%s/PfitOverPtrueVsPtrue.pdf", outDir.c_str()));
    }
}


// ------------------------------------------------------------
// main
// ------------------------------------------------------------
int main()
{
    FitP_from_Avgcos();

    return 0;
}
