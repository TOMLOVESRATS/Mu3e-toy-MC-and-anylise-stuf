#ifndef COS_THETA_UTILS_H
#define COS_THETA_UTILS_H
#include "TFile.h"
#include "TH2D.h"
#include <memory>
#include <string>
#include <iostream>

inline TH2D* load2DHistogram(const std::string& filename,
                             const std::string& histname)
{
    std::cout << "Opening ROOT file: " << filename << std::endl;

    TFile* f = TFile::Open(filename.c_str(), "READ");
    if (!f || f->IsZombie()) {
        std::cerr << "ERROR: Could not open file: " << filename << std::endl;
        return nullptr;
    }

    std::cout << "Retrieving histogram: " << histname << std::endl;

    TH2* h2_base = dynamic_cast<TH2*>(f->Get(histname.c_str()));
    if (!h2_base) {
        std::cerr << "ERROR: Histogram '" << histname
                  << "' not found in file " << filename << std::endl;
        f->Close();
        delete f;
        return nullptr;
    }

    // We specifically want TH2D. If it's not, you can adapt this,
    // but for now we enforce TH2D.
    TH2D* h2 = dynamic_cast<TH2D*>(h2_base);
    if (!h2) {
        std::cerr << "ERROR: Histogram '" << histname
                  << "' is not a TH2D (it might be TH2F, TH2I, ...)" << std::endl;
        f->Close();
        delete f;
        return nullptr;
    }

    // Clone so we can safely close the file
    TH2D* hClone = dynamic_cast<TH2D*>(h2->Clone());
    hClone->SetDirectory(nullptr);

    f->Close();
    delete f;

    return hClone;
}
//now produca histrogam of cos theta * N vs x
inline TH1D* cosThetaNvsX(TH2D* fakeMC, const std::string& hname)
{
    if (!fakeMC) {
        std::cerr << "ERROR: buildCosThetaNvsX got null fakeMC\n";
        return nullptr;
    }

    int    xBins = fakeMC->GetNbinsX();
    double xMin  = fakeMC->GetXaxis()->GetXmin();
    double xMax  = fakeMC->GetXaxis()->GetXmax();

    TH1D* hCosN_vs_x = new TH1D(
        hname.c_str(),
        "cos#theta * N vs x; x; #Sigma_{#theta} cos#theta * N(x,#theta)",
        xBins, xMin, xMax);

    int thBins = fakeMC->GetNbinsY();

    for (int ix = 1; ix <= xBins; ++ix) {

        double sumCosN = 0.0;

        for (int iy = 1; iy <= thBins; ++iy) {
            double thetaCenter = fakeMC->GetYaxis()->GetBinCenter(iy);
            double cosTheta    = std::cos(thetaCenter);
            double N           = fakeMC->GetBinContent(ix, iy);

            sumCosN += cosTheta * N;
        }

        hCosN_vs_x->SetBinContent(ix, sumCosN);
    }

    return hCosN_vs_x;

}
//compute costheta *N vs x over x vs N 
inline TH1D*AvgcosthetavsX(TH2D* fakeMC, const std::string& baseName)
{
    if (!fakeMC) return nullptr; 
    //find what N vs x is 
    TString hNname = Form("hN_vs_x_%s", baseName.c_str());
    //ProjectionX relly useful it basiclly just sum over the y-axis so i d ont have to write the for loop anymore 
    TH1D* hN_vs_x = fakeMC->ProjectionX(hNname); 
    //using the function define above to find costhetNvsx 
    std::string hCosNname = "hCosN_vs_x_" + baseName;
    TH1D* hCosN_vs_x = cosThetaNvsX(fakeMC, hCosNname);
    //check rather the two function are workgin 
    if (!hN_vs_x || !hCosN_vs_x) {
        std::cerr << "ERROR: Failed to build Nvsx or CosNvsx\n";
        return nullptr;
    }
    //now for the average cos theta 
    std::string hAvgName = "hAvgCosTheta_vs_x_" + baseName;
    TH1D* hAvgCos_vs_x   = (TH1D*)hCosN_vs_x->Clone(hAvgName.c_str());
    hAvgCos_vs_x->SetTitle("<cos#theta>(x); x; <cos#theta>(x)");  
    //loop over all the bins
    //ProjectionX/y wont help me now :(
    for (int ix = 1; ix <= hAvgCos_vs_x->GetNbinsX(); ++ix) {
        double num   = hCosN_vs_x->GetBinContent(ix);
        double denom = hN_vs_x->GetBinContent(ix);

        double avgCos = 0.0;
        if (denom > 0.0)
            avgCos = num / denom;

        hAvgCos_vs_x->SetBinContent(ix, avgCos);
    }

    return hAvgCos_vs_x;
}
inline TH1D* computeAvgCosThetaVsXFromFile(const std::string& filename,
                                           const std::string& histname,
                                           const std::string& tag)
{
    TH2D* h2 = load2DHistogram(filename, histname);
    if (!h2) {
        std::cerr << "computeAvgCosThetaVsXFromFile: failed to load 2D histogram\n";
        return nullptr;
    }

    std::string baseName = tag;
    if (baseName.empty()) {
        baseName = histname;
    }

    TH1D* hAvgCos_vs_x = AvgcosthetavsX(h2, baseName);

    // h2 was cloned from the file; we own it
    delete h2;

    return hAvgCos_vs_x;
}

#endif // COS_THETA_UTILS_H
