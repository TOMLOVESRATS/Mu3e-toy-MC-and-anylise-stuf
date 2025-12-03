//importing root and c++ stuff 

#include "TFile.h"
#include "TH1D.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TStyle.h"
#include "TROOT.h"
#include "TSystem.h"
#include <fstream>
#include "TH2D.h"
#include <iostream>
#include <vector>
#include <string>
#include "TLatex.h"
//function for grabbing the file name 
// fake mc
std::string makeFakeName(const std::string &fakeDir, double P_UChange)
{
    return fakeDir + "/FakeMC_SM_Root_P_" + std::to_string(P_UChange) + ".root";
}
//Afb vs x file 
std::string makeAfbXName(const std::string &symDir, double P)
{
    return symDir + "/Afb_vs_x_P_" + std::to_string(P) + ".root";
}
//using similar method of MEG we will need a golbal avearage cos theta  
double computeAvgAbsCosTheta(TH2D *fakeMC){
    //geting info from the fake mc historgram
    int xBins     = fakeMC->GetNbinsX();
    int thetaBins = fakeMC->GetNbinsY();
    //emtpy storage
    double sumAbsCos = 0.0;
    double sumN      = 0.0; 
    //summing over each bin 
    for (int ix = 1; ix <= xBins; ++ix) {
        for (int it = 1; it <= thetaBins; ++it) {

            double thetaCenter = fakeMC->GetYaxis()->GetBinCenter(it);
            double costh       = std::cos(thetaCenter);
            double w           = fakeMC->GetBinContent(ix, it);
            // ignore negative 
            if (w <= 0.0) continue;

            sumAbsCos += std::fabs(costh) * w;
            sumN      += w;
        }
    }
    // ensure no negative then calculate what avegea cos is 
    if (sumN > 0.0)
        return sumAbsCos / sumN;
    else
        return 0.0;
}
//anaylse function 
void AnalysePolarisation() {
    //display fit statistics 
    gStyle->SetOptFit(1111);
    //setting directory 
    const std::string baseDir     = "/home/tom/Mu3e/Photos/Theory events photos";
    const std::string fakeDir     = baseDir + "/FakeMC";
    const std::string symmetryDir = baseDir + "/Symmetry";
    // Output directory for this analysis
    const std::string outDir      = baseDir + "/LinearPolarisation";
    gSystem->mkdir(outDir.c_str(), true);

    std::cout << "FakeMC dir:   " << fakeDir     << std::endl;
    std::cout << "Symmetry dir: " << symmetryDir << std::endl;
    std::cout << "Output dir:   " << outDir      << std::endl;
    //storage for plotting 
    std::vector<double> vP_true;
    std::vector<double> vP_fit;
    std::vector<double> vE_P_fit;

    // Summary file for reports 
    const std::string summaryName = outDir + "/LinearFit_PolarisationSummary.txt";
    std::ofstream summary(summaryName);
    if (!summary.is_open()) {
        std::cerr << "[ERROR] Cannot open summary file: " << summaryName << std::endl;
        return;
    }
    summary << "# P_true   P_fit   err_P_fit   slope   err_slope   offset   err_offset   avgAbsCos\n"; 
    // same looping as before
    for(double P_UChange = -1.0; P_UChange <= 1.000001; P_UChange += 0.1){
        std::cout << "\n============================================\n";
        std::cout << "Analysing P_UChange = " << P_UChange << std::endl;
        // opening mc file to get each <|costheta|> 
        std::string rootNameFake = makeFakeName(fakeDir, P_UChange);
        std::cout << "  Opening FakeMC file: " << rootNameFake << std::endl;
        TFile *fFake = TFile::Open(rootNameFake.c_str(), "READ");
        if (!fFake || fFake->IsZombie()) {
            std::cerr << "  [WARNING] Cannot open " << rootNameFake << std::endl;
            if (fFake) fFake->Close();
            continue;
        }

        TH2D *fakeMC = dynamic_cast<TH2D*>(fFake->Get("fakeMC"));
        if (!fakeMC) {
            std::cerr << "  [WARNING] 'fakeMC' not found in " << rootNameFake << std::endl;
            fFake->Close();
            continue;
        }

        double avgAbsCos = computeAvgAbsCosTheta(fakeMC);
        std::cout << "  <|cosθ|> = " << avgAbsCos << std::endl; 
    //get A(x) from Afbvsx 
        std::string rootNameAfbX = makeAfbXName(symmetryDir, P_UChange);
        std::cout << "  Opening Afb_vs_x file: " << rootNameAfbX << std::endl;

        TFile *fAfbX = TFile::Open(rootNameAfbX.c_str(), "READ");
        if (!fAfbX || fAfbX->IsZombie()) {
            std::cerr << "  [WARNING] Cannot open " << rootNameAfbX << std::endl;
            if (fAfbX) fAfbX->Close();
            fFake->Close();
            continue;
        }

        TH1D *hAfb_vs_x = dynamic_cast<TH1D*>(fAfbX->Get("Afb_vs_x"));
        if (!hAfb_vs_x) {
            std::cerr << "  [WARNING] 'Afb_vs_x' not found in " << rootNameAfbX << std::endl;
            fAfbX->Close();
            fFake->Close();
            continue;
        }
    // linear fit for each A(x)=slope*x+c 
        double xMin = hAfb_vs_x->GetXaxis()->GetXmin();
        double xMax = hAfb_vs_x->GetXaxis()->GetXmax();
        TF1 *fit_A = new TF1("fitA", "[0]*x+[1]", xMin, xMax);
        fit_A->SetParName(0, "slope");
        fit_A->SetParName(1, "offset");
        fit_A->SetParameter(0, 0.0);  // initial slope guess
        fit_A->SetParameter(1, 0.0);  // initial offset guess 
        //fitting 
        hAfb_vs_x->Fit(fit_A, "RQ0"); 
        //fitting the parameters        
        double slope     = fit_A->GetParameter(0);
        double eSlope    = fit_A->GetParError(0);
        double offset    = fit_A->GetParameter(1);
        double eOffset   = fit_A->GetParError(1);
        double P_fit  = 0.0;
        double eP_fit = 0.0;
        //calculate it if it is not zero 
        if (avgAbsCos != 0.0) {
            P_fit  = slope  / avgAbsCos;
            eP_fit = eSlope / avgAbsCos;
            }                   
        summary << P_UChange   << "   "
        << P_fit    << "   " << eP_fit    << "   "
        << slope    << "   " << eSlope    << "   "
        << offset   << "   " << eOffset   << "   "
        << avgAbsCos << "\n"; 
        std::cout << "  slope = " << slope  << " ± " << eSlope
                  << ",  offset = " << offset << " ± " << eOffset
                  << "\n  => P_fit = " << P_fit << " ± " << eP_fit << std::endl;
        //plotting 
        TString cname;
        cname.Form("cAfb_x_P_%+0.1f", P_UChange);
        TCanvas *c_afb_x = new TCanvas(cname, "A_FB vs x with fit", 800, 600);
        c_afb_x->cd();

        TString titleAfbX;
        titleAfbX.Form("Forward-Backward Asymmetry vs x  (P = %.2f);x;A_{FB}", P_UChange);

        hAfb_vs_x->SetTitle(titleAfbX);
        hAfb_vs_x->SetLineWidth(2);
        hAfb_vs_x->SetMarkerStyle(20);
        hAfb_vs_x->SetMarkerSize(1.0);

        // Draw data points without error bars
        hAfb_vs_x->Draw("P");

        // Draw the fitted straight line on top
        fit_A->SetLineWidth(2);
        fit_A->SetLineStyle(2); // dashed so you can see it's the fit
        fit_A->Draw("SAME");

        // Optional legend
        TLegend *legA = new TLegend(0.15, 0.70, 0.45, 0.88);
        legA->AddEntry(hAfb_vs_x, "A_{FB}(x) bins", "p");
        legA->AddEntry(fit_A, "Linear fit", "l");
        legA->Draw();

        // Equation text on the plot
        TLatex latex;
        latex.SetNDC();
        latex.SetTextSize(0.04);

        latex.DrawLatex(
            0.15, 0.62,
            Form("A_{FB}(x) = (%.3f #pm %.3f) x + (%.3f #pm %.3f)",
                 slope, eSlope, offset, eOffset)
        );

        if (avgAbsCos > 0.0) {
            latex.DrawLatex(
                0.15, 0.56,
                Form("P_{#mu} = slope / #LT|cos#theta|#GT = %.3f", P_fit)
            );
        }

        // Save the plot in your output directory
        TString pdfNameAfbX;
        pdfNameAfbX.Form("%s/Afb_vs_x_with_fit_P_%+0.1f.pdf",
                         outDir.c_str(), P_UChange);
        c_afb_x->SaveAs(pdfNameAfbX);
        

        fAfbX->Close();
        fFake->Close();        
    }
    //saving the summarry 
    summary.close();
    std::cout << "\nSummary written to: " << summaryName << std::endl;
    std::cout << "Done AnalysePolarisationLinear()." << std::endl;
} 
int main()
{
    AnalysePolarisation();
    return 0;
}
