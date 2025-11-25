//Getting libraries that will be used 
#include "TCanvas.h"
//root canvas 
#include "TH1.h"
#include "TString.h"
//root 1d histogram
#include "TF1.h"
//root 1d function
#include "TF2.h"
//root 2d function
#include "TH2F.h"
//2D histogram
#include "TH3F.h"
//3D histogram.
#include "TRandom.h"
//ROOT’s random number generator
#include "TGraph2D.h"
// scatter plot
#include "TGraph2DErrors.h"
// with error bars
#include <TLegend.h>
//legend box
#include <TRandom3.h>
//root random generator again 
#include <TFile.h>
// used to save histogram
// These are normal c++ library stuff
#include <cmath>
//inculde math class
#include <random>
//Loads C++’s own random generator
#include <ctime>
//load time for some reason
#include <limits>
// constants for limit value 
#include "TSystem.h"
#include "TROOT.h"


// ----------------------------------------
// Global settings
// ----------------------------------------

double xMinPlot = 10.0 / 53.0;
double thetaMinPlot = M_PI / 2 - 1.3;
double thetaMaxPlot = M_PI / 2 + 1.4;

//root random generation 
TRandom3 randMomentum(0);
//empty stuff for storage
// Asymmetry histograms
TH1D* hAfb_vs_x      = nullptr;   // A_FB vs xp (reduced momentum/energy)
TH1D* hAfb_vs_costh  = nullptr;   // A_FB vs cos(theta_cut)
#include <vector>
// Store one copy of each asymmetry curve per polarisation
std::vector<TH1D*> gAfbX_list;
std::vector<TH1D*> gAfbCos_pos;
std::vector<TH1D*> gAfbCos_neg;
std::vector<double> gP_pos;
std::vector<double> gP_neg;
// ----------------------------------------
// Michel spectrum function
// ----------------------------------------

double michel(double *x, double *par)
{
    double xp    = x[0];     // reduced energy
    double theta = x[1];     // emission angle

    // Michel parameters given in the 2d histogram part 
    double rho     = par[0];
    double eta     = par[1];
    double epsilon = par[2];
    double delta   = par[3];
    double P_u     = par[4];

    // PDG constant
    double x_0 = 9.67e-3;

    // Michel distribution
    double value =
        pow(xp, 2) *
        (3.0 * (1.0 - xp)
        + 2.0 * rho * (4.0 * xp / 3.0 - 1.0)
        + 3.0 * eta * x_0 * (1.0 - xp) / xp
        + P_u * epsilon * cos(theta)
          * (1.0 - xp + (2.0 / 3.0) * delta * (4.0 * xp - 3.0)));

    return value > 0 ? value : 0;
}

//fill A_FB vs xp and A_FB vs cos(theta_cut)

void fillAsymmetryHists(TH2D* fakeMCHistogram,
                        int   xBins,
                        double xMin,
                        double xMax)
{
    int thetaBins = fakeMCHistogram->GetNbinsY();

    // ---------- A_FB vs x_p ----------
    if (!hAfb_vs_x) {
        hAfb_vs_x = new TH1D("hAfb_vs_x",
                             "Forward-Backward Asymmetry vs x_{p};x_{p};A_{FB}",
                             xBins, xMin, xMax);
    }
    hAfb_vs_x->Reset();

    for (int ix = 1; ix <= xBins; ++ix) {
        double N_forward  = 0.0;
        double N_backward = 0.0;

        for (int it = 1; it <= thetaBins; ++it) {
            double thetaLow  = fakeMCHistogram->GetYaxis()->GetBinLowEdge(it);
            double thetaHigh = thetaLow + fakeMCHistogram->GetYaxis()->GetBinWidth(it);

            // average cos(theta) over this theta-bin
            double costh = 0.5 * (std::cos(thetaLow) + std::cos(thetaHigh));

            double content = fakeMCHistogram->GetBinContent(ix, it);
            if (content <= 0.0) continue;

            if (costh > 0.0)
                N_forward  += content;
            else
                N_backward += content;
        }

        double Afb = 0.0;
        if (N_forward + N_backward > 0.0)
            Afb = (N_forward - N_backward) / (N_forward + N_backward);

        hAfb_vs_x->SetBinContent(ix, Afb);
    }


// ---------- A_FB vs |cos(theta)|
if (!hAfb_vs_costh) {
    int nCosBins = 100; // |cosθ| from 0 to 1
    hAfb_vs_costh = new TH1D("hAfb_vs_costh",
                             "Forward-Backward Asymmetry vs |cos#theta|;|cos#theta|;A_{FB}",
                             nCosBins, 0.0, 1.0);
}
hAfb_vs_costh->Reset();

int nCosBins = hAfb_vs_costh->GetNbinsX();

// forward / backward counts per |cosθ| slice
std::vector<double> Nf_slice(nCosBins + 1, 0.0);
std::vector<double> Nb_slice(nCosBins + 1, 0.0);

// loop over theta bins
for (int it = 1; it <= thetaBins; ++it) {

    // use BIN CENTRE for theta
    double thetaCenter = fakeMCHistogram->GetYaxis()->GetBinCenter(it);
    double costh       = std::cos(thetaCenter);
    double absCos      = std::fabs(costh);

    int binAbs = hAfb_vs_costh->GetXaxis()->FindBin(absCos);

    // integrate over ALL xp for this θ-bin
    for (int ix = 1; ix <= xBins; ++ix) {

        double content = fakeMCHistogram->GetBinContent(ix, it);
        if (content <= 0.0) continue;

        if (costh > 0.0)
            Nf_slice[binAbs] += content;  // forward
        else if (costh < 0.0)
            Nb_slice[binAbs] += content;  // backward
        // costh == 0 contributes to neither
    }
}

// convert F/B counts into A_FB per |cosθ| bin
for (int b = 1; b <= nCosBins; ++b) {
    double F = Nf_slice[b];
    double B = Nb_slice[b];

    double Afb = 0.0;
    if (F + B > 0.0)
        Afb = (F - B) / (F + B);

    hAfb_vs_costh->SetBinContent(b, Afb);
}

}


// ----------------------------------------
// Create theoretical 2D histogram and fake Mc Simulation 
// ----------------------------------------
//compute for a given polarisaiton value
void createTheoreticalHistogram(long long numberPositrons,
                                double P_UChange,
                                int xBins, int thetaBins,
                                const char* theoryDir,
                                const char* fakeDir,
                                const char* symmetryDir)
{
    //setting x and theta for bining 
    double xMin = xMinPlot;
    double xMax = 1.0;
    double thetaMin = thetaMinPlot;
    double thetaMax = thetaMaxPlot;
// michelk parament packed into parSM 
    double rhoSM = 0.75, etaSM = 0, epsilonSM = 1, deltaSM = 0.75;
    double parSM[5] = {rhoSM, etaSM, epsilonSM, deltaSM, P_UChange};
//Computes the bin width in x and θ
    double binWidthX     = (xMax - xMin) / xBins;
    double binWidthTheta = (thetaMax - thetaMin) / thetaBins;
    TString hname_theory;
    hname_theory.Form("Theoretical_Histogram_SM_P_%+.1f", P_UChange);
//create the 2d histogram 
    TH2D *theoreticalSMHistogram =
        new TH2D(hname_theory,
                 "Theoretical Michel Decay; Reduced Energy; Theta; Events",
                 xBins, xMin, xMax,
                thetaBins, thetaMin, thetaMax);

//This will accumulate the total (integrated) value of the function across all bins.
    double totalEvents = 0;
//loop over bin
// i for x-bin j for theta
    for (int i = 1; i <= xBins; ++i) {
        for (int j = 1; j <= thetaBins; ++j) {
            // getting x0center and theta center 
            double xp    = theoreticalSMHistogram->GetXaxis()->GetBinCenter(i);
            double theta = theoreticalSMHistogram->GetYaxis()->GetBinCenter(j);
            //pack xp and theta to pass in to michel function 
            double input[2] = {xp, theta};
            double value    = michel(input, parSM);
            //value is from the michale spectrum
            //multipy by the bin area to get approximate number of events in that bin 
            double binContent = value * binWidthX * binWidthTheta;
            //update to total events
            totalEvents += binContent;
            //this histogram hold only the theoretical count 
            theoreticalSMHistogram->SetBinContent(i, j, binContent);
        }
    }

    // Normalise to requested total positrons
    if (totalEvents > 0)
        theoreticalSMHistogram->Scale(numberPositrons / totalEvents);

    theoreticalSMHistogram->SetStats(0);

    //fake monte carlo
    TString hname_fake;
    hname_fake.Form("FakeMC_Histogram_SM_P_%+.1f", P_UChange);
    TH2D *fakeMCHistogram =
        (TH2D*)theoreticalSMHistogram->Clone(hname_fake);
    fakeMCHistogram->SetTitle("Fake MC Michel Decay;Reduced Energy;Theta;Events"); 

    // Apply Gaussian fluctuations bin-by-bin
    for (int i = 1; i <= xBins; ++i) {
        for (int j = 1; j <= thetaBins; ++j) {

            double binContent = theoreticalSMHistogram->GetBinContent(i, j);
            double uncertainty = (binContent > 0.0) ? std::sqrt(binContent) : 0.0;

            double newBinContent = binContent;
            if (uncertainty > 0.0) {
                newBinContent = randMomentum.Gaus(binContent, uncertainty);
                if (newBinContent < 0.0) newBinContent = 0.0; // avoid negatives
            }

            fakeMCHistogram->SetBinContent(i, j, newBinContent);
            fakeMCHistogram->SetBinError(i, j, uncertainty);
        }
    }
    // Save plot
    //drawing the graph with c being the canvas name
    // --- Draw and save theory surface ---
    {
        TCanvas *c = new TCanvas("c_theory", "Theoretical Michel Decay SM", 800, 600);
        c->cd();

        TString title;
        title.Form("Theoretical Michel Decay (P = %.2f, N = 10^{15} e^{+});x_{p};#theta;Events",
                   P_UChange);
        theoreticalSMHistogram->SetTitle(title);
        theoreticalSMHistogram->Draw("SURF1");

        std::string pdfName =
            std::string(theoryDir) + "/Theoretical_Graph_SM_P_" + std::to_string(P_UChange) + ".pdf";
        c->SaveAs(pdfName.c_str());

        std::string rootName =
            std::string(theoryDir) + "/Theoretical_SM_Root_P_" + std::to_string(P_UChange) + ".root";
        TFile *f = new TFile(rootName.c_str(), "RECREATE");
        theoreticalSMHistogram->Write("theory");
        f->Close();
    }

    // --- Draw and save fake MC surface ---
    {
        TCanvas *c_fake = new TCanvas("c_fake", "Fake MC Michel Decay SM", 800, 600);
        c_fake->cd();

        TString title_fake;
        title_fake.Form("Fake MC Michel Decay (P = %.2f, N = 10^{15} e^{+});x_{p};#theta;Events",
                        P_UChange);
        fakeMCHistogram->SetTitle(title_fake);
        fakeMCHistogram->Draw("SURF1");

        std::string pdfNameFake =
            std::string(fakeDir) + "/FakeMC_Graph_SM_P_" + std::to_string(P_UChange) + ".pdf";
        c_fake->SaveAs(pdfNameFake.c_str());

        std::string rootNameFake =
            std::string(fakeDir) + "/FakeMC_SM_Root_P_" + std::to_string(P_UChange) + ".root";
        TFile *fFake = new TFile(rootNameFake.c_str(), "RECREATE");
        fakeMCHistogram->Write("fakeMC");
        fFake->Close();
    }

    // --- Fill asymmetry histograms from this fake MC ---
    fillAsymmetryHists(fakeMCHistogram, xBins, xMin, xMax);

    // --- Plot and save A_FB vs x and A_FB vs cos(theta_cut) ---
    // --- Store a clone of the asymmetry histograms for this P ---
    {if (hAfb_vs_x) {
    TString nameX;
    nameX.Form("hAfb_vs_x_P_%+.1f", P_UChange);
    TH1D* hXclone = (TH1D*)hAfb_vs_x->Clone(nameX);
    hXclone->SetDirectory(nullptr); // detach from any file
    gAfbX_list.push_back(hXclone);
    }

// --- Store A_FB vs cos(theta) for this P, split into pos/neg P ---
if (hAfb_vs_costh) {
    TString nameC;
    nameC.Form("hAfb_vs_costh_P_%+.1f", P_UChange);
    TH1D* hCclone = (TH1D*)hAfb_vs_costh->Clone(nameC);
    hCclone->SetDirectory(nullptr);  // detach from files

    if (P_UChange > 1e-6) {
        gAfbCos_pos.push_back(hCclone);
        gP_pos.push_back(P_UChange);
    } else if (P_UChange < -1e-6) {
        gAfbCos_neg.push_back(hCclone);
        gP_neg.push_back(P_UChange);
    } else {
        // P ~ 0, not plotted; avoid leak
        delete hCclone;
    }
}
}
}
// ----------------------------------------
// Main
// ----------------------------------------
int main()
{
    const char* outDir = "/home/tom/Mu3e/Photos/Theory events photos";

    std::string theoryDir   = std::string(outDir) + "/Theory";
    std::string fakeDir     = std::string(outDir) + "/FakeMC";
    std::string symmetryDir = std::string(outDir) + "/Symmetry";

    ::gSystem->mkdir(theoryDir.c_str(),   true);
    ::gSystem->mkdir(fakeDir.c_str(),     true);
    ::gSystem->mkdir(symmetryDir.c_str(), true);

    long long numberOfPositrons = 1000000000000000LL; // 1e15
    int xBins = 100;
    int yBins = 100;

    // ======== LOOP OVER P VALUES ========
    for (double Ptest = -1.0; Ptest <= 1.0001; Ptest += 0.1)
    {
        createTheoreticalHistogram(numberOfPositrons,
                                   Ptest,
                                   xBins, yBins,
                                   theoryDir.c_str(),
                                   fakeDir.c_str(),
                                   symmetryDir.c_str());
    }

// ------------- Draw A_FB vs cos(theta) for positive P (dash markers only) -------------
{
    TCanvas* cPos = new TCanvas("cAfbCosPos",
                                "A_FB vs cos(theta), P > 0",
                                900, 700);
    cPos->cd();

    TLegend* legPos = new TLegend(0.15, 0.20, 0.45, 0.45);
    legPos->SetBorderSize(0);
    legPos->SetFillStyle(0);

    bool first = true;
    int colorIndex = 2;

    for (size_t i = 0; i < gAfbCos_pos.size(); ++i)
    {
        TH1D* h = gAfbCos_pos[i];
        double P = gP_pos[i];

        h->SetLineStyle(0);      // no lines
        h->SetLineWidth(0);
        h->SetMarkerStyle(2);    // small dash
        h->SetMarkerSize(1.4);
        h->SetMarkerColor(colorIndex);

        if (first)
        {
            h->SetStats(0);
            h->GetYaxis()->SetRangeUser(-1.0, 1.0);
            h->Draw("P");
            first = false;
        }
        else
        {
            h->Draw("P SAME");
        }

        TString lab;
        lab.Form("P = %.1f", P);
        legPos->AddEntry(h, lab, "p");

        colorIndex++;
    }

    if (!first)
    {
        legPos->Draw();
        std::string out = std::string(symmetryDir) + "/Afb_vs_cosTheta_PosP.pdf";
        cPos->SaveAs(out.c_str());
    }

    delete cPos;
}


// ------------- Draw A_FB vs cos(theta) for negative P (dash markers only) -------------
{
    TCanvas* cNeg = new TCanvas("cAfbCosNeg",
                                "A_FB vs cos(theta), P < 0",
                                900, 700);
    cNeg->cd();

    TLegend* legNeg = new TLegend(0.15, 0.20, 0.45, 0.45);
    legNeg->SetBorderSize(0);
    legNeg->SetFillStyle(0);

    bool first = true;
    int colorIndex = 2;

    for (size_t i = 0; i < gAfbCos_neg.size(); ++i)
    {
        TH1D* h = gAfbCos_neg[i];
        double P = gP_neg[i];

        h->SetLineStyle(0);      // no lines
        h->SetLineWidth(0);
        h->SetMarkerStyle(2);    // small dash
        h->SetMarkerSize(1.4);
        h->SetMarkerColor(colorIndex);

        if (first)
        {
            h->SetStats(0);
            h->GetYaxis()->SetRangeUser(-1.0, 1.0);
            h->Draw("P");
            first = false;
        }
        else
        {
            h->Draw("P SAME");
        }

        TString lab;
        lab.Form("P = %.1f", P);
        legNeg->AddEntry(h, lab, "p");

        colorIndex++;
    }

    if (!first)
    {
        legNeg->Draw();
        std::string out = std::string(symmetryDir) + "/Afb_vs_cosTheta_NegP.pdf";
        cNeg->SaveAs(out.c_str());
    }

    delete cNeg;
}
// ======== A_FB vs cos(theta): ALL P (pos + neg) ========
{
    TCanvas* cAll = new TCanvas("cAfbCosAll",
                                "A_FB vs cos(theta), All P",
                                900, 700);
    cAll->cd();

    TLegend* legAll = new TLegend(0.15, 0.20, 0.45, 0.45);
    legAll->SetBorderSize(0);
    legAll->SetFillStyle(0);

    bool first = true;
    int colorIndex = 2;

    // -------- POSITIVE P --------
    for (size_t i = 0; i < gAfbCos_pos.size(); ++i)
    {
        TH1D* h = gAfbCos_pos[i];
        double P = gP_pos[i];

        h->SetLineStyle(0);
        h->SetLineWidth(0);
        h->SetMarkerStyle(2);
        h->SetMarkerSize(1.2);
    static const int warmColors[10] = {
        632, // kRed
        633, // kRed+1
        634, // kRed+2
        635, // kRed+3
        636, // kRed+4
        800, // kOrange
        801, // kOrange+1
        802, // kOrange+2
        803, // kOrange+3
        804  // kOrange+4
    };

    h->SetMarkerColor( warmColors[i % 10] );

        if (first) {
            h->SetStats(0);
            h->GetYaxis()->SetRangeUser(-1.0, 1.0);
            h->Draw("P");
            first = false;
        } else {
            h->Draw("P SAME");
        }

        TString lab; lab.Form("P = %.1f (pos)", P);
        legAll->AddEntry(h, lab, "p");

        colorIndex++;
    }

    // -------- NEGATIVE P --------
    for (size_t i = 0; i < gAfbCos_neg.size(); ++i)
    {
        TH1D* h = gAfbCos_neg[i];
        double P = gP_neg[i];

        h->SetLineStyle(0);
        h->SetLineWidth(0);
        h->SetMarkerStyle(2);
        h->SetMarkerSize(1.2);
   static const int coolColors[10] = {
        600, // kBlue
        601, // kBlue+1
        602, // kBlue+2
        603, // kBlue+3
        604, // kBlue+4
        860, // kAzure
        861, // kAzure+1
        862, // kAzure+2
        432, // kCyan+1
        433  // kCyan+2
    };
    h->SetMarkerColor( coolColors[i % 10] );


        h->Draw("P SAME");

        TString lab; lab.Form("P = %.1f (neg)", P);
        legAll->AddEntry(h, lab, "p");

        colorIndex++;
    }

    legAll->Draw();

    std::string out = std::string(symmetryDir) + "/Afb_vs_cosTheta_AllP.pdf";
    cAll->SaveAs(out.c_str());

    delete cAll;
}


    return 0;
}



