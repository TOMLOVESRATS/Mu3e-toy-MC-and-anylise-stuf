#include "TCanvas.h" // ROOT canvas for plotting
#include "TClass.h" // ROOT class info helpers
#include "TFile.h" // ROOT file I/O
#include "TH1D.h" // ROOT 1D histogram
#include "TH2.h" // ROOT 2D histogram
#include "TKey.h" // ROOT key iterator
#include "TROOT.h" // ROOT globals
#include "TSystem.h" // ROOT system utilities
#include "TString.h" // ROOT string helpers
#include "TLatex.h" // ROOT latex text

#include <cmath> // std::sqrt, std::pow
#include <iomanip> // std::setprecision
#include <iostream> // std::cout, std::cerr
#include <sstream> // std::ostringstream
#include <string> // std::string
#include <vector> // std::vector

// ============================================================
// File guide
//
// Important analysis section:
// - makeAfbFBVsX(...) is the main physics/output step in this file.
// - ProcessOneP(...) is the main per-file workflow.
//
// What this file does:
// - reads already-produced TH2 MC histograms
// - splits them into forward and backward hemispheres in cos(theta)
// - builds A_FB(x), F(x), and B(x)
// - writes ROOT outputs and a PDF plot for each polarization point
//
// Convenience-only section:
// - filename formatting, path building, histogram discovery, and mkdir.
//
// Graph / presentation-only section:
// - TCanvas, TLatex, marker styling, and PDF export.
//
// LLM-assisted / readability-only section:
// - the section labels and the convenience/presentation comments below are
//   only for readability; they do not change the analysis result.
//
// Existing input/output locations are intentionally kept in place.
// ============================================================

// ------------------------------------------------------------
// Convenience section: helper functions and file/path handling
// ------------------------------------------------------------
static std::string FormatPTag(double P) // format P into file tag string
{ // begin FormatPTag
    std::ostringstream tag; // stream for formatting
    tag << std::fixed << std::setprecision(1) << P; // format with 1 decimal
    std::string s = tag.str(); // get formatted string
    for (char &c : s) { // replace dot with 'p'
        if (c == '.') c = 'p'; // encode decimal point
    } // end replace loop
    if (!s.empty() && s[0] == '-') s[0] = 'm'; // encode minus sign
    return s; // return tag string
} // end FormatPTag

static std::string MakeRootPath(const std::string &baseDir, double P) // build MC ROOT path
{ // begin MakeRootPath
    return baseDir + "/MC_P_" + FormatPTag(P) + ".root"; // compose filename
} // end MakeRootPath

static std::vector<std::string> ListHistogramNames(TFile *file) // list TH2 names in file
{ // begin ListHistogramNames
    std::vector<std::string> names; // output name list
    if (!file) return names; // guard null file

    TIter next(file->GetListOfKeys()); // iterate over keys
    while (TKey *key = dynamic_cast<TKey *>(next())) { // loop over keys
        TClass *cls = gROOT->GetClass(key->GetClassName()); // get object class
        if (cls && cls->InheritsFrom(TH2::Class())) { // only TH2
            names.emplace_back(key->GetName()); // store name
        } // end TH2 check
    } // end key loop

    return names; // return list
} // end ListHistogramNames

static TH2 *LoadHistogram2D(TFile *file, const std::string &histName) // load TH2 by name or first found
{ // begin LoadHistogram2D
    if (!file) return nullptr; // guard null file

    TObject *obj = nullptr; // object holder
    // if histogram is not provided load the first TH2 found // comment: optional name
    if (!histName.empty()) { // use explicit name
        obj = file->Get(histName.c_str()); // load by name
    } else { // search first TH2
        TIter next(file->GetListOfKeys()); // iterate keys
        while (TKey *key = dynamic_cast<TKey *>(next())) { // loop keys
            TClass *cls = gROOT->GetClass(key->GetClassName()); // get class
            if (cls && cls->InheritsFrom(TH2::Class())) { // only TH2
                obj = file->Get(key->GetName()); // load first TH2
                break; // stop after first
            } // end TH2 check
        } // end key loop
    } // end name selection

    TH2 *hist = dynamic_cast<TH2 *>(obj); // cast to TH2
    if (!hist) return nullptr; // return if not found

    TH2 *clone = dynamic_cast<TH2 *>(hist->Clone()); // clone histogram
    if (clone) clone->SetDirectory(nullptr); // detach from file
    return clone; // return clone
} // end LoadHistogram2D

// ============================================================
// Important analysis section: build A_FB(x), F(x), and B(x)
// ============================================================
struct AfbComponents { // bundle for outputs
    TH1D *hAfb; // Afb histogram
    TH1D *hF; // forward counts
    TH1D *hB; // backward counts
}; // end AfbComponents

static AfbComponents makeAfbFBVsX(const TH2 *h2, const std::string &baseName) // build Afb, F, B from TH2
{ // begin makeAfbFBVsX
    const int nx = h2->GetNbinsX(); // number of x bins
    const int ny = h2->GetNbinsY(); // number of cos(theta) bins

    TH1D* hAfb = new TH1D( // allocate Afb histogram
        (baseName + "_Afb").c_str(), // histogram name
        "A_{FB}(x); x=E/E_{end}; A_{FB}", // title and axes
        nx, // number of bins
        h2->GetXaxis()->GetXmin(), // x min
        h2->GetXaxis()->GetXmax() // x max
    ); // end hAfb

    TH1D* hF = new TH1D( // allocate forward histogram
        (baseName + "_F").c_str(), // histogram name
        "Forward counts F(x); x=E/E_{end}; F", // title and axes
        nx, // number of bins
        h2->GetXaxis()->GetXmin(), // x min
        h2->GetXaxis()->GetXmax() // x max
    ); // end hF

    TH1D* hB = new TH1D( // allocate backward histogram
        (baseName + "_B").c_str(), // histogram name
        "Backward counts B(x); x=E/E_{end}; B", // title and axes
        nx, // number of bins
        h2->GetXaxis()->GetXmin(), // x min
        h2->GetXaxis()->GetXmax() // x max
    ); // end hB

    for (int ix = 1; ix <= nx; ++ix) { // loop x bins
        double F = 0.0; // forward sum
        double B = 0.0; // backward sum

        for (int iy = 1; iy <= ny; ++iy) { // loop cos(theta) bins
            const double cosc = h2->GetYaxis()->GetBinCenter(iy); // cos(theta) center
            const double w    = h2->GetBinContent(ix, iy); // bin weight

            if (w <= 0.0) continue; // skip empty bins
            if (cosc > 0.0)      F += w; // forward hemisphere
            else if (cosc < 0.0) B += w; // backward hemisphere
            // if cosc == 0, ignore (measure-zero if your binning avoids exact 0) // keep symmetry
        } // end cos(theta) loop

        const double N = F + B; // total per x bin
        double A = 0.0; // Afb value
        double eA = 0.0; // Afb error

        if (N > 0.0) { // only if total is positive
            A = (F - B) / N; // compute Afb

            // sigma(A) = 2*sqrt(FB)/N^(3/2) // binomial error formula
            const double FB = F * B; // product for error
            if (FB > 0.0) { // only if both sides nonzero
                eA = 2.0 * std::sqrt(FB) / std::pow(N, 1.5); // compute error
            } else { // handle zero product
                eA = 0.0; // no error if one side empty
            } // end FB check
        } // end N check


        hAfb->SetBinContent(ix, A); // store Afb
        hAfb->SetBinError(ix, eA); // store Afb error

        hF->SetBinContent(ix, F); // store forward counts
        hF->SetBinError(ix, (F > 0.0) ? std::sqrt(F) : 0.0); // Poisson error

        hB->SetBinContent(ix, B); // store backward counts
        hB->SetBinError(ix, (B > 0.0) ? std::sqrt(B) : 0.0); // Poisson error
    } // end x loop

    hAfb->SetStats(0); // hide stats box
    hF->SetStats(0); // hide stats box
    hB->SetStats(0); // hide stats box
    return {hAfb, hF, hB}; // return all outputs
} // end makeAfbFBVsX

// ============================================================
// Important workflow section: process one polarization point
// ============================================================
static bool ProcessOneP(double P, // polarization value
                        const std::string &inDir, // input directory
                        const std::string &outDir) // output directory
{ // begin ProcessOneP
    const std::string tag = FormatPTag(P); // format tag
    const std::string inputFile = MakeRootPath(inDir, P); // input ROOT file
    const std::string defaultHistName = "HistogramRes_P_" + tag; // default TH2 name
    const std::string outRoot = outDir + "/Afb_vs_x_P_" + tag + ".root"; // output ROOT file
    const std::string outPdf  = outDir + "/Afb_vs_x_P_" + tag + ".pdf"; // output PDF file

    TFile *file = TFile::Open(inputFile.c_str(), "READ"); // open input file
    if (!file || file->IsZombie()) { // check for errors
        std::cerr << "ERROR: cannot open ROOT file: " << inputFile << "\n"; // log error
        if (file) { file->Close(); delete file; } // cleanup
        return false; // signal failure
    } // end file check

    TH2 *hist = LoadHistogram2D(file, defaultHistName); // try default histogram
    if (!hist) { // fallback if not found
        hist = LoadHistogram2D(file, ""); // load first TH2
    } // end fallback

    if (!hist) { // still not found
        std::cerr << "ERROR: could not find a TH2 histogram in: " << inputFile << "\n"; // log error
        auto names = ListHistogramNames(file); // list available TH2s
        if (!names.empty()) { // if any found
            std::cerr << "Available TH2 histograms:\n"; // header
            for (const auto &name : names) { // loop names
                std::cerr << "  - " << name << "\n"; // print name
            } // end name loop
        } // end names check
        file->Close(); // close file
        delete file; // free file object
        return false; // signal failure
    } // end hist check

    AfbComponents comp = makeAfbFBVsX(hist, "Afb_vs_x_P_" + tag); // build Afb/F/B
    TH1D *hAfb = comp.hAfb; // Afb histogram
    TH1D *hF = comp.hF; // forward histogram
    TH1D *hB = comp.hB; // backward histogram

    // Graph / presentation-only section:
    // the styling and PDF save below are for visual inspection only.
    hAfb->SetMarkerStyle(20); // set marker style
    hAfb->SetMarkerSize(0.8); // set marker size
    hAfb->SetLineWidth(1); // set line width

    TCanvas c(Form("c_afb_%s", tag.c_str()), "Afb vs x", 900, 650); // create canvas
    hAfb->Draw("E1"); // draw with errors

    TLatex latex; // latex helper
    latex.SetNDC(true); // use NDC coordinates
    latex.SetTextSize(0.04); // text size
    latex.DrawLatex(0.15, 0.92, Form("MC GPU: P = %.1f", P)); // label

    c.SaveAs(outPdf.c_str()); // save plot

    TFile outFile(outRoot.c_str(), "RECREATE"); // open output ROOT file
    hAfb->Write("Afb_vs_x"); // write Afb
    hF->Write("F_vs_x"); // write forward counts
    hB->Write("B_vs_x"); // write backward counts
    outFile.Close(); // close output file

    delete hAfb; // free Afb
    delete hF; // free F
    delete hB; // free B
    delete hist; // free TH2 clone
    file->Close(); // close input file
    delete file; // free input file
    return true; // success
} // end ProcessOneP

// ============================================================
// Convenience workflow section: batch over all available P values
// ============================================================
int main() // main entry point
{ // begin main
    gROOT->SetBatch(kTRUE); // run ROOT in batch mode

    const std::string inDir = // input base directory
        "/home/tom/Mu3e/Photos/MC Muon photos/MC GPU 3layer"; // MC outputs
    const std::string outDir = // output directory
        "/home/tom/Mu3e/Photos/MC 3 layer photos/3-layer afbvsx"; // Afb outputs

    gSystem->mkdir(outDir.c_str(), true); // ensure output directory exists

    for (int i = 0; i <= 20; ++i) { // loop P values from -1 to +1
        double P = -1.0 + 0.1 * i; // map index to P
        ProcessOneP(P, inDir, outDir); // process one P
    } // end P loop

    std::cout << "Done: Afb vs x saved to " << outDir << "\n"; // final message
    return 0; // success
} // end main
