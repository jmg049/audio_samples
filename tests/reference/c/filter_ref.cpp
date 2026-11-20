// filter_ref: independent IIR ground-truth harness.
//
// Designs IIR filters with two third-party libraries (DSPFilters, double
// precision; liquid-dsp, single precision), mapping SciPy-convention
// parameters onto each library's own parameterisation, and runs the
// library's own processing routines. See README.md for the protocol.

#include <DspFilters/Dsp.h>
#include <complex>
#include <liquid/liquid.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kImpulseLen = 2048;
constexpr int kMaxOrder = 16;
constexpr double kPi = 3.14159265358979323846264338327950288;

struct Case {
  std::string id, lib, family, response;
  int order = 0;
  double fs = 0, f1 = 0, f2 = 0, rp = 0, rs = 0, gain_db = 0, q = 0;
  std::string input_path;
};

struct Result {
  bool exact = true;
  std::string note;
  std::vector<std::array<double, 6>> sos;
  std::vector<double> impulse;
  std::vector<double> output;
};

void add_note(Result& r, const std::string& s) {
  if (!r.note.empty()) r.note += "; ";
  r.note += s;
}

struct Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------- helpers ---

double warp(double f, double fs) { return std::tan(kPi * f / fs); }
double unwarp(double w, double fs) { return std::atan(w) * fs / kPi; }

std::vector<double> read_input(const std::string& path) {
  std::vector<double> x;
  if (path == "-" || path.empty()) return x;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open input file: " + path);
  in.seekg(0, std::ios::end);
  std::streamoff n = in.tellg();
  in.seekg(0, std::ios::beg);
  if (n % 8 != 0) throw std::runtime_error("input file size is not a multiple of 8 bytes");
  x.resize(static_cast<size_t>(n / 8));
  // assumes a little-endian host (x86_64 / aarch64)
  in.read(reinterpret_cast<char*>(x.data()), n);
  return x;
}

// ---- Bessel normalisation -------------------------------------------------
// All Bessel prototypes here are expressed relative to the delay-normalised
// prototype H_d(s) = theta_n(0)/theta_n(s) (reverse Bessel polynomial).
// A prototype "with scale k" is H_d(s/k), i.e. its poles are k * p_delay.
//   scipy norm='delay': k = 1
//   scipy norm='phase': k = theta_n(0)^(-1/n)          (scipy besselap)
//   scipy norm='mag'  : k = 1 / w3dB(H_d)
//   DSPFilters        : k = 1 (it uses the raw reverse-Bessel roots)
//   liquid-dsp        : k = 1 / sqrtf((2n-1) ln 2)     (approximate 3 dB norm)

double reverse_bessel_coef(int k, int n) {  // coefficient of s^k in theta_n(s)
  // (2n-k)! / (2^(n-k) k! (n-k)!), computed in log space for safety
  double l = std::lgamma(2.0 * n - k + 1) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0) -
             (n - k) * std::log(2.0);
  return std::round(std::exp(l));
}

double bessel_w3db_delay(int n) {
  std::vector<double> c(n + 1);
  for (int k = 0; k <= n; ++k) c[k] = reverse_bessel_coef(k, n);
  auto mag2 = [&](double w) {
    std::complex<double> s(0, w), acc(0, 0);
    for (int k = n; k >= 0; --k) acc = acc * s + c[k];
    return std::norm(acc) / (c[0] * c[0]);  // |theta(jw)/theta(0)|^2, increasing in w
  };
  double lo = 0, hi = 1;
  while (mag2(hi) < 2.0) hi *= 2;
  for (int i = 0; i < 200; ++i) {
    double mid = 0.5 * (lo + hi);
    if (mag2(mid) < 2.0) lo = mid; else hi = mid;
  }
  return 0.5 * (lo + hi);
}

double bessel_target_k(const std::string& family, int n) {
  if (family == "bessel" || family == "bessel_phase") return std::pow(reverse_bessel_coef(0, n), -1.0 / n);
  if (family == "bessel_mag") return 1.0 / bessel_w3db_delay(n);
  if (family == "bessel_delay") return 1.0;
  throw Unsupported("unknown bessel family " + family);
}

bool is_bessel(const std::string& f) {
  return f == "bessel" || f == "bessel_phase" || f == "bessel_mag" || f == "bessel_delay" ||
         f == "bessel_native";
}

// Rescale SciPy-convention edges so that a library whose Bessel prototype has
// scale k_lib produces the design scipy would make with scale k_target.
// rho = k_target / k_lib. All work is on the prewarped (tan) axis.
void bessel_rescale(const std::string& response, double rho, double fs, double& f1, double& f2) {
  if (rho == 1.0) return;
  if (response == "lowpass") {
    f1 = unwarp(warp(f1, fs) * rho, fs);
  } else if (response == "highpass") {
    f1 = unwarp(warp(f1, fs) / rho, fs);
  } else {
    double w1 = warp(f1, fs), w2 = warp(f2, fs);
    double bw = w2 - w1, w0sq = w1 * w2;
    double bwn = (response == "bandpass") ? bw * rho : bw / rho;
    double w1n = 0.5 * (-bwn + std::sqrt(bwn * bwn + 4 * w0sq));
    double w2n = w1n + bwn;
    f1 = unwarp(w1n, fs);
    f2 = unwarp(w2n, fs);
  }
}

// ---- elliptic degree (for DSPFilters' "rolloff" parameter) ----------------
double ellipk_m(double m) {  // K(m), parameter m = k^2, via AGM
  double a = 1, b = std::sqrt(1 - m);
  for (int i = 0; i < 60 && std::fabs(a - b) > 1e-17 * a; ++i) {
    double an = 0.5 * (a + b);
    b = std::sqrt(a * b);
    a = an;
  }
  return kPi / (2 * a);
}
double ellipkp_m(double m) {  // K'(m) = K(1-m), accurate for tiny m
  double a = 1, b = std::sqrt(m);
  for (int i = 0; i < 60 && std::fabs(a - b) > 1e-17 * a; ++i) {
    double an = 0.5 * (a + b);
    b = std::sqrt(a * b);
    a = an;
  }
  return kPi / (2 * a);
}
// Same algorithm as scipy.signal._filter_design._ellipdeg: returns m.
double ellipdeg(int n, double m1) {
  const int MMAX = 7;
  double K1 = ellipk_m(m1), K1p = ellipkp_m(m1);
  double q1 = std::exp(-kPi * K1p / K1);
  double q = std::pow(q1, 1.0 / n);
  double num = 0, den = 0;
  for (int i = 0; i <= MMAX; ++i) num += std::pow(q, i * (i + 1.0));
  for (int i = 1; i <= MMAX + 1; ++i) den += std::pow(q, double(i) * i);
  return 16 * q * std::pow(num / (1 + 2 * den), 4);
}

// ---- exact (double) analog prototypes, used only to assess liquid's float32 ones
void exact_cheby1(int n, double ep, std::vector<std::complex<double>>& p) {
  double mu = std::asinh(1.0 / ep) / n;
  for (int k = 0; k < n; ++k) {
    double th = kPi * (2 * k + 1) / (2.0 * n);
    p.emplace_back(-std::sinh(mu) * std::sin(th), std::cosh(mu) * std::cos(th));
  }
}
void exact_cheby2(int n, double de, std::vector<std::complex<double>>& z,
                  std::vector<std::complex<double>>& p) {
  double mu = std::asinh(1.0 / de) / n;
  for (int k = 0; k < n; ++k) {
    double th = kPi * (2 * k + 1) / (2.0 * n);
    p.push_back(1.0 / std::complex<double>(-std::sinh(mu) * std::sin(th), std::cosh(mu) * std::cos(th)));
    if (2 * k + 1 != n) z.emplace_back(0.0, 1.0 / std::cos(th));
  }
}
// Orfanidis' elliptic design (the algorithm liquid implements) in double.
using cd = std::complex<double>;
std::vector<double> landen(double k, int m) {
  std::vector<double> v(m);
  for (int i = 0; i < m; ++i) {
    double kp = std::sqrt(1 - k * k);
    k = (1 - kp) / (1 + kp);
    v[i] = k;
  }
  return v;
}
cd ell_cd(cd u, double k, int m) {
  cd w = std::cos(u * kPi * 0.5);
  auto v = landen(k, m);
  for (int i = m; i > 0; --i) w = (1 + v[i - 1]) * w / (1.0 + v[i - 1] * w * w);
  return w;
}
cd ell_sn(cd u, double k, int m) {
  cd w = std::sin(u * kPi * 0.5);
  auto v = landen(k, m);
  for (int i = m; i > 0; --i) w = (1 + v[i - 1]) * w / (1.0 + v[i - 1] * w * w);
  return w;
}
cd ell_acd(cd w, double k, int m) {
  auto v = landen(k, m);
  for (int i = 0; i < m; ++i) {
    double v1 = (i == 0) ? k : v[i - 1];
    w = w / (1.0 + std::sqrt(1.0 - w * w * v1 * v1)) * 2.0 / (1 + v[i]);
  }
  return std::acos(w) * 2.0 / kPi;
}
double ellipdeg(int n, double m1);
void exact_ellip(int n, double ep, double es, std::vector<cd>& z, std::vector<cd>& p) {
  const int m = 12;
  double k1 = ep / es;
  double k = std::sqrt(ellipdeg(n, k1 * k1));
  int L = n / 2;
  for (int i = 1; i <= L; ++i) {
    double u = (2.0 * i - 1.0) / n;
    cd zeta = ell_cd(u, k, m);
    z.push_back(cd(0, 1) / (k * zeta));
    z.push_back(std::conj(z.back()));
  }
  cd v0 = cd(0, -1) * (1.0 - ell_acd(cd(0, 1) / ep, k1, m)) / double(n);
  for (int i = 1; i <= L; ++i) {
    double u = (2.0 * i - 1.0) / n;
    p.push_back(cd(0, 1) * ell_cd(u - cd(0, 1) * v0, k, m));
    p.push_back(std::conj(p.back()));
  }
  if (n % 2) p.push_back(cd(0, 1) * ell_sn(cd(0, 1) * v0, k, m));
}
double max_rel_root_err(const std::vector<cd>& a, const std::vector<cd>& ref) {
  double e = 0;
  for (auto& x : a) {
    double best = 1e300;
    for (auto& y : ref) best = std::min(best, std::abs(x - y) / std::abs(y));
    e = std::max(e, best);
  }
  return a.empty() ? 0.0 : e;
}

// ------------------------------------------------------------- DSPFilters ---

using Dsp::TransposedDirectFormI;

// Exposes the library's own (protected) Cascade::applyScale.
template <class Cascaded>
struct DspCascade : Dsp::SimpleFilter<Cascaded, 1, TransposedDirectFormI> {
  void scale(double g) { this->applyScale(g); }
};

template <class Cascaded, class Setup>
void run_dsp_cascade(Setup setup, double ref_freq, const std::vector<double>& in, Result& r) {
  // TransposedDirectFormI is used deliberately: DSPFilters' default
  // DirectFormII / DirectFormI states inject an alternating +/-1e-8
  // "anti-denormal" signal into every sample, which corrupts the reference
  // at the 1e-8 level. TransposedDirectFormI ignores that term.
  DspCascade<Cascaded> f;
  setup(f);
  // Polarity check. BiquadBase::setOnePole() writes the first-order numerator
  // as (-z0 + z^-1) instead of (1 - z0 z^-1); for the zero at z=+1 of an
  // odd-order high-pass this inverts the sign, and Cascade::setLayout only
  // normalises |H|. The SciPy/analog convention is H(ref_freq) > 0 at the
  // normalisation frequency (DC / Nyquist / band centre), so restore it with
  // the library's own applyScale(-1).
  if (std::real(f.response(ref_freq)) < 0) {
    f.scale(-1.0);
    add_note(r, "polarity corrected with Cascade::applyScale(-1) (DSPFilters setOnePole numerator bug)");
  }
  for (int i = 0; i < f.getNumStages(); ++i) {
    const auto& s = f[i];
    double a0 = s.getA0();
    r.sos.push_back({s.getB0() / a0, s.getB1() / a0, s.getB2() / a0, 1.0, s.getA1() / a0,
                     s.getA2() / a0});
  }
  r.impulse.assign(kImpulseLen, 0.0);
  r.impulse[0] = 1.0;
  f.reset();
  double* ch[1] = {r.impulse.data()};
  f.process(kImpulseLen, ch);
  r.output = in;
  if (!r.output.empty()) {
    f.reset();
    ch[0] = r.output.data();
    f.process(static_cast<int>(r.output.size()), ch);
  }
}

template <class Biquad, class Setup>
void run_dsp_biquad(Setup setup, const std::vector<double>& in, Result& r) {
  Dsp::SimpleFilter<Biquad, 1, TransposedDirectFormI> f;
  setup(f);
  double a0 = f.getA0();
  r.sos.push_back({f.getB0() / a0, f.getB1() / a0, f.getB2() / a0, 1.0, f.getA1() / a0,
                   f.getA2() / a0});
  r.impulse.assign(kImpulseLen, 0.0);
  r.impulse[0] = 1.0;
  f.reset();
  double* ch[1] = {r.impulse.data()};
  f.process(kImpulseLen, ch);
  r.output = in;
  if (!r.output.empty()) {
    f.reset();
    ch[0] = r.output.data();
    f.process(static_cast<int>(r.output.size()), ch);
  }
}

// Dispatch a pole-filter family (Butterworth, ChebyshevI, ...) over the four
// responses. `extra...` are the trailing family-specific setup arguments.
#define DSP_POLE_FAMILY(NS, ...)                                                          \
  do {                                                                                    \
    if (c.response == "lowpass")                                                          \
      run_dsp_cascade<NS::LowPass<kMaxOrder>>(                                            \
          [&](auto& f) { f.setup(c.order, c.fs, f1, ##__VA_ARGS__); }, 0.0, in, r);       \
    else if (c.response == "highpass")                                                    \
      run_dsp_cascade<NS::HighPass<kMaxOrder>>(                                           \
          [&](auto& f) { f.setup(c.order, c.fs, f1, ##__VA_ARGS__); }, 0.5, in, r);       \
    else if (c.response == "bandpass")                                                    \
      run_dsp_cascade<NS::BandPass<kMaxOrder>>(                                           \
          [&](auto& f) { f.setup(c.order, c.fs, centre, width, ##__VA_ARGS__); },         \
          std::atan(std::sqrt(warp(f1, c.fs) * warp(f2, c.fs))) / kPi, in, r);            \
    else if (c.response == "bandstop")                                                    \
      run_dsp_cascade<NS::BandStop<kMaxOrder>>(                                           \
          [&](auto& f) { f.setup(c.order, c.fs, centre, width, ##__VA_ARGS__); }, 0.0, in, r); \
    else                                                                                  \
      throw Unsupported("unknown response " + c.response);                               \
  } while (0)

void run_dspfilters(const Case& c, const std::vector<double>& in, Result& r) {
  const std::string& fam = c.family;
  if (fam.rfind("rbj_", 0) == 0) {
    double w0 = 2 * kPi * c.f1 / c.fs;
    if (fam == "rbj_lowpass") {
      run_dsp_biquad<Dsp::RBJ::LowPass>([&](auto& f) { f.setup(c.fs, c.f1, c.q); }, in, r);
    } else if (fam == "rbj_highpass") {
      run_dsp_biquad<Dsp::RBJ::HighPass>([&](auto& f) { f.setup(c.fs, c.f1, c.q); }, in, r);
    } else if (fam == "rbj_bandpass") {
      // BandPass2 = cookbook "constant 0 dB peak gain"; its 'bandWidth' arg is used as Q.
      run_dsp_biquad<Dsp::RBJ::BandPass2>([&](auto& f) { f.setup(c.fs, c.f1, c.q); }, in, r);
    } else if (fam == "rbj_bandstop") {
      run_dsp_biquad<Dsp::RBJ::BandStop>([&](auto& f) { f.setup(c.fs, c.f1, c.q); }, in, r);
    } else if (fam == "rbj_peak") {
      // BandShelf uses alpha = sin(w0) sinh(ln2/2 * BW * w0/sin(w0)); invert for Q.
      double bw = 2.0 * std::asinh(1.0 / (2.0 * c.q)) * std::sin(w0) / (std::log(2.0) * w0);
      run_dsp_biquad<Dsp::RBJ::BandShelf>([&](auto& f) { f.setup(c.fs, c.f1, c.gain_db, bw); },
                                          in, r);
      add_note(r, "BandShelf bandwidth(oct)=" + std::to_string(bw) + " derived from Q");
    } else if (fam == "rbj_lowshelf" || fam == "rbj_highshelf") {
      // shelves use the cookbook slope S; convert Q -> S exactly.
      double A = std::pow(10.0, c.gain_db / 40.0);
      double S = 1.0 / ((1.0 / (c.q * c.q) - 2.0) / (A + 1.0 / A) + 1.0);
      if (fam == "rbj_lowshelf")
        run_dsp_biquad<Dsp::RBJ::LowShelf>([&](auto& f) { f.setup(c.fs, c.f1, c.gain_db, S); },
                                           in, r);
      else
        run_dsp_biquad<Dsp::RBJ::HighShelf>(
            [&](auto& f) { f.setup(c.fs, c.f1, c.gain_db, S); }, in, r);
      add_note(r, "shelf slope S=" + std::to_string(S) + " derived from Q");
    } else {
      throw Unsupported("unknown rbj family " + fam);
    }
    return;
  }

  if (c.order < 1 || c.order > kMaxOrder)
    throw Unsupported("order must be in 1.." + std::to_string(kMaxOrder));

  double f1 = c.f1, f2 = c.f2;
  if (is_bessel(fam)) {
    if (fam == "bessel_native") {
      r.exact = false;
      add_note(r, "DSPFilters native Bessel (raw reverse-Bessel roots == scipy norm='delay')");
    } else {
      bessel_rescale(c.response, bessel_target_k(fam, c.order) / 1.0, c.fs, f1, f2);
      add_note(r, "Bessel edges rescaled on prewarped axis to emulate scipy norm");
    }
  }
  // DSPFilters band transforms take the arithmetic centre/width of the *digital*
  // band edges; this is the same Constantinides transform scipy obtains via
  // prewarp(f1,f2) + lp2bp/lp2bs + bilinear.
  double centre = 0.5 * (f1 + f2), width = f2 - f1;

  if (fam == "butter") {
    DSP_POLE_FAMILY(Dsp::Butterworth);
  } else if (fam == "cheby1") {
    DSP_POLE_FAMILY(Dsp::ChebyshevI, c.rp);
  } else if (fam == "cheby2") {
    DSP_POLE_FAMILY(Dsp::ChebyshevII, c.rs);
  } else if (fam == "ellip") {
    // DSPFilters parameterises the elliptic selectivity by "rolloff":
    //   xi = ws/wp = 5*exp(rolloff-1) + 1.
    // scipy fixes xi from (N, rp, rs) via the degree equation.
    double eps2 = std::pow(10.0, 0.1 * c.rp) - 1.0;
    double m1 = eps2 / (std::pow(10.0, 0.1 * c.rs) - 1.0);
    double m = ellipdeg(c.order, m1);
    double xi = 1.0 / std::sqrt(m);
    double rolloff = 1.0 + std::log((xi - 1.0) / 5.0);
    DSP_POLE_FAMILY(Dsp::Elliptic, c.rp, rolloff);
    add_note(r, "DSPFilters elliptic prototype is root-polished only to ~1e-6 (Bairstow "
                "tolerance in Elliptic::AnalogLowPass::findfact): expect ~1e-7 pole and "
                "~1e-4 dB agreement, not 1e-12");
    add_note(r, "rolloff=" + std::to_string(rolloff) + " (xi=" + std::to_string(xi) +
                    ") derived from rs via elliptic degree equation");
  } else if (is_bessel(fam)) {
    DSP_POLE_FAMILY(Dsp::Bessel);
  } else {
    throw Unsupported("unknown family " + fam + " for dspfilters");
  }
}

// ----------------------------------------------------------------- liquid ---

void run_liquid(const Case& c, const std::vector<double>& in, Result& r) {
  const std::string& fam = c.family;
  liquid_iirdes_filtertype ft;
  if (fam == "butter") ft = LIQUID_IIRDES_BUTTER;
  else if (fam == "cheby1") ft = LIQUID_IIRDES_CHEBY1;
  else if (fam == "cheby2") ft = LIQUID_IIRDES_CHEBY2;
  else if (fam == "ellip") ft = LIQUID_IIRDES_ELLIP;
  else if (is_bessel(fam)) ft = LIQUID_IIRDES_BESSEL;
  else throw Unsupported("family " + fam + " not provided by liquid-dsp");

  liquid_iirdes_bandtype bt;
  if (c.response == "lowpass") bt = LIQUID_IIRDES_LOWPASS;
  else if (c.response == "highpass") bt = LIQUID_IIRDES_HIGHPASS;
  else if (c.response == "bandpass") bt = LIQUID_IIRDES_BANDPASS;
  else if (c.response == "bandstop") bt = LIQUID_IIRDES_BANDSTOP;
  else throw Unsupported("unknown response " + c.response);

  if (c.order < 1 || c.order > kMaxOrder)
    throw Unsupported("order must be in 1.." + std::to_string(kMaxOrder));

  double f1 = c.f1, f2 = c.f2;
  if (is_bessel(fam)) {
    if (fam == "bessel_native") {
      r.exact = false;
      add_note(r, "liquid native Bessel: delay-normalised poles divided by approximate "
                  "w3dB=sqrt((2n-1)ln2); matches no scipy norm exactly");
    } else {
      float w3a = sqrtf((2 * c.order - 1) * logf(2.0f));  // exactly as liquid computes it
      double rho = bessel_target_k(fam, c.order) * static_cast<double>(w3a);
      bessel_rescale(c.response, rho, c.fs, f1, f2);
      add_note(r, "Bessel edges rescaled on prewarped axis to cancel liquid's approximate "
                  "3 dB normalisation and emulate scipy norm");
    }
  }

  // liquid parameters: fc = "low-pass prototype cut-off", f0 = centre.
  //  lowpass/highpass: fc = cutoff/fs (prewarp tan(pi fc) resp. cot(pi fc)).
  //  bandpass/bandstop: liquid uses the K=1 Constantinides transform with
  //    cos(2 pi f0) = cos(pi (f1+f2)/fs) / cos(pi (f2-f1)/fs)
  //    and a prototype prewarp computed from (fc, f0) that equals
  //    tan(pi (f2-f1)/fs) (bandpass) / cot(...) (bandstop) when fc is either
  //    band edge. The harness passes fc = f2/fs.
  double fc, f0 = 0.0;
  if (bt == LIQUID_IIRDES_LOWPASS || bt == LIQUID_IIRDES_HIGHPASS) {
    fc = f1 / c.fs;
  } else {
    fc = f2 / c.fs;
    double ca = std::cos(kPi * (f1 + f2) / c.fs) / std::cos(kPi * (f2 - f1) / c.fs);
    f0 = std::acos(ca) / (2 * kPi);
  }

  // Unused ripple parameters must still be > 0 for liquid's validation.
  float ap = c.rp > 0 ? static_cast<float>(c.rp) : 1.0f;
  float as = c.rs > 0 ? static_cast<float>(c.rs) : 40.0f;
  if (ft == LIQUID_IIRDES_CHEBY2) {
    // liquid's cheby2 uses epsilon = 10^(-As/20) instead of the textbook
    // 1/sqrt(10^(As/10)-1); pass As' so that 10^(-As'/20) equals the latter.
    as = static_cast<float>(10.0 * std::log10(std::pow(10.0, c.rs / 10.0) - 1.0));
    add_note(r, "cheby2 As'=" + std::to_string(as) + " dB passed to liquid (epsilon convention)");
  }

  // Library-precision diagnostic (the parameter mapping itself is exact).
  // liquid designs its analog prototype in float32 and some of its formulas
  // lose far more than float precision:
  //  * cheby1/cheby2_azpkf: tm = (t0 - 1/e)^(1/n) cancels catastrophically
  //    for small e (large As for cheby2); mathematically tm = 1/tp.
  //  * ellip_azpkf: float Landen recursions (ellipkf/ellipdegf/ellip_cdf)
  //    lose precision for small k1 = ep/es and for selectivity k close to 1.
  // The harness calls liquid's own *_azpkf with exactly the float inputs that
  // liquid_iirdes() uses and compare against the same prototype computed in
  // double precision; if the relative pole/zero error exceeds kTol the case
  // is flagged exact_convention=false.
  if (ft == LIQUID_IIRDES_CHEBY1 || ft == LIQUID_IIRDES_CHEBY2 || ft == LIQUID_IIRDES_ELLIP) {
    const double kTol = 2e-6;
    unsigned int n = static_cast<unsigned int>(c.order);
    std::vector<liquid_float_complex> za(n + 1), pa(n + 1);
    liquid_float_complex ka;
    std::vector<std::complex<double>> zx, px;
    int rc = LIQUID_OK;
    if (ft == LIQUID_IIRDES_CHEBY1) {
      float epsilon = sqrtf(powf(10.0f, ap / 10.0f) - 1.0f);  // as in liquid_iirdes
      rc = cheby1_azpkf(n, epsilon, za.data(), pa.data(), &ka);
      exact_cheby1(c.order, std::sqrt(std::pow(10.0, c.rp / 10.0) - 1.0), px);
    } else if (ft == LIQUID_IIRDES_CHEBY2) {
      float epsilon = powf(10.0f, -as / 20.0f);  // as in liquid_iirdes
      rc = cheby2_azpkf(n, epsilon, za.data(), pa.data(), &ka);
      exact_cheby2(c.order, 1.0 / std::sqrt(std::pow(10.0, c.rs / 10.0) - 1.0), zx, px);
    } else {
      float Gp = powf(10.0f, -ap / 20.0f), Gs = powf(10.0f, -as / 20.0f);  // as in liquid_iirdes
      float ep = sqrtf(1.0f / (Gp * Gp) - 1.0f), es = sqrtf(1.0f / (Gs * Gs) - 1.0f);
      rc = ellip_azpkf(n, ep, es, za.data(), pa.data(), &ka);
      exact_ellip(c.order, std::sqrt(std::pow(10.0, c.rp / 10.0) - 1.0),
                  std::sqrt(std::pow(10.0, c.rs / 10.0) - 1.0), zx, px);
    }
    if (rc == LIQUID_OK) {
      std::vector<std::complex<double>> zl, pl;
      for (unsigned int i = 0; i < zx.size(); ++i) zl.emplace_back(za[i].real(), za[i].imag());
      for (unsigned int i = 0; i < n; ++i) pl.emplace_back(pa[i].real(), pa[i].imag());
      double rel = std::max(max_rel_root_err(pl, px), max_rel_root_err(zl, zx));
      if (rel > kTol) {
        r.exact = false;
        add_note(r, "liquid float32 analog prototype inaccurate: max rel pole/zero error " +
                        std::to_string(rel) + " vs double-precision design");
      }
    }
  }

  unsigned int n = static_cast<unsigned int>(c.order);
  unsigned int r_odd = n % 2, L = (n - r_odd) / 2;
  unsigned int nsos = (bt == LIQUID_IIRDES_BANDPASS || bt == LIQUID_IIRDES_BANDSTOP) ? n : L + r_odd;
  std::vector<float> B(3 * nsos), A(3 * nsos);
  int rc = liquid_iirdes(ft, bt, LIQUID_IIRDES_SOS, n, static_cast<float>(fc),
                         static_cast<float>(f0), ap, as, B.data(), A.data());
  if (rc != LIQUID_OK) throw Unsupported("liquid_iirdes failed (code " + std::to_string(rc) + ")");

  for (unsigned int i = 0; i < nsos; ++i) {
    double a0 = A[3 * i];
    r.sos.push_back({B[3 * i] / a0, B[3 * i + 1] / a0, B[3 * i + 2] / a0, 1.0, A[3 * i + 1] / a0,
                     A[3 * i + 2] / a0});
  }
  iirfilt_rrrf q = iirfilt_rrrf_create_sos(B.data(), A.data(), nsos);
  if (!q) throw Unsupported("iirfilt_rrrf_create_sos failed");
  r.impulse.resize(kImpulseLen);
  for (int i = 0; i < kImpulseLen; ++i) {
    float y;
    iirfilt_rrrf_execute(q, i == 0 ? 1.0f : 0.0f, &y);
    r.impulse[i] = y;
  }
  iirfilt_rrrf_reset(q);
  r.output.resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    float y;
    iirfilt_rrrf_execute(q, static_cast<float>(in[i]), &y);
    r.output[i] = y;
  }
  iirfilt_rrrf_destroy(q);
  add_note(r, "single precision (float32) design and processing");

  // float32 stability of the designed sections (roots of 1 + a1 z^-1 + a2 z^-2)
  double rmax = 0;
  for (auto& s : r.sos) {
    std::complex<double> d = std::sqrt(std::complex<double>(s[4] * s[4] - 4 * s[5]));
    rmax = std::max({rmax, std::abs((-s[4] + d) / 2.0), std::abs((-s[4] - d) / 2.0)});
  }
  // float32 processing check: run the same (float-designed) sections in
  // double and compare with liquid's own float32 iirfilt output. Narrow /
  // near-DC designs can be swamped by DF-II float round-off.
  {
    std::vector<double> hd(kImpulseLen, 0.0);
    hd[0] = 1.0;
    for (auto& sec : r.sos) {
      double w1 = 0, w2 = 0;
      for (auto& x : hd) {
        double w = x - sec[4] * w1 - sec[5] * w2;
        x = sec[0] * w + sec[1] * w1 + sec[2] * w2;
        w2 = w1;
        w1 = w;
      }
    }
    double num = 0, den = 0;
    for (int i = 0; i < kImpulseLen; ++i) {
      num = std::max(num, std::fabs(r.impulse[i] - hd[i]));
      den = std::max(den, std::fabs(hd[i]));
    }
    if (num > 1e-3 * den) {
      r.exact = false;
      add_note(r, "liquid float32 processing round-off: impulse differs from double-precision "
                  "run of the same sections by " + std::to_string(num / den) + " (relative)");
    }
  }
  if (rmax >= 1.0) {
    r.exact = false;
    add_note(r, "UNSTABLE: float32 sections have a pole with |p|=" + std::to_string(rmax) + " >= 1");
  }
}

// ------------------------------------------------------------------- json ---

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", ch);
          o += buf;
        } else {
          o += ch;
        }
    }
  }
  return o + "\"";
}

void jnum(std::string& o, double v) {
  if (!std::isfinite(v)) { o += "null"; return; }
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  o += buf;
}

void jarr(std::string& o, const std::vector<double>& v) {
  o += "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) o += ", ";
    jnum(o, v[i]);
  }
  o += "]";
}

std::string to_json(const std::string& id, const Result& r) {
  std::string o = "{\"id\": " + jstr(id) + ", \"ok\": true, \"exact_convention\": ";
  o += r.exact ? "true" : "false";
  o += ", \"note\": " + jstr(r.note) + ", \"sos\": [";
  for (size_t i = 0; i < r.sos.size(); ++i) {
    if (i) o += ", ";
    o += "[";
    for (int j = 0; j < 6; ++j) {
      if (j) o += ", ";
      jnum(o, r.sos[i][j]);
    }
    o += "]";
  }
  o += "], \"impulse\": ";
  jarr(o, r.impulse);
  o += ", \"output\": ";
  jarr(o, r.output);
  o += "}";
  return o;
}

void validate(const Case& c) {
  const double nyq = 0.5 * c.fs;
  if (!(c.fs > 0)) throw Unsupported("fs must be > 0");
  bool rbj = c.family.rfind("rbj_", 0) == 0;
  bool band = !rbj && (c.response == "bandpass" || c.response == "bandstop");
  if (!(c.f1 > 0 && c.f1 < nyq)) throw Unsupported("f1 must satisfy 0 < f1 < fs/2");
  if (band && !(c.f2 > c.f1 && c.f2 < nyq))
    throw Unsupported("band edges must satisfy 0 < f1 < f2 < fs/2");
  if (rbj && !(c.q > 0)) throw Unsupported("q must be > 0 for rbj_* families");
  if ((c.family == "cheby1" || c.family == "ellip") && !(c.rp > 0))
    throw Unsupported("rp must be > 0");
  if ((c.family == "cheby2" || c.family == "ellip") && !(c.rs > 0))
    throw Unsupported("rs must be > 0");
  if (c.family == "ellip" && !(c.rs > c.rp)) throw Unsupported("rs must exceed rp");
}

}  // namespace

int main() {
  std::ios::sync_with_stdio(false);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream ss(line);
    Case c;
    if (!(ss >> c.id)) continue;  // blank line
    std::string id = c.id;
    try {
      if (!(ss >> c.lib >> c.family >> c.response >> c.order >> c.fs >> c.f1 >> c.f2 >> c.rp >>
            c.rs >> c.gain_db >> c.q >> c.input_path))
        throw Unsupported("malformed line (expected 13 fields)");
      validate(c);
      std::vector<double> in = read_input(c.input_path);
      Result r;
      if (c.lib == "dspfilters") run_dspfilters(c, in, r);
      else if (c.lib == "liquid") run_liquid(c, in, r);
      else throw Unsupported("unknown lib " + c.lib);
      std::cout << to_json(id, r) << "\n";
    } catch (const std::exception& e) {
      std::cout << "{\"id\": " << jstr(id) << ", \"ok\": false, \"error\": " << jstr(e.what())
                << "}\n";
    }
    std::cout.flush();
  }
  return 0;
}
