// feature_registration_no_cv.cpp
// Self-contained feature-based 2D registration (C++17, no OpenCV).
//
// Pipeline:
//   1) Keypoints: Harris / Shi-Tomasi / FAST9
//   2) Orientation: quick gradient-based estimate
//   3) Descriptors: SIFT-like 128D, normalized patch, BRIEF (binary)
//   4) Matching: ratio test + optional mutual check
//   5) Robust model: RANSAC for similarity / affine / homography
//
// Build: see accompanying CMakeLists.txt
//
// Notes:
// - Images are float in [0,1].
// - PGM (P5/P2) I/O included for dependency-free usage.
// - Focus is clarity and completeness, not peak performance.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace reg2d {

// -----------------------------
// Small math helpers
// -----------------------------

static inline double sqr(double x) { return x * x; }

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

struct Mat33 {
  // Row-major 3x3.
  double m[9] = {
      1,0,0,
      0,1,0,
      0,0,1
  };

  static Mat33 Identity() { return Mat33(); }

  double& operator()(int r, int c) { return m[r * 3 + c]; }
  double  operator()(int r, int c) const { return m[r * 3 + c]; }
};

static Mat33 mul(const Mat33& A, const Mat33& B) {
  Mat33 C;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += A(r, k) * B(k, c);
      C(r, c) = s;
    }
  }
  return C;
}

static Vec2 proj(const Mat33& H, const Vec2& p) {
  const double x = p.x, y = p.y;
  const double X = H(0,0)*x + H(0,1)*y + H(0,2);
  const double Y = H(1,0)*x + H(1,1)*y + H(1,2);
  const double Z = H(2,0)*x + H(2,1)*y + H(2,2);
  const double iz = 1.0 / (Z + 1e-12);
  return {X * iz, Y * iz};
}

static bool invert(const Mat33& A, Mat33& invA) {
  // 3x3 inverse via adjugate / determinant.
  const double a00 = A(0,0), a01 = A(0,1), a02 = A(0,2);
  const double a10 = A(1,0), a11 = A(1,1), a12 = A(1,2);
  const double a20 = A(2,0), a21 = A(2,1), a22 = A(2,2);

  const double c00 =  (a11*a22 - a12*a21);
  const double c01 = -(a10*a22 - a12*a20);
  const double c02 =  (a10*a21 - a11*a20);

  const double c10 = -(a01*a22 - a02*a21);
  const double c11 =  (a00*a22 - a02*a20);
  const double c12 = -(a00*a21 - a01*a20);

  const double c20 =  (a01*a12 - a02*a11);
  const double c21 = -(a00*a12 - a02*a10);
  const double c22 =  (a00*a11 - a01*a10);

  const double det = a00*c00 + a01*c01 + a02*c02;
  if (std::abs(det) < 1e-15) return false;

  const double id = 1.0 / det;
  invA(0,0) = c00*id; invA(0,1) = c10*id; invA(0,2) = c20*id;
  invA(1,0) = c01*id; invA(1,1) = c11*id; invA(1,2) = c21*id;
  invA(2,0) = c02*id; invA(2,1) = c12*id; invA(2,2) = c22*id;
  return true;
}

// Solve linear system Ax=b (n small) via Gaussian elimination with partial pivot.
static bool solve_linear(std::vector<double>& A, std::vector<double>& b, int n, std::vector<double>& x) {
  x.assign(n, 0.0);
  for (int k = 0; k < n; ++k) {
    int piv = k;
    double best = std::abs(A[k*n + k]);
    for (int i = k + 1; i < n; ++i) {
      const double v = std::abs(A[i*n + k]);
      if (v > best) { best = v; piv = i; }
    }
    if (best < 1e-14) return false;

    if (piv != k) {
      for (int j = k; j < n; ++j) std::swap(A[k*n + j], A[piv*n + j]);
      std::swap(b[k], b[piv]);
    }

    const double akk = A[k*n + k];
    for (int j = k; j < n; ++j) A[k*n + j] /= akk;
    b[k] /= akk;

    for (int i = 0; i < n; ++i) {
      if (i == k) continue;
      const double f = A[i*n + k];
      if (std::abs(f) < 1e-18) continue;
      for (int j = k; j < n; ++j) A[i*n + j] -= f * A[k*n + j];
      b[i] -= f * b[k];
    }
  }
  for (int i = 0; i < n; ++i) x[i] = b[i];
  return true;
}

// Least squares via normal equations: minimize ||A p - b|| (A: MxN).
static bool least_squares_normal(const std::vector<double>& A, const std::vector<double>& b, int M, int N, std::vector<double>& p) {
  std::vector<double> ATA(N*N, 0.0);
  std::vector<double> ATb(N, 0.0);

  for (int i = 0; i < M; ++i) {
    const double* row = &A[i*N];
    for (int c = 0; c < N; ++c) {
      ATb[c] += row[c] * b[i];
      for (int d = 0; d < N; ++d) ATA[c*N + d] += row[c] * row[d];
    }
  }
  return solve_linear(ATA, ATb, N, p);
}

// -----------------------------
// Image container + PGM I/O
// -----------------------------

struct ImageF {
  int h = 0;
  int w = 0;
  std::vector<float> a;  // row-major

  ImageF() = default;
  ImageF(int H, int W) : h(H), w(W), a(static_cast<size_t>(H*W), 0.0f) {}

  float& operator()(int y, int x) { return a[static_cast<size_t>(y*w + x)]; }
  float  operator()(int y, int x) const { return a[static_cast<size_t>(y*w + x)]; }
};

static void skip_comments(std::istream& is) {
  while (true) {
    int c = is.peek();
    if (c == '#') {
      std::string line;
      std::getline(is, line);
    } else {
      break;
    }
  }
}

static bool read_pgm(const std::string& path, ImageF& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;

  std::string magic;
  f >> magic;
  if (magic != "P5" && magic != "P2") return false;

  skip_comments(f);
  int w = 0, h = 0, maxv = 0;
  f >> w; skip_comments(f);
  f >> h; skip_comments(f);
  f >> maxv;
  f.get(); // consume whitespace

  if (w <= 0 || h <= 0 || maxv <= 0) return false;

  out = ImageF(h, w);
  if (magic == "P5") {
    std::vector<uint8_t> buf(static_cast<size_t>(w*h));
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (!f) return false;
    const float inv = 1.0f / static_cast<float>(maxv);
    for (int i = 0; i < w*h; ++i) out.a[static_cast<size_t>(i)] = buf[static_cast<size_t>(i)] * inv;
  } else {
    const float inv = 1.0f / static_cast<float>(maxv);
    for (int i = 0; i < w*h; ++i) {
      int v = 0;
      f >> v;
      out.a[static_cast<size_t>(i)] = static_cast<float>(v) * inv;
    }
  }
  return true;
}

static bool write_pgm(const std::string& path, const ImageF& img) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << "P5\n" << img.w << " " << img.h << "\n255\n";
  std::vector<uint8_t> buf(static_cast<size_t>(img.w*img.h));
  for (int i = 0; i < img.w*img.h; ++i) {
    float v = img.a[static_cast<size_t>(i)];
    v = std::min(1.0f, std::max(0.0f, v));
    buf[static_cast<size_t>(i)] = static_cast<uint8_t>(std::lround(v * 255.0f));
  }
  f.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  return static_cast<bool>(f);
}

// -----------------------------
// Convolution, blur, gradients
// -----------------------------

static std::vector<float> gaussian_kernel1d(float sigma) {
  if (sigma <= 0.0f) return {1.0f};
  const int r = static_cast<int>(std::ceil(3.0f * sigma));
  const int n = 2*r + 1;
  std::vector<float> k(static_cast<size_t>(n), 0.0f);
  float s = 0.0f;
  for (int i = -r; i <= r; ++i) {
    const float v = std::exp(-(i*i) / (2.0f*sigma*sigma));
    k[static_cast<size_t>(i + r)] = v;
    s += v;
  }
  const float inv = 1.0f / (s + 1e-12f);
  for (auto& v : k) v *= inv;
  return k;
}

static inline int reflect_index(int i, int n) {
  if (n <= 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i - 1;
    if (i >= n) i = 2*n - i - 1;
  }
  return i;
}

static ImageF convolve1d_reflect(const ImageF& img, const std::vector<float>& k, int axis) {
  const int r = static_cast<int>((k.size() - 1) / 2);
  ImageF out(img.h, img.w);

  if (axis == 0) {
    for (int y = 0; y < img.h; ++y) {
      for (int x = 0; x < img.w; ++x) {
        float s = 0.0f;
        for (int t = -r; t <= r; ++t) {
          const int yy = reflect_index(y + t, img.h);
          s += img(yy, x) * k[static_cast<size_t>(t + r)];
        }
        out(y, x) = s;
      }
    }
  } else {
    for (int y = 0; y < img.h; ++y) {
      for (int x = 0; x < img.w; ++x) {
        float s = 0.0f;
        for (int t = -r; t <= r; ++t) {
          const int xx = reflect_index(x + t, img.w);
          s += img(y, xx) * k[static_cast<size_t>(t + r)];
        }
        out(y, x) = s;
      }
    }
  }
  return out;
}

static ImageF gaussian_blur(const ImageF& img, float sigma) {
  const auto k = gaussian_kernel1d(sigma);
  return convolve1d_reflect(convolve1d_reflect(img, k, 1), k, 0);
}

static void sobel_gradients(const ImageF& img, ImageF& gx, ImageF& gy) {
  const std::array<float,3> kx = {1.0f, 0.0f, -1.0f};
  const std::array<float,3> ky = {1.0f, 2.0f, 1.0f};

  auto conv_axis = [&](const ImageF& in, const std::array<float,3>& k, int axis)->ImageF{
    ImageF out(in.h, in.w);
    if (axis == 0) {
      for (int y = 0; y < in.h; ++y) {
        for (int x = 0; x < in.w; ++x) {
          float s = 0.0f;
          for (int t = -1; t <= 1; ++t) {
            const int yy = reflect_index(y + t, in.h);
            s += in(yy, x) * k[static_cast<size_t>(t + 1)];
          }
          out(y, x) = s;
        }
      }
    } else {
      for (int y = 0; y < in.h; ++y) {
        for (int x = 0; x < in.w; ++x) {
          float s = 0.0f;
          for (int t = -1; t <= 1; ++t) {
            const int xx = reflect_index(x + t, in.w);
            s += in(y, xx) * k[static_cast<size_t>(t + 1)];
          }
          out(y, x) = s;
        }
      }
    }
    return out;
  };

  const ImageF tmpx = conv_axis(img, ky, 0);
  gx = conv_axis(tmpx, kx, 1);

  const ImageF tmpy = conv_axis(img, ky, 1);
  gy = conv_axis(tmpy, kx, 0);
}

// -----------------------------
// Keypoints + configs
// -----------------------------

struct Keypoint {
  float x = 0.0f;
  float y = 0.0f;
  float sigma = 1.6f;
  float response = 0.0f;
  float angle = 0.0f; // radians
};

enum class DetectorKind { Harris, ShiTomasi, FAST9 };
enum class DescriptorKind { SIFT128, Patch, BRIEF };
enum class ModelKind { Similarity, Affine, Homography };
enum class MetricKind { L2, Hamming };

struct DetectorConfig {
  DetectorKind kind = DetectorKind::Harris;
  std::vector<float> sigmas = {1.4f, 2.2f};
  int max_total = 260;
  int max_per_scale = 140;
  int nms_radius = 3;
  float rel_threshold = 0.015f;
  float fast_threshold = 0.10f;
  int fast_border = 4;
};

struct SiftConfig {
  int num_cells = 4;
  int num_bins = 8;
  int cell_size = 4;       // 4x4 cells -> 16x16 window
  float step_scale = 1.0f; // step ~ sigma
};

struct PatchConfig {
  int patch_radius = 11;
  float blur_sigma = 1.0f;
};

struct BriefConfig {
  int nbits = 256;
  int patch_radius = 15;
  int seed = 13;
  bool rotate = true;
  float blur_sigma = 1.0f;
};

struct DescriptorConfig {
  DescriptorKind kind = DescriptorKind::SIFT128;
  SiftConfig sift;
  PatchConfig patch;
  BriefConfig brief;
};

struct MatchConfig {
  float ratio = 0.75f;
  bool mutual = true;
  int top_k = 2500;
};

struct RansacConfig {
  int iters = 700;
  float threshold_px = 3.0f;
  int seed = 0;
  bool symmetric = true;
  bool prosac = true;
  int prosac_step = 25;
  int prosac_min_pool = 40;
};

struct RegistrationConfig {
  DetectorConfig detector;
  DescriptorConfig descriptor;
  MatchConfig matcher;
  ModelKind model = ModelKind::Homography;
  RansacConfig ransac;
  bool quick_orientation = true;
};

// -----------------------------
// Corner detectors
// -----------------------------

static void structure_tensor(const ImageF& img, float sigma, ImageF& ixx, ImageF& iyy, ImageF& ixy) {
  const ImageF b = gaussian_blur(img, sigma);
  ImageF gx, gy;
  sobel_gradients(b, gx, gy);

  ixx = ImageF(img.h, img.w);
  iyy = ImageF(img.h, img.w);
  ixy = ImageF(img.h, img.w);

  for (int y = 0; y < img.h; ++y) {
    for (int x = 0; x < img.w; ++x) {
      const float vx = gx(y,x);
      const float vy = gy(y,x);
      ixx(y,x) = vx*vx;
      iyy(y,x) = vy*vy;
      ixy(y,x) = vx*vy;
    }
  }

  const float s = std::max(1.0f, 1.5f*sigma);
  ixx = gaussian_blur(ixx, s);
  iyy = gaussian_blur(iyy, s);
  ixy = gaussian_blur(ixy, s);
}

static ImageF harris_response(const ImageF& img, float sigma, float k = 0.04f) {
  ImageF ixx, iyy, ixy;
  structure_tensor(img, sigma, ixx, iyy, ixy);

  ImageF R(img.h, img.w);
  for (int y = 0; y < img.h; ++y) {
    for (int x = 0; x < img.w; ++x) {
      const float det = ixx(y,x)*iyy(y,x) - ixy(y,x)*ixy(y,x);
      const float tr  = ixx(y,x) + iyy(y,x);
      R(y,x) = det - k * tr * tr;
    }
  }
  return R;
}

static ImageF shitomasi_response(const ImageF& img, float sigma) {
  ImageF ixx, iyy, ixy;
  structure_tensor(img, sigma, ixx, iyy, ixy);

  ImageF R(img.h, img.w);
  for (int y = 0; y < img.h; ++y) {
    for (int x = 0; x < img.w; ++x) {
      const float tr = ixx(y,x) + iyy(y,x);
      const float det = ixx(y,x)*iyy(y,x) - ixy(y,x)*ixy(y,x);
      const float disc = std::max(0.0f, tr*tr - 4.0f*det);
      R(y,x) = 0.5f * (tr - std::sqrt(disc)); // min eigenvalue
    }
  }
  return R;
}

static std::vector<std::tuple<int,int,float>> nms_peaks(const ImageF& R, int radius, float thr) {
  std::vector<std::tuple<int,int,float>> peaks;
  const int r = radius;
  for (int y = r; y < R.h - r; ++y) {
    for (int x = r; x < R.w - r; ++x) {
      const float v = R(y,x);
      if (v < thr) continue;
      float mx = v;
      for (int yy = y - r; yy <= y + r; ++yy) {
        for (int xx = x - r; xx <= x + r; ++xx) {
          mx = std::max(mx, R(yy,xx));
        }
      }
      if (v >= mx) peaks.emplace_back(y, x, v);
    }
  }
  return peaks;
}

static const std::array<std::pair<int,int>,16> FAST_CIRCLE = {{
  {0,-3},{1,-3},{2,-2},{3,-1},
  {3,0},{3,1},{2,2},{1,3},
  {0,3},{-1,3},{-2,2},{-3,1},
  {-3,0},{-3,-1},{-2,-2},{-1,-3}
}};

static std::vector<std::tuple<int,int,float>> detect_fast9(const ImageF& img, float threshold, int border) {
  std::vector<std::tuple<int,int,float>> corners;
  const float t = threshold;
  const std::array<int,4> quick = {0,4,8,12};

  for (int y = border; y < img.h - border; ++y) {
    for (int x = border; x < img.w - border; ++x) {
      const float p = img(y,x);

      int brighter = 0, darker = 0;
      for (int qi : quick) {
        const auto [dy, dx] = FAST_CIRCLE[static_cast<size_t>(qi)];
        const float v = img(y + dy, x + dx);
        brighter += (v > p + t);
        darker   += (v < p - t);
      }
      if (brighter < 3 && darker < 3) continue;

      bool hi[16], lo[16];
      float circ[16];
      for (int i = 0; i < 16; ++i) {
        const auto [dy, dx] = FAST_CIRCLE[static_cast<size_t>(i)];
        circ[i] = img(y + dy, x + dx);
        hi[i] = (circ[i] > p + t);
        lo[i] = (circ[i] < p - t);
      }

      bool ok = false;
      for (int s = 0; s < 16; ++s) {
        bool all_hi = true, all_lo = true;
        for (int k = 0; k < 9; ++k) {
          const int idx = (s + k) % 16;
          all_hi = all_hi && hi[idx];
          all_lo = all_lo && lo[idx];
        }
        if (all_hi || all_lo) { ok = true; break; }
      }
      if (!ok) continue;

      float score = 0.0f;
      for (int i = 0; i < 16; ++i) score += std::abs(circ[i] - p);
      corners.emplace_back(y, x, score);
    }
  }
  return corners;
}

static std::vector<Keypoint> detect_keypoints(const ImageF& img, const DetectorConfig& cfg) {
  std::vector<Keypoint> kps;

  if (cfg.kind == DetectorKind::Harris || cfg.kind == DetectorKind::ShiTomasi) {
    for (float s : cfg.sigmas) {
      ImageF R = (cfg.kind == DetectorKind::Harris) ? harris_response(img, s) : shitomasi_response(img, s);

      float mx = 0.0f;
      for (float& v : R.a) { v = std::max(0.0f, v); mx = std::max(mx, v); }
      const float thr = cfg.rel_threshold * (mx + 1e-12f);

      auto peaks = nms_peaks(R, cfg.nms_radius, thr);
      std::sort(peaks.begin(), peaks.end(),
                [](const auto& a, const auto& b){ return std::get<2>(a) > std::get<2>(b); });

      if (static_cast<int>(peaks.size()) > cfg.max_per_scale) peaks.resize(static_cast<size_t>(cfg.max_per_scale));

      for (const auto& p : peaks) {
        const int y = std::get<0>(p);
        const int x = std::get<1>(p);
        const float v = std::get<2>(p);
        kps.push_back(Keypoint{static_cast<float>(x), static_cast<float>(y), s, v, 0.0f});
      }
    }
  } else {
    auto corners = detect_fast9(img, cfg.fast_threshold, cfg.fast_border);
    std::sort(corners.begin(), corners.end(),
              [](const auto& a, const auto& b){ return std::get<2>(a) > std::get<2>(b); });

    std::vector<uint8_t> taken(static_cast<size_t>(img.w*img.h), 0);
    const int r = cfg.nms_radius;

    for (const auto& c : corners) {
      const int y = std::get<0>(c);
      const int x = std::get<1>(c);
      const float sc = std::get<2>(c);

      if (taken[static_cast<size_t>(y*img.w + x)]) continue;

      for (int yy = std::max(0, y - r); yy <= std::min(img.h - 1, y + r); ++yy) {
        for (int xx = std::max(0, x - r); xx <= std::min(img.w - 1, x + r); ++xx) {
          taken[static_cast<size_t>(yy*img.w + xx)] = 1;
        }
      }

      kps.push_back(Keypoint{static_cast<float>(x), static_cast<float>(y), 1.6f, sc, 0.0f});
      if (static_cast<int>(kps.size()) >= cfg.max_total) break;
    }
  }

  std::sort(kps.begin(), kps.end(), [](const Keypoint& a, const Keypoint& b){ return a.response > b.response; });
  if (static_cast<int>(kps.size()) > cfg.max_total) kps.resize(static_cast<size_t>(cfg.max_total));
  return kps;
}

// -----------------------------
// Orientation
// -----------------------------

static std::vector<Keypoint> quick_orientations(const ImageF& img, const std::vector<Keypoint>& kps, float blur_sigma = 1.2f) {
  const ImageF b = gaussian_blur(img, blur_sigma);
  ImageF gx, gy;
  sobel_gradients(b, gx, gy);

  std::vector<Keypoint> out;
  out.reserve(kps.size());

  for (const auto& kp : kps) {
    const int x = static_cast<int>(std::lround(kp.x));
    const int y = static_cast<int>(std::lround(kp.y));
    if (x < 1 || y < 1 || x >= img.w - 1 || y >= img.h - 1) continue;

    float gxx = 0.0f, gyy = 0.0f;
    for (int yy = y - 1; yy <= y + 1; ++yy) {
      for (int xx = x - 1; xx <= x + 1; ++xx) {
        gxx += gx(yy,xx);
        gyy += gy(yy,xx);
      }
    }
    gxx /= 9.0f; gyy /= 9.0f;

    Keypoint k = kp;
    k.angle = static_cast<float>(std::atan2(gyy, gxx));
    out.push_back(k);
  }
  return out;
}

// -----------------------------
// Descriptors
// -----------------------------

static inline float bilinear(const ImageF& img, float x, float y) {
  int x0 = static_cast<int>(std::floor(x));
  int y0 = static_cast<int>(std::floor(y));
  const float ax = x - x0;
  const float ay = y - y0;

  x0 = std::max(0, std::min(img.w - 2, x0));
  y0 = std::max(0, std::min(img.h - 2, y0));

  const float v00 = img(y0, x0);
  const float v10 = img(y0, x0 + 1);
  const float v01 = img(y0 + 1, x0);
  const float v11 = img(y0 + 1, x0 + 1);
  return (1 - ax)*(1 - ay)*v00 + ax*(1 - ay)*v10 + (1 - ax)*ay*v01 + ax*ay*v11;
}

struct DescF { std::vector<float> v; };
struct DescB { std::vector<uint8_t> b; };

static bool sift128_descriptor(
    const ImageF& mag, const ImageF& ang, const Keypoint& kp, const SiftConfig& cfg, std::vector<float>& out) {
  const int win = cfg.num_cells * cfg.cell_size; // e.g. 16
  const float half = 0.5f * win;
  const float step = std::max(1.0f, cfg.step_scale * kp.sigma);

  const float rad = (half + 1.0f) * step;
  if (kp.x - rad < 1.0f || kp.y - rad < 1.0f || kp.x + rad >= mag.w - 2.0f || kp.y + rad >= mag.h - 2.0f)
    return false;

  out.assign(static_cast<size_t>(cfg.num_cells * cfg.num_cells * cfg.num_bins), 0.0f);

  const float ca = std::cos(kp.angle);
  const float sa = std::sin(kp.angle);

  const float w_sigma = 0.5f * win * step;
  const float inv2ws2 = 1.0f / (2.0f*w_sigma*w_sigma + 1e-12f);

  for (int iy = 0; iy < win; ++iy) {
    for (int ix = 0; ix < win; ++ix) {
      const float xx = (ix + 0.5f - half);
      const float yy = (iy + 0.5f - half);

      const float dx = (ca*xx - sa*yy) * step;
      const float dy = (sa*xx + ca*yy) * step;

      const float xs = kp.x + dx;
      const float ys = kp.y + dy;

      const float m = bilinear(mag, xs, ys);
      const float a = bilinear(ang, xs, ys);

      float rel = a - kp.angle;
      rel = std::fmod(rel + 2.0f*static_cast<float>(M_PI), 2.0f*static_cast<float>(M_PI));

      const float w = std::exp(-(dx*dx + dy*dy) * inv2ws2);
      const float mm = m * w;

      const int ci = iy / cfg.cell_size;
      const int cj = ix / cfg.cell_size;

      const float binf = (rel / (2.0f*static_cast<float>(M_PI))) * cfg.num_bins;
      const int b0 = static_cast<int>(std::floor(binf)) % cfg.num_bins;
      const float frac = binf - std::floor(binf);
      const int b1 = (b0 + 1) % cfg.num_bins;

      const int base = (ci * cfg.num_cells + cj) * cfg.num_bins;
      out[static_cast<size_t>(base + b0)] += (1.0f - frac) * mm;
      out[static_cast<size_t>(base + b1)] += frac * mm;
    }
  }

  auto l2 = [&](const std::vector<float>& v)->float{
    double s = 0.0;
    for (float x : v) s += x*x;
    return static_cast<float>(std::sqrt(s) + 1e-12);
  };

  float n = l2(out);
  for (float& x : out) x /= n;
  for (float& x : out) x = std::min(0.2f, std::max(0.0f, x));
  n = l2(out);
  for (float& x : out) x /= n;

  return true;
}

static bool patch_descriptor(const ImageF& blur_img, const Keypoint& kp, const PatchConfig& cfg, std::vector<float>& out) {
  const int r = cfg.patch_radius;
  const int x0 = static_cast<int>(std::lround(kp.x));
  const int y0 = static_cast<int>(std::lround(kp.y));
  if (x0 - r < 1 || y0 - r < 1 || x0 + r >= blur_img.w - 1 || y0 + r >= blur_img.h - 1) return false;

  const int side = 2*r + 1;
  out.assign(static_cast<size_t>(side*side), 0.0f);

  double mean = 0.0;
  int idx = 0;
  for (int yy = -r; yy <= r; ++yy) {
    for (int xx = -r; xx <= r; ++xx) {
      const float v = blur_img(y0 + yy, x0 + xx);
      out[static_cast<size_t>(idx++)] = v;
      mean += v;
    }
  }
  mean /= static_cast<double>(side*side);

  double norm2 = 0.0;
  for (float& v : out) {
    v = static_cast<float>(v - mean);
    norm2 += static_cast<double>(v) * v;
  }
  const float inv = 1.0f / static_cast<float>(std::sqrt(norm2) + 1e-12);
  for (float& v : out) v *= inv;
  return true;
}

static std::vector<std::array<int,4>> brief_pairs(int nbits, int radius, int seed) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::uniform_int_distribution<int> dist(-radius, radius);
  std::vector<std::array<int,4>> p(static_cast<size_t>(nbits));
  for (int i = 0; i < nbits; ++i) p[static_cast<size_t>(i)] = {dist(rng), dist(rng), dist(rng), dist(rng)};
  return p;
}

static bool brief_descriptor(
    const ImageF& blur_img, const Keypoint& kp, const BriefConfig& cfg,
    const std::vector<std::array<int,4>>& pairs, std::vector<uint8_t>& out) {
  const int r = cfg.patch_radius;
  if (kp.x - r - 1 < 0 || kp.y - r - 1 < 0 || kp.x + r + 2 >= blur_img.w || kp.y + r + 2 >= blur_img.h)
    return false;

  const int nbits = cfg.nbits;
  std::vector<uint8_t> bits(static_cast<size_t>(nbits), 0);

  const float ca = cfg.rotate ? std::cos(kp.angle) : 1.0f;
  const float sa = cfg.rotate ? std::sin(kp.angle) : 0.0f;

  for (int i = 0; i < nbits; ++i) {
    const auto& pr = pairs[static_cast<size_t>(i)];
    float x1 = static_cast<float>(pr[0]);
    float y1 = static_cast<float>(pr[1]);
    float x2 = static_cast<float>(pr[2]);
    float y2 = static_cast<float>(pr[3]);

    if (cfg.rotate) {
      const float rx1 = ca*x1 - sa*y1;
      const float ry1 = sa*x1 + ca*y1;
      const float rx2 = ca*x2 - sa*y2;
      const float ry2 = sa*x2 + ca*y2;
      x1 = rx1; y1 = ry1; x2 = rx2; y2 = ry2;
    }

    const float v1 = bilinear(blur_img, kp.x + x1, kp.y + y1);
    const float v2 = bilinear(blur_img, kp.x + x2, kp.y + y2);
    bits[static_cast<size_t>(i)] = (v1 < v2) ? 1 : 0;
  }

  const int nbytes = nbits / 8;
  out.assign(static_cast<size_t>(nbytes), 0);
  for (int i = 0; i < nbits; ++i) {
    const int byte = i / 8;
    const int bit  = i % 8;
    out[static_cast<size_t>(byte)] |= static_cast<uint8_t>(bits[static_cast<size_t>(i)] << bit);
  }
  return true;
}

static void compute_descriptors(
    const ImageF& img, const std::vector<Keypoint>& kps, const DescriptorConfig& cfg,
    std::vector<Keypoint>& kept_kps,
    std::vector<DescF>& desc_f,
    std::vector<DescB>& desc_b,
    MetricKind& metric) {

  kept_kps.clear();
  desc_f.clear();
  desc_b.clear();

  if (cfg.kind == DescriptorKind::SIFT128) {
    metric = MetricKind::L2;

    std::vector<float> sigs;
    sigs.reserve(kps.size());
    for (const auto& k : kps) sigs.push_back(k.sigma);
    std::sort(sigs.begin(), sigs.end());
    sigs.erase(std::unique(sigs.begin(), sigs.end()), sigs.end());

    struct GradCache { float sigma; ImageF mag; ImageF ang; };
    std::vector<GradCache> cache;
    cache.reserve(sigs.size());

    for (float s : sigs) {
      const ImageF b = gaussian_blur(img, s);
      ImageF gx, gy;
      sobel_gradients(b, gx, gy);
      ImageF mag(img.h, img.w), ang(img.h, img.w);
      for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
          const float vx = gx(y,x);
          const float vy = gy(y,x);
          mag(y,x) = std::sqrt(vx*vx + vy*vy);
          ang(y,x) = std::atan2(vy, vx);
        }
      }
      cache.push_back(GradCache{s, std::move(mag), std::move(ang)});
    }

    auto find_cache = [&](float s)->const GradCache*{
      for (const auto& c : cache) if (std::abs(c.sigma - s) < 1e-6f) return &c;
      return nullptr;
    };

    for (const auto& kp : kps) {
      const auto* c = find_cache(kp.sigma);
      if (!c) continue;

      std::vector<float> d;
      if (!sift128_descriptor(c->mag, c->ang, kp, cfg.sift, d)) continue;

      kept_kps.push_back(kp);
      desc_f.push_back(DescF{std::move(d)});
    }
    return;
  }

  if (cfg.kind == DescriptorKind::Patch) {
    metric = MetricKind::L2;
    const ImageF blur_img = (cfg.patch.blur_sigma > 0.0f) ? gaussian_blur(img, cfg.patch.blur_sigma) : img;

    for (const auto& kp : kps) {
      std::vector<float> d;
      if (!patch_descriptor(blur_img, kp, cfg.patch, d)) continue;
      kept_kps.push_back(kp);
      desc_f.push_back(DescF{std::move(d)});
    }
    return;
  }

  metric = MetricKind::Hamming;
  const ImageF blur_img = (cfg.brief.blur_sigma > 0.0f) ? gaussian_blur(img, cfg.brief.blur_sigma) : img;
  const auto pairs = brief_pairs(cfg.brief.nbits, cfg.brief.patch_radius, cfg.brief.seed);

  for (const auto& kp : kps) {
    std::vector<uint8_t> d;
    if (!brief_descriptor(blur_img, kp, cfg.brief, pairs, d)) continue;
    kept_kps.push_back(kp);
    desc_b.push_back(DescB{std::move(d)});
  }
}

// -----------------------------
// Matching
// -----------------------------

struct Match {
  int i = -1;
  int j = -1;
  float dist = 0.0f;
};

static float l2_dist(const std::vector<float>& a, const std::vector<float>& b) {
  const size_t n = std::min(a.size(), b.size());
  double s = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    s += d*d;
  }
  return static_cast<float>(std::sqrt(s));
}

static int popcount8(uint8_t x) {
  static const uint8_t table[256] = {
    0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    1,2,2,3,2,3,3,4,2,3,3,4,3,4,4,5,2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    2,3,3,4,3,4,4,5,3,4,4,5,4,5,5,6,3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,
    3,4,4,5,4,5,5,6,4,5,5,6,5,6,6,7,4,5,5,6,5,6,6,7,5,6,6,7,6,7,7,8
  };
  return table[x];
}

static int hamming_dist(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  int s = 0;
  for (size_t i = 0; i < n; ++i) s += popcount8(static_cast<uint8_t>(a[i] ^ b[i]));
  return s;
}

static std::vector<Match> match_ratio_mutual(
    const std::vector<DescF>& D1f, const std::vector<DescF>& D2f,
    const std::vector<DescB>& D1b, const std::vector<DescB>& D2b,
    MetricKind metric, const MatchConfig& cfg) {

  std::vector<Match> matches;
  const int N1 = (metric == MetricKind::L2) ? static_cast<int>(D1f.size()) : static_cast<int>(D1b.size());
  const int N2 = (metric == MetricKind::L2) ? static_cast<int>(D2f.size()) : static_cast<int>(D2b.size());
  if (N1 == 0 || N2 == 0) return matches;

  std::vector<int> best_j(static_cast<size_t>(N1), -1);
  std::vector<float> best_d(static_cast<size_t>(N1), std::numeric_limits<float>::infinity());
  std::vector<float> second_d(static_cast<size_t>(N1), std::numeric_limits<float>::infinity());

  for (int i = 0; i < N1; ++i) {
    for (int j = 0; j < N2; ++j) {
      float d = 0.0f;
      if (metric == MetricKind::L2) d = l2_dist(D1f[static_cast<size_t>(i)].v, D2f[static_cast<size_t>(j)].v);
      else d = static_cast<float>(hamming_dist(D1b[static_cast<size_t>(i)].b, D2b[static_cast<size_t>(j)].b));

      if (d < best_d[static_cast<size_t>(i)]) {
        second_d[static_cast<size_t>(i)] = best_d[static_cast<size_t>(i)];
        best_d[static_cast<size_t>(i)] = d;
        best_j[static_cast<size_t>(i)] = j;
      } else if (d < second_d[static_cast<size_t>(i)]) {
        second_d[static_cast<size_t>(i)] = d;
      }
    }
  }

  for (int i = 0; i < N1; ++i) {
    const int j = best_j[static_cast<size_t>(i)];
    if (j < 0) continue;
    const float d1 = best_d[static_cast<size_t>(i)];
    const float d2 = second_d[static_cast<size_t>(i)] + 1e-12f;
    if (d1 <= cfg.ratio * d2) matches.push_back(Match{i, j, d1});
  }

  if (cfg.mutual && !matches.empty()) {
    std::vector<int> best_i_for_j(static_cast<size_t>(N2), -1);
    std::vector<float> best_d_for_j(static_cast<size_t>(N2), std::numeric_limits<float>::infinity());

    for (int j = 0; j < N2; ++j) {
      for (int i = 0; i < N1; ++i) {
        float d = 0.0f;
        if (metric == MetricKind::L2) d = l2_dist(D1f[static_cast<size_t>(i)].v, D2f[static_cast<size_t>(j)].v);
        else d = static_cast<float>(hamming_dist(D1b[static_cast<size_t>(i)].b, D2b[static_cast<size_t>(j)].b));
        if (d < best_d_for_j[static_cast<size_t>(j)]) {
          best_d_for_j[static_cast<size_t>(j)] = d;
          best_i_for_j[static_cast<size_t>(j)] = i;
        }
      }
    }

    std::vector<Match> filtered;
    filtered.reserve(matches.size());
    for (const auto& m : matches) {
      if (best_i_for_j[static_cast<size_t>(m.j)] == m.i) filtered.push_back(m);
    }
    matches.swap(filtered);
  }

  std::sort(matches.begin(), matches.end(), [](const Match& a, const Match& b){ return a.dist < b.dist; });
  if (cfg.top_k > 0 && static_cast<int>(matches.size()) > cfg.top_k) matches.resize(static_cast<size_t>(cfg.top_k));
  return matches;
}

// -----------------------------
// Model fitting: normalization
// -----------------------------

static Mat33 norm_matrix_for_points(const std::vector<Vec2>& pts) {
  Vec2 mean{0,0};
  for (const auto& p : pts) { mean.x += p.x; mean.y += p.y; }
  mean.x /= (pts.empty() ? 1.0 : static_cast<double>(pts.size()));
  mean.y /= (pts.empty() ? 1.0 : static_cast<double>(pts.size()));

  double rms = 0.0;
  for (const auto& p : pts) rms += sqr(p.x - mean.x) + sqr(p.y - mean.y);
  rms = std::sqrt(rms / (pts.empty() ? 1.0 : static_cast<double>(pts.size())) + 1e-12);

  const double s = std::sqrt(2.0) / (rms + 1e-12);
  Mat33 T;
  T(0,0) = s; T(0,1) = 0; T(0,2) = -s * mean.x;
  T(1,0) = 0; T(1,1) = s; T(1,2) = -s * mean.y;
  T(2,0) = 0; T(2,1) = 0; T(2,2) = 1;
  return T;
}

static std::vector<Vec2> apply_norm(const Mat33& T, const std::vector<Vec2>& pts) {
  std::vector<Vec2> out;
  out.reserve(pts.size());
  for (const auto& p : pts) out.push_back(proj(T, p));
  return out;
}

// -----------------------------
// Fit similarity / affine / homography
// -----------------------------

static int min_samples(ModelKind m) {
  switch (m) {
    case ModelKind::Similarity: return 2;
    case ModelKind::Affine:     return 3;
    case ModelKind::Homography: return 4;
  }
  return 4;
}

static bool fit_similarity_ls(const std::vector<Vec2>& p1, const std::vector<Vec2>& p2, Mat33& H) {
  const int n = static_cast<int>(p1.size());
  if (n < 2) return false;

  Vec2 muX{0,0}, muY{0,0};
  for (int i = 0; i < n; ++i) { muX.x += p1[i].x; muX.y += p1[i].y; muY.x += p2[i].x; muY.y += p2[i].y; }
  muX.x /= n; muX.y /= n; muY.x /= n; muY.y /= n;

  double s00=0, s01=0, s10=0, s11=0;
  double varX = 0.0;
  for (int i = 0; i < n; ++i) {
    const double x0 = p1[i].x - muX.x;
    const double x1 = p1[i].y - muX.y;
    const double y0 = p2[i].x - muY.x;
    const double y1 = p2[i].y - muY.y;
    s00 += y0*x0; s01 += y0*x1;
    s10 += y1*x0; s11 += y1*x1;
    varX += x0*x0 + x1*x1;
  }
  s00 /= n; s01 /= n; s10 /= n; s11 /= n;
  varX /= n;
  if (varX < 1e-12) return false;

  // Polar decomposition in 2D.
  const double a00 = s00*s00 + s10*s10;
  const double a01 = s00*s01 + s10*s11;
  const double a11 = s01*s01 + s11*s11;

  const double tr = a00 + a11;
  const double det = a00*a11 - a01*a01;
  const double disc = std::max(0.0, tr*tr - 4.0*det);
  const double l0 = 0.5 * (tr + std::sqrt(disc));
  const double l1 = 0.5 * (tr - std::sqrt(disc));

  double q00=1, q01=0, q10=0, q11=1;
  if (std::abs(a01) > 1e-18) {
    const double v0 = a01;
    const double v1 = l0 - a00;
    const double nrm = std::sqrt(v0*v0 + v1*v1) + 1e-12;
    q00 = v0 / nrm; q10 = v1 / nrm;
    q01 = -q10; q11 = q00;
  }

  const double il0 = 1.0 / std::sqrt(std::max(l0, 1e-18));
  const double il1 = 1.0 / std::sqrt(std::max(l1, 1e-18));

  const double is00 = q00*q00*il0 + q01*q01*il1;
  const double is01 = q00*q10*il0 + q01*q11*il1;
  const double is10 = is01;
  const double is11 = q10*q10*il0 + q11*q11*il1;

  double r00 = s00*is00 + s01*is10;
  double r01 = s00*is01 + s01*is11;
  double r10 = s10*is00 + s11*is10;
  double r11 = s10*is01 + s11*is11;

  const double detR = r00*r11 - r01*r10;
  if (detR < 0) { r01 = -r01; r11 = -r11; }

  const double trRS = r00*s00 + r01*s01 + r10*s10 + r11*s11;
  const double sc = trRS / (varX + 1e-12);

  const double tx = muY.x - sc*(r00*muX.x + r01*muX.y);
  const double ty = muY.y - sc*(r10*muX.x + r11*muX.y);

  H = Mat33::Identity();
  H(0,0) = sc*r00; H(0,1) = sc*r01; H(0,2) = tx;
  H(1,0) = sc*r10; H(1,1) = sc*r11; H(1,2) = ty;
  return true;
}

static bool fit_affine_ls(const std::vector<Vec2>& p1, const std::vector<Vec2>& p2, Mat33& H) {
  const int n = static_cast<int>(p1.size());
  if (n < 3) return false;

  const Mat33 T1 = norm_matrix_for_points(p1);
  const Mat33 T2 = norm_matrix_for_points(p2);
  const std::vector<Vec2> x1 = apply_norm(T1, p1);
  const std::vector<Vec2> x2 = apply_norm(T2, p2);

  const int M = 2*n;
  const int N = 6;
  std::vector<double> A(static_cast<size_t>(M*N), 0.0);
  std::vector<double> b(static_cast<size_t>(M), 0.0);

  for (int i = 0; i < n; ++i) {
    const double x = x1[i].x;
    const double y = x1[i].y;
    const double u = x2[i].x;
    const double v = x2[i].y;

    A[(2*i+0)*N + 0] = x;
    A[(2*i+0)*N + 1] = y;
    A[(2*i+0)*N + 2] = 1.0;
    b[static_cast<size_t>(2*i+0)] = u;

    A[(2*i+1)*N + 3] = x;
    A[(2*i+1)*N + 4] = y;
    A[(2*i+1)*N + 5] = 1.0;
    b[static_cast<size_t>(2*i+1)] = v;
  }

  std::vector<double> p;
  if (!least_squares_normal(A, b, M, N, p)) return false;

  Mat33 An = Mat33::Identity();
  An(0,0) = p[0]; An(0,1) = p[1]; An(0,2) = p[2];
  An(1,0) = p[3]; An(1,1) = p[4]; An(1,2) = p[5];

  Mat33 invT2;
  if (!invert(T2, invT2)) return false;
  H = mul(mul(invT2, An), T1);

  const double s = H(2,2);
  for (double& v : H.m) v /= (s + 1e-12);
  return true;
}

static bool fit_homography_ls(const std::vector<Vec2>& p1, const std::vector<Vec2>& p2, Mat33& H) {
  const int n = static_cast<int>(p1.size());
  if (n < 4) return false;

  const Mat33 T1 = norm_matrix_for_points(p1);
  const Mat33 T2 = norm_matrix_for_points(p2);
  const std::vector<Vec2> x1 = apply_norm(T1, p1);
  const std::vector<Vec2> x2 = apply_norm(T2, p2);

  const int M = 2*n;
  const int N = 8;
  std::vector<double> A(static_cast<size_t>(M*N), 0.0);
  std::vector<double> b(static_cast<size_t>(M), 0.0);

  for (int i = 0; i < n; ++i) {
    const double x = x1[i].x;
    const double y = x1[i].y;
    const double u = x2[i].x;
    const double v = x2[i].y;

    A[(2*i+0)*N + 0] = x;
    A[(2*i+0)*N + 1] = y;
    A[(2*i+0)*N + 2] = 1.0;
    A[(2*i+0)*N + 6] = -u*x;
    A[(2*i+0)*N + 7] = -u*y;
    b[static_cast<size_t>(2*i+0)] = u;

    A[(2*i+1)*N + 3] = x;
    A[(2*i+1)*N + 4] = y;
    A[(2*i+1)*N + 5] = 1.0;
    A[(2*i+1)*N + 6] = -v*x;
    A[(2*i+1)*N + 7] = -v*y;
    b[static_cast<size_t>(2*i+1)] = v;
  }

  std::vector<double> p;
  if (!least_squares_normal(A, b, M, N, p)) return false;

  Mat33 Hn = Mat33::Identity();
  Hn(0,0) = p[0]; Hn(0,1) = p[1]; Hn(0,2) = p[2];
  Hn(1,0) = p[3]; Hn(1,1) = p[4]; Hn(1,2) = p[5];
  Hn(2,0) = p[6]; Hn(2,1) = p[7]; Hn(2,2) = 1.0;

  Mat33 invT2;
  if (!invert(T2, invT2)) return false;
  H = mul(mul(invT2, Hn), T1);

  const double s = H(2,2);
  for (double& v : H.m) v /= (s + 1e-12);
  return true;
}

static bool fit_model_ls(ModelKind model, const std::vector<Vec2>& p1, const std::vector<Vec2>& p2, Mat33& H) {
  switch (model) {
    case ModelKind::Similarity: return fit_similarity_ls(p1, p2, H);
    case ModelKind::Affine:     return fit_affine_ls(p1, p2, H);
    case ModelKind::Homography: return fit_homography_ls(p1, p2, H);
  }
  return false;
}

// -----------------------------
// RANSAC
// -----------------------------

static std::vector<float> forward_error(const Mat33& H, const std::vector<Vec2>& p1, const std::vector<Vec2>& p2) {
  std::vector<float> e(p1.size(), 0.0f);
  for (size_t i = 0; i < p1.size(); ++i) {
    const Vec2 q = proj(H, p1[i]);
    const double dx = q.x - p2[i].x;
    const double dy = q.y - p2[i].y;
    e[i] = static_cast<float>(std::sqrt(dx*dx + dy*dy));
  }
  return e;
}

static std::vector<float> symmetric_error(const Mat33& H, const std::vector<Vec2>& p1, const std::vector<Vec2>& p2) {
  Mat33 invH;
  if (!invert(H, invH)) return forward_error(H, p1, p2);

  std::vector<float> e(p1.size(), 0.0f);
  for (size_t i = 0; i < p1.size(); ++i) {
    const Vec2 q = proj(H, p1[i]);
    const Vec2 r = proj(invH, p2[i]);
    const double dx1 = q.x - p2[i].x;
    const double dy1 = q.y - p2[i].y;
    const double dx2 = r.x - p1[i].x;
    const double dy2 = r.y - p1[i].y;
    e[i] = static_cast<float>(std::sqrt(dx1*dx1 + dy1*dy1) + std::sqrt(dx2*dx2 + dy2*dy2));
  }
  return e;
}

struct RansacResult {
  Mat33 H;
  std::vector<uint8_t> inliers;
};

static bool ransac_fit(
    const std::vector<Vec2>& p1, const std::vector<Vec2>& p2,
    ModelKind model, const RansacConfig& cfg,
    const std::vector<int>& prior_order,
    RansacResult& out) {

  const int N = static_cast<int>(p1.size());
  const int m = min_samples(model);
  if (N < m) return false;

  std::mt19937 rng(static_cast<uint32_t>(cfg.seed));

  auto sample_indices = [&](int pool)->std::vector<int>{
    std::vector<int> idx;
    idx.reserve(static_cast<size_t>(m));
    std::uniform_int_distribution<int> dist(0, pool - 1);
    while (static_cast<int>(idx.size()) < m) {
      const int pick = prior_order.empty() ? dist(rng) : prior_order[static_cast<size_t>(dist(rng))];
      if (std::find(idx.begin(), idx.end(), pick) == idx.end()) idx.push_back(pick);
    }
    return idx;
  };

  Mat33 bestH = Mat33::Identity();
  std::vector<uint8_t> bestIn(static_cast<size_t>(N), 0);
  int bestCnt = -1;
  double bestCost = std::numeric_limits<double>::infinity();

  for (int it = 0; it < cfg.iters; ++it) {
    int pool = N;
    if (cfg.prosac) {
      pool = std::min(N, cfg.prosac_min_pool + it / std::max(1, cfg.prosac_step));
      pool = std::max(pool, m);
    }

    const auto idx = sample_indices(pool);
    std::vector<Vec2> s1, s2;
    s1.reserve(static_cast<size_t>(m));
    s2.reserve(static_cast<size_t>(m));
    for (int k : idx) { s1.push_back(p1[static_cast<size_t>(k)]); s2.push_back(p2[static_cast<size_t>(k)]); }

    Mat33 H;
    if (!fit_model_ls(model, s1, s2, H)) continue;

    const auto errs = cfg.symmetric ? symmetric_error(H, p1, p2) : forward_error(H, p1, p2);

    std::vector<uint8_t> in(static_cast<size_t>(N), 0);
    int cnt = 0;
    double cost = 0.0;
    const double thr2 = static_cast<double>(cfg.threshold_px) * cfg.threshold_px;

    for (int i = 0; i < N; ++i) {
      const double e = errs[static_cast<size_t>(i)];
      if (e < cfg.threshold_px) { in[static_cast<size_t>(i)] = 1; ++cnt; }
      cost += std::min(e*e, thr2);
    }

    if (cnt > bestCnt || (cnt == bestCnt && cost < bestCost)) {
      bestH = H;
      bestIn.swap(in);
      bestCnt = cnt;
      bestCost = cost;
    }
  }

  if (bestCnt < m) return false;

  std::vector<Vec2> in1, in2;
  for (int i = 0; i < N; ++i) {
    if (!bestIn[static_cast<size_t>(i)]) continue;
    in1.push_back(p1[static_cast<size_t>(i)]);
    in2.push_back(p2[static_cast<size_t>(i)]);
  }
  Mat33 refined;
  if (!fit_model_ls(model, in1, in2, refined)) refined = bestH;

  out.H = refined;
  out.inliers = std::move(bestIn);
  return true;
}

// -----------------------------
// Warping for visualization
// -----------------------------

static ImageF warp_perspective(const ImageF& src, const Mat33& H, int out_h, int out_w) {
  Mat33 invH;
  if (!invert(H, invH)) invH = Mat33::Identity();

  ImageF out(out_h, out_w);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const Vec2 q = proj(invH, Vec2{static_cast<double>(x), static_cast<double>(y)});
      const float xf = static_cast<float>(q.x);
      const float yf = static_cast<float>(q.y);

      if (xf >= 0.0f && yf >= 0.0f && xf < src.w - 1.0f && yf < src.h - 1.0f) {
        out(y,x) = bilinear(src, xf, yf);
      } else {
        out(y,x) = 0.0f;
      }
    }
  }
  return out;
}

// -----------------------------
// Full registration
// -----------------------------

struct RegistrationResult {
  Mat33 H;
  std::vector<Match> matches;
  std::vector<uint8_t> inliers;
  float mean_inlier_error = 0.0f;
};

static bool register_pair(const ImageF& img1, const ImageF& img2, const RegistrationConfig& cfg, RegistrationResult& out) {
  auto k1 = detect_keypoints(img1, cfg.detector);
  auto k2 = detect_keypoints(img2, cfg.detector);

  if (cfg.quick_orientation) {
    k1 = quick_orientations(img1, k1, 1.2f);
    k2 = quick_orientations(img2, k2, 1.2f);
  }

  std::vector<Keypoint> kk1, kk2;
  std::vector<DescF> D1f, D2f;
  std::vector<DescB> D1b, D2b;
  MetricKind metric = MetricKind::L2;

  compute_descriptors(img1, k1, cfg.descriptor, kk1, D1f, D1b, metric);
  compute_descriptors(img2, k2, cfg.descriptor, kk2, D2f, D2b, metric);

  auto matches = match_ratio_mutual(D1f, D2f, D1b, D2b, metric, cfg.matcher);
  if (static_cast<int>(matches.size()) < std::max(8, 2*min_samples(cfg.model))) return false;

  std::vector<Vec2> p1, p2;
  p1.reserve(matches.size());
  p2.reserve(matches.size());

  std::vector<int> prior(matches.size());
  for (size_t i = 0; i < matches.size(); ++i) prior[i] = static_cast<int>(i);
  std::sort(prior.begin(), prior.end(), [&](int a, int b){ return matches[static_cast<size_t>(a)].dist < matches[static_cast<size_t>(b)].dist; });

  for (const auto& m : matches) {
    const auto& a = kk1[static_cast<size_t>(m.i)];
    const auto& b = kk2[static_cast<size_t>(m.j)];
    p1.push_back(Vec2{a.x, a.y});
    p2.push_back(Vec2{b.x, b.y});
  }

  RansacResult rr;
  if (!ransac_fit(p1, p2, cfg.model, cfg.ransac, prior, rr)) return false;

  const auto errs = cfg.ransac.symmetric ? symmetric_error(rr.H, p1, p2) : forward_error(rr.H, p1, p2);
  double sum = 0.0;
  int cnt = 0;
  for (size_t i = 0; i < errs.size(); ++i) {
    if (!rr.inliers[i]) continue;
    sum += errs[i];
    ++cnt;
  }

  out.H = rr.H;
  out.matches = std::move(matches);
  out.inliers = std::move(rr.inliers);
  out.mean_inlier_error = (cnt > 0) ? static_cast<float>(sum / cnt) : 0.0f;
  return true;
}

// -----------------------------
// Synthetic demo helpers
// -----------------------------

static ImageF make_synth(int h = 200, int w = 260, int seed = 0) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::normal_distribution<float> n(0.0f, 0.07f);

  ImageF img(h, w);
  for (int i = 0; i < h*w; ++i) img.a[static_cast<size_t>(i)] = std::min(1.0f, std::max(0.0f, 0.25f + n(rng)));

  auto clamp01 = [](float x){ return std::min(1.0f, std::max(0.0f, x)); };

  std::uniform_int_distribution<int> ydist(0, h - 20);
  std::uniform_int_distribution<int> xdist(0, w - 20);
  std::uniform_int_distribution<int> hdist(8, 40);
  std::uniform_int_distribution<int> wdist(8, 55);
  std::uniform_real_distribution<float> vdist(0.25f, 0.9f);

  for (int k = 0; k < 22; ++k) {
    const int y0 = ydist(rng);
    const int x0 = xdist(rng);
    const int hh = hdist(rng);
    const int ww = wdist(rng);
    const float val = vdist(rng);
    for (int y = y0; y < std::min(h, y0 + hh); ++y) {
      for (int x = x0; x < std::min(w, x0 + ww); ++x) {
        img(y,x) = clamp01(img(y,x) + val);
      }
    }
  }

  std::uniform_real_distribution<float> cyd(25.0f, h - 25.0f);
  std::uniform_real_distribution<float> cxd(25.0f, w - 25.0f);
  std::uniform_real_distribution<float> rd(8.0f, 24.0f);
  std::uniform_real_distribution<float> ampd(0.2f, 0.85f);
  std::uniform_real_distribution<float> offd(0.1f, 0.4f);

  for (int k = 0; k < 8; ++k) {
    const float cy = cyd(rng);
    const float cx = cxd(rng);
    const float r = rd(rng);
    const float a = ampd(rng);
    const float b = offd(rng);
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const float dx = x - cx;
        const float dy = y - cy;
        if (dx*dx + dy*dy <= r*r) img(y,x) = clamp01(img(y,x)*a + b);
      }
    }
  }

  std::uniform_int_distribution<int> yld(0, h - 1);
  std::uniform_int_distribution<int> xld(0, w - 1);
  std::uniform_real_distribution<float> lvd(0.35f, 0.9f);

  for (int k = 0; k < 10; ++k) {
    const int y0 = yld(rng), x0 = xld(rng);
    const int y1 = yld(rng), x1 = xld(rng);
    const int npts = std::max(std::abs(y1 - y0), std::abs(x1 - x0)) + 1;
    const float val = lvd(rng);
    for (int i = 0; i < npts; ++i) {
      const float t = (npts == 1) ? 0.0f : static_cast<float>(i) / (npts - 1);
      const int yy = static_cast<int>(std::lround((1 - t)*y0 + t*y1));
      const int xx = static_cast<int>(std::lround((1 - t)*x0 + t*x1));
      img(yy,xx) = clamp01(img(yy,xx) + val);
    }
  }

  img = gaussian_blur(img, 1.0f);
  return img;
}

static Mat33 random_homography(int w, int h, int seed = 7) {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::uniform_real_distribution<double> angd(-0.22, 0.22);
  std::uniform_real_distribution<double> scd(0.9, 1.12);
  std::uniform_real_distribution<double> txd(-18.0, 18.0);
  std::uniform_real_distribution<double> tyd(-14.0, 14.0);
  std::uniform_real_distribution<double> pd(-1.2e-4, 1.2e-4);

  const double ang = angd(rng);
  const double sc = scd(rng);
  const double tx = txd(rng);
  const double ty = tyd(rng);
  const double ca = std::cos(ang);
  const double sa = std::sin(ang);

  Mat33 A = Mat33::Identity();
  A(0,0) = sc*ca; A(0,1) = -sc*sa; A(0,2) = tx;
  A(1,0) = sc*sa; A(1,1) =  sc*ca; A(1,2) = ty;

  Mat33 P = Mat33::Identity();
  P(2,0) = pd(rng);
  P(2,1) = pd(rng);

  Mat33 C1 = Mat33::Identity();
  C1(0,2) = -w / 2.0;
  C1(1,2) = -h / 2.0;

  Mat33 C2 = Mat33::Identity();
  C2(0,2) =  w / 2.0;
  C2(1,2) =  h / 2.0;

  Mat33 H = mul(C2, mul(P, mul(A, C1)));
  const double s = H(2,2);
  for (double& v : H.m) v /= (s + 1e-12);
  return H;
}

static double corner_error(const Mat33& H_est, const Mat33& H_gt, int w, int h) {
  const std::array<Vec2,4> c = {
    Vec2{0,0},
    Vec2{static_cast<double>(w-1),0},
    Vec2{static_cast<double>(w-1), static_cast<double>(h-1)},
    Vec2{0, static_cast<double>(h-1)}
  };
  double s = 0.0;
  for (const auto& p : c) {
    const Vec2 pe = proj(H_est, p);
    const Vec2 pg = proj(H_gt, p);
    const double dx = pe.x - pg.x;
    const double dy = pe.y - pg.y;
    s += std::sqrt(dx*dx + dy*dy);
  }
  return s / 4.0;
}

static void print_mat(const Mat33& H) {
  std::cout << std::fixed << std::setprecision(6);
  for (int r = 0; r < 3; ++r) {
    std::cout << "  ";
    for (int c = 0; c < 3; ++c) std::cout << std::setw(12) << H(r,c) << " ";
    std::cout << "\n";
  }
}

} // namespace reg2d

int main(int argc, char** argv) {
  using namespace reg2d;

  ImageF img1, img2;
  Mat33 H_gt = Mat33::Identity();
  bool has_gt = false;

  if (argc >= 3) {
    if (!read_pgm(argv[1], img1) || !read_pgm(argv[2], img2)) {
      std::cerr << "Failed to read PGM inputs.\n";
      return 1;
    }
  } else {
    img1 = make_synth(200, 260, 0);
    H_gt = random_homography(img1.w, img1.h, 7);
    img2 = warp_perspective(img1, H_gt, img1.h, img1.w);
    has_gt = true;

    std::mt19937 rng(123);
    std::normal_distribution<float> n(0.0f, 0.03f);
    for (float& v : img2.a) v = std::min(1.0f, std::max(0.0f, v*1.10f + 0.04f + n(rng)));
    for (int y = 110; y < 165; ++y) for (int x = 20; x < 95; ++x) img2(y,x) = 0.0f;

    write_pgm("demo_base.pgm", img1);
    write_pgm("demo_target.pgm", img2);
  }

  RegistrationConfig cfg;
  cfg.detector.kind = DetectorKind::ShiTomasi;
  cfg.descriptor.kind = DescriptorKind::SIFT128;
  cfg.model = ModelKind::Homography;
  cfg.ransac.threshold_px = 3.2f;

  RegistrationResult res;
  if (!register_pair(img1, img2, cfg, res)) {
    std::cerr << "Registration failed (insufficient matches or RANSAC failure).\n";
    return 2;
  }

  int inl = 0;
  for (uint8_t v : res.inliers) inl += (v != 0);
  std::cout << "Matches: " << res.matches.size() << "\n";
  std::cout << "Inliers: " << inl << " (ratio " << (res.matches.empty() ? 0.0 : double(inl)/res.matches.size()) << ")\n";
  std::cout << "Mean inlier error (px): " << res.mean_inlier_error << "\n";
  std::cout << "Estimated H:\n";
  print_mat(res.H);

  if (has_gt) {
    std::cout << "Corner error vs GT (px): " << corner_error(res.H, H_gt, img1.w, img1.h) << "\n";
  }

  const ImageF warped = warp_perspective(img1, res.H, img2.h, img2.w);
  write_pgm("registered_warp.pgm", warped);

  std::cout << "Wrote: registered_warp.pgm\n";
  if (argc < 3) std::cout << "Also wrote: demo_base.pgm, demo_target.pgm\n";
  return 0;
}
