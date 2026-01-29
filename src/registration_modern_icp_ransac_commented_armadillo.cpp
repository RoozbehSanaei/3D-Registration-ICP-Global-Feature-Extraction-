// registration_modern_icp_ransac_commented_armadillo.cpp
//
// Modern C++20 3D rigid registration (SE(3)) using Armadillo.
//
// Implements:
//   (2) ICP family
//       - point-to-point ICP
//       - trimmed ICP (drops the worst correspondences each iteration)
//   (3) Robust global registration
//       - RANSAC over provided correspondences (e.g., feature matches)
//
#include <armadillo>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using Vec3 = arma::Col<double>::fixed<3>;
using Mat3 = arma::Mat<double>::fixed<3,3>;

// -----------------------------
// SE(3): rotation + translation
// -----------------------------
// Kept as an aggregate type (no user-defined constructors), which keeps it easy to
// initialize and pass around. For identity transforms, use identitySE3().
struct TransformSE3 final {
  Mat3 R;
  Vec3 t;
};

[[nodiscard]] static inline TransformSE3 identitySE3() {
  TransformSE3 T{};
  T.R.eye();
  T.t.zeros();
  return T;
}

[[nodiscard]] static inline TransformSE3 makeSE3(const Mat3& R, const Vec3& t) {
  TransformSE3 T{};
  T.R = R;
  T.t = t;
  return T;
}

struct PointCloud final {
  std::vector<Vec3> pts;

  [[nodiscard]] std::size_t size() const noexcept { return pts.size(); }
  [[nodiscard]] bool empty() const noexcept { return pts.empty(); }

  [[nodiscard]] const Vec3& operator[](std::size_t i) const noexcept { return pts[i]; }
  [[nodiscard]] Vec3& operator[](std::size_t i) noexcept { return pts[i]; }
};

[[nodiscard]] static inline double sqr(double x) noexcept { return x * x; }

// -----------------------------
// Transform utilities
// -----------------------------

[[nodiscard]] static inline Vec3 applyPoint(const TransformSE3& T, const Vec3& p) noexcept {
  return T.R * p + T.t;
}

[[nodiscard]] static inline PointCloud applyCloud(const TransformSE3& T, const PointCloud& pc) {
  PointCloud out;
  out.pts.reserve(pc.size());
  for (const auto& p : pc.pts) out.pts.push_back(applyPoint(T, p));
  return out;
}

// Compose transforms: C = A ∘ B (apply B first, then A)
[[nodiscard]] static inline TransformSE3 compose(const TransformSE3& A, const TransformSE3& B) noexcept {
  TransformSE3 C{};
  C.R = A.R * B.R;
  C.t = A.R * B.t + A.t;
  return C;
}

[[nodiscard]] static inline TransformSE3 inverse(const TransformSE3& T) noexcept {
  TransformSE3 inv{};
  inv.R = T.R.t();
  inv.t = -(inv.R * T.t);
  return inv;
}

[[nodiscard]] static inline double rotationAngleRad(const Mat3& R) noexcept {
  const double tr = arma::trace(R);
  const double c = std::clamp((tr - 1.0) / 2.0, -1.0, 1.0);
  return std::acos(c);
}

[[nodiscard]] static inline std::pair<double,double>
poseError(const TransformSE3& T_est, const TransformSE3& T_gt) noexcept {
  const TransformSE3 Terr = compose(inverse(T_est), T_gt);
  return {rotationAngleRad(Terr.R), arma::norm(Terr.t, 2)};
}

// -----------------------------
// Kabsch rigid alignment (SVD)
// -----------------------------
// Given A[i] <-> B[i], solve least-squares R,t minimizing sum ||R*A + t - B||^2.
//
// Uses std::span to accept any contiguous container without copies.
[[nodiscard]] static TransformSE3 kabschSE3(std::span<const Vec3> A, std::span<const Vec3> B) {
  if (A.size() != B.size() || A.size() < 3) {
    throw std::runtime_error("kabschSE3: need matching A,B with size >= 3");
  }

  Vec3 cA{}; cA.zeros();
  Vec3 cB{}; cB.zeros();

  for (std::size_t i = 0; i < A.size(); ++i) {
    cA += A[i];
    cB += B[i];
  }
  cA /= static_cast<double>(A.size());
  cB /= static_cast<double>(B.size());

  Mat3 H{}; H.zeros();
  for (std::size_t i = 0; i < A.size(); ++i) {
    const Vec3 a = A[i] - cA;
    const Vec3 b = B[i] - cB;
    H += a * b.t();
  }

  arma::mat U, V;
  arma::vec s;
  arma::svd(U, s, V, arma::mat(H));

  Mat3 R{}; R = V.cols(0,2) * U.cols(0,2).t();

  // Reflection fix: enforce det(R)=+1
  if (arma::det(R) < 0.0) {
    arma::mat V2 = V;
    V2.col(2) *= -1.0;
    R = V2.cols(0,2) * U.cols(0,2).t();
  }

  const Vec3 t = cB - R * cA;
  return makeSE3(R, t);
}

// -----------------------------
// Nearest neighbor (brute force)
// -----------------------------
// This is O(N*M). Keep it for clarity; swap with KD-tree for speed.

struct NNMatch final {
  std::vector<std::size_t> idx;
  std::vector<double> dist;
};

[[nodiscard]] static NNMatch nearestNeighborBruteforce(const PointCloud& src, const PointCloud& dst) {
  if (dst.empty()) throw std::runtime_error("NN: target cloud is empty");

  NNMatch out;
  out.idx.resize(src.size());
  out.dist.resize(src.size());

  for (std::size_t i = 0; i < src.size(); ++i) {
    double best_d2 = std::numeric_limits<double>::infinity();
    std::size_t best_j = 0;

    for (std::size_t j = 0; j < dst.size(); ++j) {
      const Vec3 d = src[i] - dst[j];
      const double d2 = arma::dot(d, d);
      if (d2 < best_d2) {
        best_d2 = d2;
        best_j = j;
      }
    }
    out.idx[i] = best_j;
    out.dist[i] = std::sqrt(best_d2);
  }
  return out;
}

// Baseline correspondences via NN (keeps max_pairs smallest distances).
[[nodiscard]] static std::pair<std::vector<Vec3>, std::vector<Vec3>>
correspondencesViaNN(const PointCloud& src, const PointCloud& dst, std::size_t max_pairs) {
  const NNMatch nn = nearestNeighborBruteforce(src, dst);

  std::vector<std::size_t> order(src.size());
  std::iota(order.begin(), order.end(), 0);

  const auto keep = std::min(max_pairs, order.size());
  std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                    [&](std::size_t a, std::size_t b){ return nn.dist[a] < nn.dist[b]; });

  std::vector<Vec3> A; A.reserve(keep);
  std::vector<Vec3> B; B.reserve(keep);

  for (std::size_t k = 0; k < keep; ++k) {
    const auto i = order[k];
    A.push_back(src[i]);
    B.push_back(dst[nn.idx[i]]);
  }
  return {A, B};
}

// -----------------------------
// ICP point-to-point (+ trimming)
// -----------------------------

struct ICPParams final {
  int max_iter = 80;
  double tol = 1e-7;
  double trim_ratio = 0.3; // 0..0.95
};

struct ICPResult final {
  TransformSE3 T{};
  double rmse = std::numeric_limits<double>::infinity();
  int iters = 0;
  bool converged = false;
};

[[nodiscard]] static std::vector<std::size_t>
trimSelection(const std::vector<double>& dists, double trim_ratio) {
  if (trim_ratio <= 0.0) {
    std::vector<std::size_t> sel(dists.size());
    std::iota(sel.begin(), sel.end(), 0);
    return sel;
  }
  if (!(trim_ratio >= 0.0 && trim_ratio < 0.95)) {
    throw std::runtime_error("trim_ratio must be in [0,0.95)");
  }

  std::vector<std::size_t> order(dists.size());
  std::iota(order.begin(), order.end(), 0);

  const std::size_t keep = static_cast<std::size_t>(
      std::ceil((1.0 - trim_ratio) * static_cast<double>(dists.size())));

  std::nth_element(order.begin(), order.begin() + keep, order.end(),
                   [&](std::size_t a, std::size_t b){ return dists[a] < dists[b]; });
  order.resize(keep);

  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b){ return dists[a] < dists[b]; });
  return order;
}

[[nodiscard]] static ICPResult icpPointToPoint(const PointCloud& source,
                                              const PointCloud& target,
                                              const std::optional<TransformSE3>& init_T,
                                              const ICPParams& params) {
  if (source.empty()) throw std::runtime_error("ICP: source cloud is empty");
  if (target.empty()) throw std::runtime_error("ICP: target cloud is empty");

  TransformSE3 T = init_T.value_or(identitySE3());
  double prev = std::numeric_limits<double>::infinity();

  for (int it = 1; it <= params.max_iter; ++it) {
    const PointCloud src_t = applyCloud(T, source);
    const NNMatch nn = nearestNeighborBruteforce(src_t, target);
    const auto sel = trimSelection(nn.dist, params.trim_ratio);

    std::vector<Vec3> A; A.reserve(sel.size());
    std::vector<Vec3> B; B.reserve(sel.size());
    for (auto i : sel) {
      A.push_back(src_t[i]);
      B.push_back(target[nn.idx[i]]);
    }

    const TransformSE3 T_delta = kabschSE3(A, B);
    T = compose(T_delta, T);

    double sum2 = 0.0;
    for (auto i : sel) sum2 += sqr(nn.dist[i]);
    const double rmse = std::sqrt(sum2 / static_cast<double>(sel.size()));

    if (std::abs(prev - rmse) < params.tol) {
      return ICPResult{.T = T, .rmse = rmse, .iters = it, .converged = true};
    }
    prev = rmse;
  }

  return ICPResult{.T = T, .rmse = prev, .iters = params.max_iter, .converged = false};
}

// -----------------------------
// RANSAC over correspondences
// -----------------------------

struct RANSACParams final {
  double threshold = 0.06;
  int max_iter = 2500;
  std::uint64_t seed = 0;
};

struct RANSACResult final {
  TransformSE3 T{};
  std::vector<bool> inliers{};
  int num_inliers = 0;
  double rmse_inliers = std::numeric_limits<double>::infinity();
};

[[nodiscard]] static bool nondegenerateTriple(const Vec3& a, const Vec3& b, const Vec3& c) noexcept {
  const Vec3 cr = arma::cross(b - a, c - a);
  return arma::norm(cr, 2) > 1e-8;
}

[[nodiscard]] static RANSACResult ransacSE3(std::span<const Vec3> src_corr,
                                           std::span<const Vec3> dst_corr,
                                           const RANSACParams& params) {
  if (src_corr.size() != dst_corr.size() || src_corr.size() < 3) {
    throw std::runtime_error("ransacSE3: need matching correspondences with size >= 3");
  }

  std::mt19937_64 rng(params.seed);
  std::uniform_int_distribution<std::size_t> uni(0, src_corr.size() - 1);

  TransformSE3 best_T = identitySE3();
  std::vector<bool> best_inl(src_corr.size(), false);
  int best_cnt = 0;
  double best_rmse = std::numeric_limits<double>::infinity();

  for (int it = 0; it < params.max_iter; ++it) {
    std::size_t i0 = uni(rng), i1 = uni(rng), i2 = uni(rng);
    while (i1 == i0) i1 = uni(rng);
    while (i2 == i0 || i2 == i1) i2 = uni(rng);

    if (!nondegenerateTriple(src_corr[i0], src_corr[i1], src_corr[i2])) continue;

    std::array<Vec3,3> A{src_corr[i0], src_corr[i1], src_corr[i2]};
    std::array<Vec3,3> B{dst_corr[i0], dst_corr[i1], dst_corr[i2]};

    const TransformSE3 T = kabschSE3(A, B);

    std::vector<bool> inl(src_corr.size(), false);
    int cnt = 0;

    for (std::size_t k = 0; k < src_corr.size(); ++k) {
      const double d = arma::norm(applyPoint(T, src_corr[k]) - dst_corr[k], 2);
      if (d < params.threshold) {
        inl[k] = true;
        ++cnt;
      }
    }

    if (cnt <= best_cnt) continue;

    // Standard refinement: re-fit on all inliers.
    std::vector<Vec3> Ain; Ain.reserve(static_cast<std::size_t>(cnt));
    std::vector<Vec3> Bin; Bin.reserve(static_cast<std::size_t>(cnt));
    for (std::size_t k = 0; k < src_corr.size(); ++k) {
      if (inl[k]) { Ain.push_back(src_corr[k]); Bin.push_back(dst_corr[k]); }
    }

    TransformSE3 T_ref = T;
    if (Ain.size() >= 3) T_ref = kabschSE3(Ain, Bin);

    double sum2 = 0.0;
    int cnt2 = 0;
    for (std::size_t k = 0; k < src_corr.size(); ++k) {
      if (!inl[k]) continue;
      const double d = arma::norm(applyPoint(T_ref, src_corr[k]) - dst_corr[k], 2);
      sum2 += d * d;
      ++cnt2;
    }

    const double rmse = (cnt2 > 0) ? std::sqrt(sum2 / static_cast<double>(cnt2))
                                   : std::numeric_limits<double>::infinity();

    best_T = T_ref;
    best_inl = std::move(inl);
    best_cnt = cnt;
    best_rmse = rmse;
  }

  return RANSACResult{.T = best_T, .inliers = best_inl, .num_inliers = best_cnt, .rmse_inliers = best_rmse};
}

// -----------------------------
// Synthetic tests
// -----------------------------

struct SynthConfig final {
  std::string name;

  int N = 1200;
  double noise = 0.01;
  double outlier_ratio = 0.2;
  double overlap = 1.0;

  double ransac_thresh = 0.06;
  int ransac_iter = 2500;
  std::size_t max_pairs = 3000;

  int icp_iter = 80;
  double trim_ratio = 0.3;

  double init_rot_sigma = 0.06;
  double init_trans_sigma = 0.06;

  int trueK = 600;
  double wrong_match_ratio = 0.7;
  double true_thresh = 0.03;
  int true_iter = 3000;
};

[[nodiscard]] static Mat3 randomRotation(std::mt19937_64& rng) {
  std::normal_distribution<double> nd(0.0, 1.0);

  arma::mat M(3,3);
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) M(r,c) = nd(rng);

  arma::mat Q, R;
  arma::qr(Q, R, M);

  Mat3 QQ{}; QQ = Q;
  if (arma::det(QQ) < 0.0) QQ.col(0) *= -1.0;
  return QQ;
}

[[nodiscard]] static Vec3 uniformVec3(std::mt19937_64& rng, double lo, double hi) {
  std::uniform_real_distribution<double> ud(lo, hi);
  Vec3 v{};
  v(0)=ud(rng); v(1)=ud(rng); v(2)=ud(rng);
  return v;
}

[[nodiscard]] static PointCloud randomCloud(std::mt19937_64& rng, int N, double lo=-1.0, double hi=1.0) {
  PointCloud pc;
  pc.pts.reserve(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) pc.pts.push_back(uniformVec3(rng, lo, hi));
  return pc;
}

[[nodiscard]] static PointCloud selectSubset(std::mt19937_64& rng, const PointCloud& pc, int M) {
  std::vector<int> idx(static_cast<std::size_t>(pc.size()));
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), rng);

  PointCloud out;
  out.pts.reserve(static_cast<std::size_t>(M));
  for (int i = 0; i < M; ++i) out.pts.push_back(pc.pts[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])]);
  return out;
}

struct SynthPair final {
  PointCloud src_obs;
  PointCloud tgt;
  TransformSE3 T_gt;
  PointCloud src_inlier;
};

[[nodiscard]] static SynthPair synthPair(const SynthConfig& cfg, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> noiseN(0.0, cfg.noise);

  PointCloud tgt_full = randomCloud(rng, cfg.N, -1.0, 1.0);
  PointCloud tgt = tgt_full;

  if (cfg.overlap < 1.0) {
    const int M = std::max(3, static_cast<int>(std::floor(cfg.overlap * cfg.N)));
    tgt = selectSubset(rng, tgt_full, M);
  }

  const TransformSE3 T_gt = makeSE3(randomRotation(rng), uniformVec3(rng, -0.5, 0.5));
  const TransformSE3 T_inv = inverse(T_gt);

  const PointCloud src_clean = applyCloud(T_inv, tgt);

  PointCloud src_inlier;
  src_inlier.pts.reserve(src_clean.size());
  for (const auto& p : src_clean.pts) {
    Vec3 n{}; n(0)=noiseN(rng); n(1)=noiseN(rng); n(2)=noiseN(rng);
    src_inlier.pts.push_back(p + n);
  }

  PointCloud src_obs = src_inlier;

  if (cfg.outlier_ratio > 0.0) {
    const int outN = static_cast<int>(std::floor(cfg.outlier_ratio * static_cast<double>(src_inlier.size())));
    for (int i = 0; i < outN; ++i) src_obs.pts.push_back(uniformVec3(rng, -3.0, 3.0));
  }

  return SynthPair{.src_obs = std::move(src_obs), .tgt = std::move(tgt), .T_gt = T_gt, .src_inlier = std::move(src_inlier)};
}

// Rodrigues (SO(3) exponential map) for small perturbations.
[[nodiscard]] static Mat3 rodrigues(const Vec3& w) noexcept {
  const double th = arma::norm(w, 2);
  Mat3 I{}; I.eye();
  if (th < 1e-12) return I;

  const Vec3 k = w / th;

  Mat3 K{}; K.zeros();
  K(0,1) = -k(2); K(0,2) =  k(1);
  K(1,0) =  k(2); K(1,2) = -k(0);
  K(2,0) = -k(1); K(2,1) =  k(0);

  return I + std::sin(th) * K + (1.0 - std::cos(th)) * (K * K);
}

[[nodiscard]] static TransformSE3 perturbPose(std::mt19937_64& rng,
                                             const TransformSE3& T_gt,
                                             double rot_sigma,
                                             double trans_sigma) {
  std::normal_distribution<double> ndR(0.0, rot_sigma);
  std::normal_distribution<double> ndT(0.0, trans_sigma);

  Vec3 w{}; w(0)=ndR(rng); w(1)=ndR(rng); w(2)=ndR(rng);
  const Mat3 dR = rodrigues(w);

  Vec3 dt{}; dt(0)=ndT(rng); dt(1)=ndT(rng); dt(2)=ndT(rng);

  return compose(makeSE3(dR, dt), T_gt);
}

// Simulate feature matches: many correct correspondences plus wrong matches.
[[nodiscard]] static std::pair<std::vector<Vec3>, std::vector<Vec3>>
trueCorrespondencesWithWrong(std::mt19937_64& rng,
                            const PointCloud& src_inlier,
                            const PointCloud& tgt,
                            int K,
                            double wrong_ratio) {
  const int N = static_cast<int>(tgt.size());
  const int kk = std::min(K, N);

  std::vector<int> idx(static_cast<std::size_t>(N));
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), rng);
  idx.resize(static_cast<std::size_t>(kk));

  std::vector<Vec3> A; A.reserve(static_cast<std::size_t>(kk));
  std::vector<Vec3> B; B.reserve(static_cast<std::size_t>(kk));

  for (int i = 0; i < kk; ++i) {
    const int j = idx[static_cast<std::size_t>(i)];
    A.push_back(src_inlier.pts[static_cast<std::size_t>(j)]);
    B.push_back(tgt.pts[static_cast<std::size_t>(j)]);
  }

  const int wrong = static_cast<int>(std::floor(wrong_ratio * static_cast<double>(kk)));
  std::vector<int> bad(static_cast<std::size_t>(kk));
  std::iota(bad.begin(), bad.end(), 0);
  std::shuffle(bad.begin(), bad.end(), rng);
  bad.resize(static_cast<std::size_t>(wrong));

  for (const int bi : bad) {
    B[static_cast<std::size_t>(bi)] = uniformVec3(rng, -1.0, 1.0);
  }

  return {A, B};
}

// -----------------------------
// Trials and summary
// -----------------------------

struct TrialMetrics final {
  std::string config;
  int trial = 0;

  double A0_rot = 0.0;
  double A0_trans = 0.0;

  double B_rot = 0.0;
  double B_trans = 0.0;

  double C_rot = 0.0;
  double C_trans = 0.0;

  int C_inliers = 0;

  bool A0_success = false;
  bool B_success = false;
  bool C_success = false;
};

[[nodiscard]] static TrialMetrics runOne(const SynthConfig& cfg, int trial, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  const SynthPair sp = synthPair(cfg, seed);

  // A0: NN -> RANSAC -> ICP
  auto [A0_src, A0_dst] = correspondencesViaNN(sp.src_obs, sp.tgt, cfg.max_pairs);
  const RANSACResult r0 = ransacSE3(A0_src, A0_dst,
                                   RANSACParams{.threshold = cfg.ransac_thresh, .max_iter = cfg.ransac_iter, .seed = seed});
  const ICPResult icpA0 = icpPointToPoint(sp.src_obs, sp.tgt, std::optional<TransformSE3>{r0.T},
                                         ICPParams{.max_iter = cfg.icp_iter, .tol = 1e-7, .trim_ratio = cfg.trim_ratio});
  const auto [a0_rot, a0_trans] = poseError(icpA0.T, sp.T_gt);

  // B: ICP with a good init
  const TransformSE3 T_init = perturbPose(rng, sp.T_gt, cfg.init_rot_sigma, cfg.init_trans_sigma);
  const ICPResult icpB = icpPointToPoint(sp.src_obs, sp.tgt, std::optional<TransformSE3>{T_init},
                                        ICPParams{.max_iter = cfg.icp_iter, .tol = 1e-7, .trim_ratio = cfg.trim_ratio});
  const auto [b_rot, b_trans] = poseError(icpB.T, sp.T_gt);

  // C: RANSAC on meaningful correspondences -> ICP
  auto [C_src, C_dst] = trueCorrespondencesWithWrong(rng, sp.src_inlier, sp.tgt, cfg.trueK, cfg.wrong_match_ratio);
  const RANSACResult rC = ransacSE3(C_src, C_dst,
                                   RANSACParams{.threshold = cfg.true_thresh, .max_iter = cfg.true_iter, .seed = seed + 999});
  const ICPResult icpC = icpPointToPoint(sp.src_obs, sp.tgt, std::optional<TransformSE3>{rC.T},
                                        ICPParams{.max_iter = cfg.icp_iter, .tol = 1e-7, .trim_ratio = cfg.trim_ratio});
  const auto [c_rot, c_trans] = poseError(icpC.T, sp.T_gt);

  constexpr double rot_ok = 0.05;   // ~2.9 deg
  constexpr double trans_ok = 0.05; // in the synthetic scene scale

  TrialMetrics m;
  m.config = cfg.name;
  m.trial = trial;

  m.A0_rot = a0_rot; m.A0_trans = a0_trans;
  m.B_rot  = b_rot;  m.B_trans  = b_trans;
  m.C_rot  = c_rot;  m.C_trans  = c_trans;

  m.C_inliers = rC.num_inliers;

  m.A0_success = (a0_rot < rot_ok) && (a0_trans < trans_ok);
  m.B_success  = (b_rot < rot_ok)  && (b_trans < trans_ok);
  m.C_success  = (c_rot < rot_ok)  && (c_trans < trans_ok);

  return m;
}

[[nodiscard]] static double median(std::vector<double> v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const std::size_t n = v.size();
  const std::size_t mid = n / 2;

  std::nth_element(v.begin(), v.begin() + mid, v.end());
  double m = v[mid];

  if (n % 2 == 0) {
    std::nth_element(v.begin(), v.begin() + (mid - 1), v.end());
    m = 0.5 * (m + v[mid - 1]);
  }
  return m;
}

struct SummaryRow final {
  std::string config;
  int trials = 0;

  double A0_success_rate = 0.0;
  double B_success_rate  = 0.0;
  double C_success_rate  = 0.0;

  double A0_med_rot = 0.0;
  double B_med_rot  = 0.0;
  double C_med_rot  = 0.0;

  double C_med_inliers = 0.0;
};

[[nodiscard]] static std::vector<SummaryRow> summarize(const std::vector<TrialMetrics>& rows) {
  std::vector<std::string> configs;
  configs.reserve(rows.size());
  for (const auto& r : rows) configs.push_back(r.config);
  std::sort(configs.begin(), configs.end());
  configs.erase(std::unique(configs.begin(), configs.end()), configs.end());

  std::vector<SummaryRow> out;
  out.reserve(configs.size());

  for (const auto& name : configs) {
    std::vector<const TrialMetrics*> g;
    for (const auto& r : rows) if (r.config == name) g.push_back(&r);

    SummaryRow s;
    s.config = name;
    s.trials = static_cast<int>(g.size());

    int a0 = 0, b = 0, c = 0;
    std::vector<double> a0rot, brot, crot, cinl;
    a0rot.reserve(g.size()); brot.reserve(g.size()); crot.reserve(g.size()); cinl.reserve(g.size());

    for (auto* pr : g) {
      a0 += pr->A0_success ? 1 : 0;
      b  += pr->B_success  ? 1 : 0;
      c  += pr->C_success  ? 1 : 0;

      a0rot.push_back(pr->A0_rot);
      brot.push_back(pr->B_rot);
      crot.push_back(pr->C_rot);
      cinl.push_back(static_cast<double>(pr->C_inliers));
    }

    s.A0_success_rate = static_cast<double>(a0) / static_cast<double>(s.trials);
    s.B_success_rate  = static_cast<double>(b)  / static_cast<double>(s.trials);
    s.C_success_rate  = static_cast<double>(c)  / static_cast<double>(s.trials);

    s.A0_med_rot = median(a0rot);
    s.B_med_rot  = median(brot);
    s.C_med_rot  = median(crot);
    s.C_med_inliers = median(cinl);

    out.push_back(s);
  }

  return out;
}

static void printSummaryTable(const std::vector<SummaryRow>& s) {
  std::cout << "\n| Config | Trials | A0 success | B success | C success | A0 med rot | B med rot | C med rot | C med inliers |\n";
  std::cout << "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n";
  std::cout << std::fixed << std::setprecision(6);

  for (const auto& r : s) {
    std::cout << "| " << r.config
              << " | " << r.trials
              << " | " << r.A0_success_rate
              << " | " << r.B_success_rate
              << " | " << r.C_success_rate
              << " | " << r.A0_med_rot
              << " | " << r.B_med_rot
              << " | " << r.C_med_rot
              << " | " << r.C_med_inliers
              << " |\n";
  }
  std::cout << "\n";
}

int main(int argc, char** argv) {
  int trials = 3;
  std::uint64_t seed0 = 21;

  if (argc >= 2) {
    try { trials = std::stoi(argv[1]); } catch (...) {}
  }
  if (argc >= 3) {
    try { seed0 = static_cast<std::uint64_t>(std::stoull(argv[2])); } catch (...) {}
  }

  const std::vector<SynthConfig> configs = {
    SynthConfig{
      .name = "moderate: noise 0.01, outliers 20%, overlap 100%",
      .N = 1200, .noise = 0.01, .outlier_ratio = 0.2, .overlap = 1.0,
      .ransac_thresh = 0.06, .ransac_iter = 2000, .max_pairs = 2500,
      .icp_iter = 60, .trim_ratio = 0.3,
      .init_rot_sigma = 0.05, .init_trans_sigma = 0.05,
      .trueK = 600, .wrong_match_ratio = 0.7, .true_thresh = 0.03, .true_iter = 3000
    },
    SynthConfig{
      .name = "partial: noise 0.01, outliers 20%, overlap 60%",
      .N = 1600, .noise = 0.01, .outlier_ratio = 0.2, .overlap = 0.6,
      .ransac_thresh = 0.07, .ransac_iter = 2500, .max_pairs = 3000,
      .icp_iter = 90, .trim_ratio = 0.4,
      .init_rot_sigma = 0.05, .init_trans_sigma = 0.05,
      .trueK = 700, .wrong_match_ratio = 0.7, .true_thresh = 0.035, .true_iter = 4000
    },
    SynthConfig{
      .name = "hard: noise 0.02, outliers 40%, overlap 100%",
      .N = 1200, .noise = 0.02, .outlier_ratio = 0.4, .overlap = 1.0,
      .ransac_thresh = 0.08, .ransac_iter = 2500, .max_pairs = 3000,
      .icp_iter = 80, .trim_ratio = 0.4,
      .init_rot_sigma = 0.06, .init_trans_sigma = 0.06,
      .trueK = 600, .wrong_match_ratio = 0.75, .true_thresh = 0.045, .true_iter = 4500
    },
  };

  std::vector<TrialMetrics> rows;
  rows.reserve(static_cast<std::size_t>(trials) * configs.size());

  const auto t0 = std::chrono::steady_clock::now();

  for (std::size_t ci = 0; ci < configs.size(); ++ci) {
    for (int tr = 0; tr < trials; ++tr) {
      const std::uint64_t seed = seed0 + static_cast<std::uint64_t>(1000 * tr) + static_cast<std::uint64_t>(100 * ci);
      rows.push_back(runOne(configs[ci], tr, seed));
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  printSummaryTable(summarize(rows));
  std::cerr << "Elapsed: " << ms << " ms\n";
  return 0;
}
