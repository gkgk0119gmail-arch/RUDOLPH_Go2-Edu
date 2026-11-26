// HiPAN low-level student (MLP over a 50-step proprioceptive history) for the real Go2.
// Mirrors deploy/hipan_low_policy.py (the numpy reference) exactly; weights from deploy/export_cpp_weights.py.
#pragma once

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipan {

using Vec = Eigen::VectorXf;
using Mat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// policy (Isaac) joint i: hips FL FR RL RR, thighs, calves  ->  Unitree SDK motor index (FR FL RR RL legs x hip thigh calf)
constexpr std::array<int, 12> kIsaacToSdk = {3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8};
// foot_force order FR FL RR RL -> policy FL FR RL RR
constexpr std::array<int, 4> kFootSdkToPolicy = {1, 0, 3, 2};

// Go2 URDF joint limits (policy order), rad
constexpr std::array<float, 12> kQMin = {-1.0472f, -1.0472f, -1.0472f, -1.0472f, -1.5708f, -1.5708f, -0.5236f,
                                         -0.5236f, -2.7227f, -2.7227f, -2.7227f, -2.7227f};
constexpr std::array<float, 12> kQMax = {1.0472f, 1.0472f, 1.0472f, 1.0472f, 3.4907f, 3.4907f, 4.5379f,
                                         4.5379f, -0.83776f, -0.83776f, -0.83776f, -0.83776f};

inline Vec elu(const Vec& x) {
  Vec y(x.size());
  for (int i = 0; i < x.size(); ++i) y[i] = x[i] > 0.f ? x[i] : std::expm1(x[i]);
  return y;
}

// foot positions in the body frame from joint angles (policy order), 12 = (x, y, z) x FL FR RL RR
inline std::array<float, 12> feet_pos_body(const float* q) {
  constexpr float HIP_X = 0.1934f, HIP_Y = 0.0465f, THIGH_Y = 0.0955f, L1 = 0.213f, L2 = 0.213f;
  constexpr int SX[4] = {1, 1, -1, -1}, SY[4] = {1, -1, 1, -1};
  std::array<float, 12> out{};
  for (int k = 0; k < 4; ++k) {
    const float qh = q[k], qt = q[4 + k], qc = q[8 + k];
    const float x = -L1 * std::sin(qt) - L2 * std::sin(qt + qc);
    const float z = -L1 * std::cos(qt) - L2 * std::cos(qt + qc);
    const float y = SY[k] * THIGH_Y;
    const float c = std::cos(qh), s = std::sin(qh);
    out[3 * k + 0] = SX[k] * HIP_X + x;
    out[3 * k + 1] = SY[k] * HIP_Y + c * y - s * z;
    out[3 * k + 2] = s * y + c * z;
  }
  return out;
}

// IMU quaternion (w, x, y, z) -> roll, pitch (same as Isaac Lab euler_xyz_from_quat)
inline void roll_pitch(float w, float x, float y, float z, float& roll, float& pitch) {
  roll = std::atan2(2.f * (w * x + y * z), 1.f - 2.f * (x * x + y * y));
  float sp = 2.f * (w * y - z * x);
  sp = std::max(-1.f, std::min(1.f, sp));
  pitch = std::asin(sp);
}

class Policy {
 public:
  int H = 0, D = 0, EST = 0;
  std::array<float, 12> default_q{}, offset{}, scale{}, kp{}, kd{};

  explicit Policy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "HPL1", 4) != 0) throw std::runtime_error("bad weights magic");
    int32_t hdr[3];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    H = hdr[0], D = hdr[1], EST = hdr[2];
    for (auto* a : {&default_q, &offset, &scale, &kp, &kd}) f.read(reinterpret_cast<char*>(a->data()), 12 * 4);
    int32_t n;
    f.read(reinterpret_cast<char*>(&n), 4);
    for (int i = 0; i < n; ++i) {
      int32_t rc[2];
      f.read(reinterpret_cast<char*>(rc), 8);
      Mat W(rc[0], rc[1]);
      Vec b(rc[0]);
      f.read(reinterpret_cast<char*>(W.data()), sizeof(float) * rc[0] * rc[1]);
      f.read(reinterpret_cast<char*>(b.data()), sizeof(float) * rc[0]);
      W_.push_back(W);
      b_.push_back(b);
    }
    if (!f || n != 8) throw std::runtime_error("truncated weights");
    hist_ = Mat::Zero(H, D);
  }

  void reset() { filled_ = false; }

  // obs [D] -> action [12] (raw policy output), est [EST]; keeps the history
  void step(const Vec& obs, Vec& action, Vec& est) {
    if (!filled_) {
      for (int t = 0; t < H; ++t) hist_.row(t) = obs.transpose();
      filled_ = true;
    } else {
      hist_.topRows(H - 1) = hist_.bottomRows(H - 1).eval();
      hist_.row(H - 1) = obs.transpose();
    }
    forward(hist_, action, est);
  }

  // hist [H, D] row-major, newest last
  void forward(const Mat& hist, Vec& action, Vec& est) const {
    Eigen::Map<const Vec> x(hist.data(), H * D);
    Vec e = elu(lin(0, x));
    e = elu(lin(1, e));
    Vec z = lin(2, e);
    est = lin(3, z);
    Vec p(z.size() + est.size() + D);
    p << z, est, hist.row(H - 1).transpose();
    p = elu(lin(4, p));
    p = elu(lin(5, p));
    p = elu(lin(6, p));
    action = lin(7, p);
  }

 private:
  Vec lin(int i, const Vec& x) const { return W_[i] * x + b_[i]; }
  std::vector<Mat> W_;
  std::vector<Vec> b_;
  Mat hist_;
  bool filled_ = false;
};

}  // namespace hipan
