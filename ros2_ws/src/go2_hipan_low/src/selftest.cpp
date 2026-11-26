// C++ policy vs the sim replay stored by export_hipan_low.py (torch actions), plus forward timing.
//   ros2 run go2_hipan_low hipan_low_selftest <weights.bin> <weights_test.bin>
#include <chrono>
#include <cstdio>

#include "go2_hipan_low/hipan_policy.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::printf("usage: %s weights.bin weights_test.bin\n", argv[0]);
    return 1;
  }
  hipan::Policy pol(argv[1]);
  std::ifstream f(argv[2], std::ios::binary);
  char magic[4];
  f.read(magic, 4);
  int32_t hdr[4];
  f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  const int T = hdr[0], D = hdr[1], A = hdr[2], E = hdr[3];
  std::vector<float> obs(T * D), act(T * A), est(T * E);
  f.read(reinterpret_cast<char*>(obs.data()), obs.size() * 4);
  f.read(reinterpret_cast<char*>(act.data()), act.size() * 4);
  f.read(reinterpret_cast<char*>(est.data()), est.size() * 4);

  float max_a = 0, max_e = 0, max_fk = 0;
  hipan::Vec a, e, o(D);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < D; ++i) o[i] = obs[t * D + i];
    pol.step(o, a, e);
    for (int i = 0; i < A; ++i) max_a = std::max(max_a, std::abs(a[i] - act[t * A + i]));
    for (int i = 0; i < E; ++i) max_e = std::max(max_e, std::abs(e[i] - est[t * E + i]));
    float q[12];
    for (int i = 0; i < 12; ++i) q[i] = o[i] + pol.default_q[i];
    auto fk = hipan::feet_pos_body(q);
    for (int i = 0; i < 12; ++i) max_fk = std::max(max_fk, std::abs(fk[i] - o[24 + i]));
  }
  std::printf("C++ vs torch: action max err %.2e, estimate max err %.2e, FK max err %.4f m (obs noise +-0.01)\n",
              max_a, max_e, max_fk);
  auto t0 = std::chrono::steady_clock::now();
  for (int k = 0; k < 500; ++k) pol.step(o, a, e);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 500;
  std::printf("step %.3f ms\n", ms);
  return (max_a < 1e-4f && max_e < 1e-4f) ? 0 : 2;
}
