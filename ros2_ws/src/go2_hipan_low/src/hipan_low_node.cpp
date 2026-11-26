// HiPAN low-level controller for the real Go2: /lowstate -> student MLP (50 Hz) -> /lowcmd.
//
// mode:=shadow (default)  computes everything, logs, publishes NOTHING (robot stays under Unitree's controller)
// mode:=live              publishes /lowcmd. Unitree's sport mode must be released first (tools/mode_switch.py),
//                         robot hung up for the first runs.
//
// live states (like Unitree's own remote flow):
//   start: WAIT -> RAMP (lying -> sim default pose, ramp_s, stand_kp/kd) -> HOLD (standing)
//   HOLD  --[start button]--> POLICY (walking policy; gains blend stand -> trained over gain_blend_s)
//   POLICY --[L2+A]--> RAMP (back to the default pose, 1 s) -> HOLD
//   HOLD  --[L2+A]--> RAMP (sit down slowly, sit_s) -> DOWN (lying, soft hold)
//   DOWN  --[L2+A]--> RAMP (stand up, ramp_s) -> HOLD
//   EMERGENCY only: [L2+B] or any safety trip -> DAMP (motors damped, robot sinks; latched until restart)
// safety trips: /lowstate older than state_timeout_ms, |roll| or |pitch| > max_tilt, joint outside URDF limits,
//               non-finite action
// command:      cmd_source:=remote (sticks: ly vx, lx vy, rx wz; up/down height +-2 cm, left/right roll +-0.05)
//               cmd_source:=topic  (/hipan/cmd std_msgs/Float32MultiArray [vx vy wz h roll], 0.5 s timeout -> stand)
#include <pthread.h>
#include <sched.h>
#include <time.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>

#include "go2_hipan_low/hipan_policy.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/empty.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "unitree_go/msg/low_cmd.hpp"
#include "unitree_go/msg/low_state.hpp"

// ---------------------------------------------------------------- LowCmd CRC (unitree_ros2 example/common/motor_crc)
namespace crc {
typedef struct {
  uint8_t off;
  std::array<uint8_t, 3> reserve;
} BmsCmd;
typedef struct {
  uint8_t mode;
  float q, dq, tau, Kp, Kd;
  std::array<uint32_t, 3> reserve;
} MotorCmd;
typedef struct {
  std::array<uint8_t, 2> head;
  uint8_t levelFlag;
  uint8_t frameReserve;
  std::array<uint32_t, 2> SN;
  std::array<uint32_t, 2> version;
  uint16_t bandWidth;
  std::array<MotorCmd, 20> motorCmd;
  BmsCmd bms;
  std::array<uint8_t, 40> wirelessRemote;
  std::array<uint8_t, 12> led;
  std::array<uint8_t, 2> fan;
  uint8_t gpio;
  uint32_t reserve;
  uint32_t crc;
} LowCmdRaw;

inline uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t poly = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    uint32_t xbit = 1u << 31, data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= poly;
      } else {
        CRC32 <<= 1;
      }
      if (data & xbit) CRC32 ^= poly;
      xbit >>= 1;
    }
  }
  return CRC32;
}

inline void set_crc(unitree_go::msg::LowCmd& msg) {
  LowCmdRaw raw{};
  std::memcpy(&raw.head[0], &msg.head[0], 2);
  raw.levelFlag = msg.level_flag;
  raw.frameReserve = msg.frame_reserve;
  std::memcpy(&raw.SN[0], &msg.sn[0], 8);
  std::memcpy(&raw.version[0], &msg.version[0], 8);
  raw.bandWidth = msg.bandwidth;
  for (int i = 0; i < 20; i++) {
    raw.motorCmd[i].mode = msg.motor_cmd[i].mode;
    raw.motorCmd[i].q = msg.motor_cmd[i].q;
    raw.motorCmd[i].dq = msg.motor_cmd[i].dq;
    raw.motorCmd[i].tau = msg.motor_cmd[i].tau;
    raw.motorCmd[i].Kp = msg.motor_cmd[i].kp;
    raw.motorCmd[i].Kd = msg.motor_cmd[i].kd;
    std::memcpy(&raw.motorCmd[i].reserve[0], &msg.motor_cmd[i].reserve[0], 12);
  }
  raw.bms.off = msg.bms_cmd.off;
  std::memcpy(&raw.bms.reserve[0], &msg.bms_cmd.reserve[0], 3);
  std::memcpy(&raw.wirelessRemote[0], &msg.wireless_remote[0], 40);
  std::memcpy(&raw.led[0], &msg.led[0], 12);
  std::memcpy(&raw.fan[0], &msg.fan[0], 2);
  raw.gpio = msg.gpio;
  raw.reserve = msg.reserve;
  msg.crc = crc32_core(reinterpret_cast<uint32_t*>(&raw), (sizeof(LowCmdRaw) >> 2) - 1);
}
}  // namespace crc

constexpr double kPosStop = 2.146E+9f, kVelStop = 16000.0f;

// ---------------------------------------------------------------- remote (wireless_remote bytes in LowState)
struct Remote {
  uint16_t keys = 0;
  float lx = 0, rx = 0, ry = 0, ly = 0;
  bool key(int bit) const { return (keys >> bit) & 1; }
  // bits: R1 0, L1 1, start 2, select 3, R2 4, L2 5, F1 6, F2 7, A 8, B 9, X 10, Y 11, up 12, right 13, down 14, left 15
  static Remote parse(const std::array<uint8_t, 40>& b) {
    Remote r;
    std::memcpy(&r.keys, &b[2], 2);
    std::memcpy(&r.lx, &b[4], 4);
    std::memcpy(&r.rx, &b[8], 4);
    std::memcpy(&r.ry, &b[12], 4);
    std::memcpy(&r.ly, &b[20], 4);
    return r;
  }
};

enum State : uint8_t { WAIT = 0, RAMP = 1, HOLD = 2, POLICY = 3, DAMP = 4, DOWN = 5 };
static const char* kStateName[] = {"WAIT", "RAMP", "HOLD", "POLICY", "DAMP", "DOWN"};
// lying pose (policy order: hips FL FR RL RR, thighs, calves), inside the calf limit; front legs ~1.7 cm taller so the
// chin-mounted L1 lidar clears the floor
// intermediate crouch for sitting (base ~0.18 m, feet under the hips): below this the chin lidar meets the floor,
// so the rest of the way down is slow
constexpr std::array<float, 12> kSitMidQ = {0.f, 0.f, 0.f, 0.f, 1.13f, 1.13f, 1.13f, 1.13f, -2.27f, -2.27f, -2.27f, -2.27f};
// hips: front legs tucked in (FL -, FR +), rear legs splayed out like Unitree's own lying pose (RL +0.4, RR -0.4):
// the rear sinks lower and the front stays up, so the chin lidar does not hit the floor
constexpr std::array<float, 12> kLyingQ = {-0.10f, 0.10f, 0.40f, -0.40f, 1.22f, 1.22f, 1.25f, 1.25f, -2.60f, -2.60f, -2.68f, -2.68f};

#pragma pack(push, 1)
struct LogRow {
  double t;
  uint8_t state;
  float loop_dt, state_age;
  float obs[62], action[12], target[12], q[12], dq[12], est[5], foot_raw[4], cmd[5], rpy[3], tau_est[12];
};
#pragma pack(pop)

class HipanLowNode : public rclcpp::Node {
 public:
  HipanLowNode() : Node("hipan_low") {
    weights_ = declare_parameter<std::string>("weights", "");
    live_ = declare_parameter<std::string>("mode", "shadow") == "live";
    cmd_topic_ = declare_parameter<std::string>("cmd_source", "remote") == "topic";
    contact_thr_ = declare_parameter<double>("contact_thr", 20.0);  // until calibrated
    contact_margin_ = declare_parameter<double>("contact_margin", 5.0);  // per-foot threshold = lying baseline + margin
    thr_.fill(contact_thr_);
    h_default_ = declare_parameter<double>("h_default", 0.30);
    ramp_s_ = declare_parameter<double>("ramp_s", 3.0);
    stand_kp_ = declare_parameter<double>("stand_kp", 50.0);  // RAMP/HOLD gains (stand up from lying, Unitree example)
    stand_kd_ = declare_parameter<double>("stand_kd", 3.0);
    blend_s_ = declare_parameter<double>("gain_blend_s", 1.0);  // stand gains -> policy gains after start
    sit_s_ = declare_parameter<double>("sit_s", 1.8);        // stand -> sit_mid
    sit2_s_ = declare_parameter<double>("sit_slow_s", 2.2);  // sit_mid -> lying (slow: chin lidar touches here)
    back_s_ = declare_parameter<double>("back_to_stand_s", 1.0);
    max_tilt_ = declare_parameter<double>("max_tilt", 0.8);
    state_timeout_ = declare_parameter<double>("state_timeout_ms", 100.0) * 1e-3;
    damp_kd_ = declare_parameter<double>("damping_kd", 4.0);
    consistent_ = declare_parameter<bool>("shadow_consistent", false);
    rt_prio_ = declare_parameter<int>("rt_priority", 80);
    vmax_ = declare_parameter<std::vector<double>>("cmd_max", {0.8, 0.4, 1.0});
    std::string log_dir = declare_parameter<std::string>("log_dir", "/tmp");

    pol_ = std::make_unique<hipan::Policy>(weights_);
    if (pol_->D != 62) throw std::runtime_error("expected 62-dim observation");
    cmd_ = {0, 0, 0, (float)h_default_, 0};

    sub_ = create_subscription<unitree_go::msg::LowState>(
        "/lowstate", rclcpp::QoS(1).best_effort(), [this](unitree_go::msg::LowState::SharedPtr m) {
          std::lock_guard<std::mutex> lk(mu_);
          ls_ = *m;
          ls_time_ = now_s();
          have_ls_ = true;
        });
    // safe stop: stand -> sit down -> exit (never kill the node while the robot stands; it would collapse)
    exit_sub_ = create_subscription<std_msgs::msg::Empty>(
        "/hipan/sit_and_exit", 10, [this](std_msgs::msg::Empty::SharedPtr) {
          exit_req_ = true;
          RCLCPP_WARN(get_logger(), "sit_and_exit requested");
        });
    cmd_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/hipan/cmd", 10, [this](std_msgs::msg::Float32MultiArray::SharedPtr m) {
          if (m->data.size() != 5) return;
          std::lock_guard<std::mutex> lk(mu_);
          for (int i = 0; i < 5; ++i) topic_cmd_[i] = m->data[i];
          topic_cmd_time_ = now_s();
        });
    if (live_) pub_ = create_publisher<unitree_go::msg::LowCmd>("/lowcmd", 10);
    // 50 Hz debug: [state, cmd(5), est(5: vx vy vz h roll), roll, pitch, yaw, contact(4), obs(62)]
    dbg_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("/hipan/debug", 10);

    char fn[256];
    std::time_t tt = std::time(nullptr);
    std::strftime(fn, sizeof(fn), "/hipan_low_%Y%m%d_%H%M%S.bin", std::localtime(&tt));
    log_path_ = log_dir + fn;
    log_.open(log_path_, std::ios::binary);
    RCLCPP_INFO(get_logger(), "mode=%s cmd_source=%s weights=%s log=%s", live_ ? "LIVE" : "shadow",
                cmd_topic_ ? "topic" : "remote", weights_.c_str(), log_path_.c_str());
    if (live_)
      RCLCPP_WARN(get_logger(), "LIVE: sport mode must be released; L2+A starts the policy after HOLD, L2+B = damp");
    running_ = true;
    th_ = std::thread([this] { loop(); });
  }

  ~HipanLowNode() override {
    running_ = false;
    if (th_.joinable()) th_.join();
    log_.close();
  }

 private:
  static double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
  }

  void loop() {
    sched_param sp{};
    sp.sched_priority = rt_prio_;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) != 0)
      RCLCPP_WARN(get_logger(), "SCHED_FIFO not granted (run with sudo / rtprio limits); timing may jitter");
    timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    const long period_ns = 20000000;  // 50 Hz
    double t0 = now_s(), last = t0;
    while (running_ && rclcpp::ok()) {
      next.tv_nsec += period_ns;
      while (next.tv_nsec >= 1000000000) {
        next.tv_nsec -= 1000000000;
        next.tv_sec += 1;
      }
      clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
      double t = now_s();
      tick(t - t0, t - last);
      last = t;
    }
  }

  void damp(const char* why) {
    if (state_ != DAMP) RCLCPP_ERROR(get_logger(), "DAMP: %s", why);
    state_ = DAMP;
  }

  void tick(double t, double dt) {
    unitree_go::msg::LowState ls;
    double age;
    std::array<float, 5> tcmd;
    double tcmd_age;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (!have_ls_) return;
      ls = ls_;
      age = now_s() - ls_time_;
      tcmd = topic_cmd_;
      tcmd_age = now_s() - topic_cmd_time_;
    }
    // ---- sensors in policy order
    float q[12], dq[12], tau[12], raw[4], contact[4], roll, pitch, gyro[3];
    for (int i = 0; i < 12; ++i) {
      const auto& m = ls.motor_state[hipan::kIsaacToSdk[i]];
      q[i] = m.q, dq[i] = m.dq, tau[i] = m.tau_est;
    }
    for (int k = 0; k < 4; ++k) {
      raw[k] = ls.foot_force[hipan::kFootSdkToPolicy[k]];
      contact[k] = raw[k] > thr_[k] ? 1.f : 0.f;
    }
    hipan::roll_pitch(ls.imu_state.quaternion[0], ls.imu_state.quaternion[1], ls.imu_state.quaternion[2],
                      ls.imu_state.quaternion[3], roll, pitch);
    for (int i = 0; i < 3; ++i) gyro[i] = ls.imu_state.gyroscope[i];
    Remote rc = Remote::parse(ls.wireless_remote);

    // ---- command
    auto dz = [](float v) { return std::abs(v) < 0.1f ? 0.f : v; };
    const bool sticks = std::abs(rc.lx) > 0.2f || std::abs(rc.ly) > 0.2f || std::abs(rc.rx) > 0.2f;
    if (cmd_topic_ && !sticks) {  // high-level commands; any stick movement takes over (manual override)
      if (tcmd_age < 0.5) cmd_ = tcmd;
      else cmd_ = {0, 0, 0, (float)h_default_, 0};
      manual_ = false;
    } else {
      if (cmd_topic_ && !manual_) {
        manual_ = true;
        RCLCPP_WARN(get_logger(), "manual override (stick moved): /hipan/cmd ignored while the sticks are used");
      }
      cmd_[0] = dz(rc.ly) * vmax_[0];
      cmd_[1] = -dz(rc.lx) * vmax_[1];
      cmd_[2] = -dz(rc.rx) * vmax_[2];
      bool up = rc.key(12), down = rc.key(14), left = rc.key(15), right = rc.key(13);
      if (up && !prev_.key(12)) cmd_[3] += 0.02f;
      if (down && !prev_.key(14)) cmd_[3] -= 0.02f;
      if (left && !prev_.key(15)) cmd_[4] -= 0.05f;
      if (right && !prev_.key(13)) cmd_[4] += 0.05f;
    }
    cmd_[0] = std::max(-0.5f, std::min(1.0f, cmd_[0]));
    cmd_[1] = std::max(-0.4f, std::min(0.4f, cmd_[1]));
    cmd_[2] = std::max(-1.2f, std::min(1.2f, cmd_[2]));
    cmd_[3] = std::max(0.12f, std::min(0.36f, cmd_[3]));
    cmd_[4] = std::max(-0.5f, std::min(0.5f, cmd_[4]));

    // ---- state machine + safety
    bool estop = rc.key(5) && rc.key(9);                          // L2 + B (emergency)
    bool l2a = rc.key(5) && rc.key(8) && !(prev_.key(5) && prev_.key(8));  // L2 + A, edge
    bool start = rc.key(2) && !prev_.key(2);                      // start button, edge
    if (state_ == WAIT) {
      if (live_) {
        begin_ramp(q, pol_->default_q, ramp_s_, HOLD, t);
      } else {
        state_ = POLICY;
        begin_policy(q);
      }
      RCLCPP_INFO(get_logger(), "state -> %s", kStateName[state_]);
    }
    if (state_ != DAMP) {
      if (estop) damp("remote L2+B");
      else if (age > state_timeout_) damp("lowstate timeout");
      else if (live_ && (std::abs(roll) > max_tilt_ || std::abs(pitch) > max_tilt_)) damp("tilt");
      else if (live_ && (state_ == HOLD || state_ == POLICY)) {  // lying pose sits on the calf stop
        for (int i = 0; i < 12; ++i)
          if (q[i] < hipan::kQMin[i] - 0.05f || q[i] > hipan::kQMax[i] + 0.05f) {
            damp("joint limit");
            break;
          }
      }
    }
    if (calib_n_ >= 0 && live_ && state_ == RAMP && after_ramp_ == HOLD && t - ramp_t0_ < 0.4) {
      for (int k = 0; k < 4; ++k) calib_sum_[k] += raw[k];  // lying, feet unloaded
      ++calib_n_;
    } else if (calib_n_ > 5) {
      for (int k = 0; k < 4; ++k) thr_[k] = std::max<float>(calib_sum_[k] / calib_n_ + contact_margin_, 8.f);
      RCLCPP_INFO(get_logger(), "foot contact thresholds (unloaded baseline + %.0f): FL %.1f FR %.1f RL %.1f RR %.1f",
                  contact_margin_, thr_[0], thr_[1], thr_[2], thr_[3]);
      calib_n_ = -1;
    }
    if (state_ == RAMP && t - ramp_t0_ >= ramp_dur_ && sit_phase2_) {
      sit_phase2_ = false;
      begin_ramp(q, kLyingQ, sit2_s_, DOWN, t);
    }
    if (state_ == RAMP && t - ramp_t0_ >= ramp_dur_) {
      state_ = after_ramp_;
      RCLCPP_INFO(get_logger(), "state -> %s%s", kStateName[state_],
                  state_ == HOLD ? " (standing). start = walk policy, L2+A = sit down" : " (lying). L2+A = stand up");
    }
    if (exit_req_) {  // safe stop sequence
      if (!live_) {
        finish();
      } else if (state_ == POLICY) {
        begin_ramp(q, pol_->default_q, back_s_, HOLD, t);
        RCLCPP_INFO(get_logger(), "exit: policy stopped -> standing pose");
      } else if (state_ == HOLD) {
        start_sit(q, t);
        RCLCPP_INFO(get_logger(), "exit: sitting down");
      } else if (state_ == DOWN || state_ == DAMP) {
        if (down_t0_ < 0) down_t0_ = t;
        if (t - down_t0_ > 1.0) finish();
      }
    } else if (live_) {
      if (state_ == HOLD && start) {
        state_ = POLICY;
        policy_t0_ = t;
        begin_policy(q);
        RCLCPP_INFO(get_logger(), "state -> POLICY (L2+A = stop and stand)");
      } else if (state_ == POLICY && l2a) {
        begin_ramp(q, pol_->default_q, back_s_, HOLD, t);
        RCLCPP_INFO(get_logger(), "policy stopped -> back to the standing pose");
      } else if (state_ == HOLD && l2a) {
        start_sit(q, t);
        RCLCPP_INFO(get_logger(), "sitting down");
      } else if (state_ == DOWN && l2a) {
        begin_ramp(q, pol_->default_q, ramp_s_, HOLD, t);
        RCLCPP_INFO(get_logger(), "standing up");
      }
    }

    // ---- policy (also in shadow)
    hipan::Vec obs(62), action = hipan::Vec::Zero(12), est = hipan::Vec::Zero(5);
    std::array<float, 12> target{};
    if (state_ == POLICY) {
      if (consistent_ && !live_)
        for (int i = 0; i < 12; ++i) last_action_[i] = (q[i] - pol_->offset[i]) / pol_->scale[i];
      auto fp = hipan::feet_pos_body(q);
      int k = 0;
      for (int i = 0; i < 12; ++i) obs[k++] = q[i] - pol_->default_q[i];
      for (int i = 0; i < 12; ++i) obs[k++] = dq[i] * 0.05f;
      for (int i = 0; i < 12; ++i) obs[k++] = fp[i];
      for (int i = 0; i < 4; ++i) obs[k++] = contact[i];
      obs[k++] = roll;
      obs[k++] = pitch;
      for (int i = 0; i < 3; ++i) obs[k++] = gyro[i] * 0.25f;
      for (int i = 0; i < 12; ++i) obs[k++] = last_action_[i];
      for (int i = 0; i < 5; ++i) obs[k++] = cmd_[i];
      pol_->step(obs, action, est);
      if (!action.allFinite()) damp("non-finite action");
      for (int i = 0; i < 12; ++i) {
        last_action_[i] = action[i];
        target[i] = pol_->offset[i] + pol_->scale[i] * action[i];
      }
    } else {
      obs.setZero();
    }

    // ---- command out
    if (live_) {
      unitree_go::msg::LowCmd cmd;
      cmd.head[0] = 0xFE;
      cmd.head[1] = 0xEF;
      cmd.level_flag = 0xFF;
      cmd.gpio = 0;
      for (int j = 0; j < 20; ++j) {
        cmd.motor_cmd[j].mode = 0x01;
        cmd.motor_cmd[j].q = kPosStop;
        cmd.motor_cmd[j].dq = kVelStop;
        cmd.motor_cmd[j].kp = 0;
        cmd.motor_cmd[j].kd = 0;
        cmd.motor_cmd[j].tau = 0;
      }
      for (int i = 0; i < 12; ++i) {
        auto& m = cmd.motor_cmd[hipan::kIsaacToSdk[i]];
        float qd = target[i];
        float kp = pol_->kp[i], kd = pol_->kd[i];
        if (state_ == POLICY) {  // blend from the stand gains to the trained gains
          float b = std::min(1.0, (t - policy_t0_) / blend_s_);
          kp = (1 - b) * stand_kp_ + b * kp;
          kd = (1 - b) * stand_kd_ + b * kd;
        }
        if (state_ == RAMP || state_ == HOLD || state_ == DOWN) {
          kp = state_ == DOWN ? 20.f : stand_kp_;
          kd = state_ == DOWN ? 2.f : stand_kd_;
          float a = std::min(1.0, (t - ramp_t0_) / ramp_dur_);
          a = 0.5f - 0.5f * std::cos(a * 3.14159265f);  // smooth start and stop
          qd = (1 - a) * ramp_from_[i] + a * ramp_to_[i];
          target[i] = qd;
        }
        if (state_ == DAMP) {
          m.q = 0;
          m.dq = 0;
          m.kp = 0;
          m.kd = damp_kd_;
          m.tau = 0;
        } else {
          m.q = qd;
          m.dq = 0;
          m.kp = kp;
          m.kd = kd;
          m.tau = 0;
        }
      }
      crc::set_crc(cmd);
      pub_->publish(cmd);
    }

    {
      std_msgs::msg::Float32MultiArray dm;
      dm.data.reserve(81);
      dm.data.push_back(state_);
      for (int i = 0; i < 5; ++i) dm.data.push_back(cmd_[i]);
      for (int i = 0; i < 5; ++i) dm.data.push_back(est[i]);
      dm.data.push_back(roll);
      dm.data.push_back(pitch);
      dm.data.push_back(ls.imu_state.rpy[2]);
      for (int i = 0; i < 4; ++i) dm.data.push_back(contact[i]);
      for (int i = 0; i < 62; ++i) dm.data.push_back(obs[i]);  // low student observation (for the nav student)
      dbg_pub_->publish(dm);
    }

    // ---- log + status
    LogRow r{};
    r.t = t;
    r.state = state_;
    r.loop_dt = dt;
    r.state_age = age;
    for (int i = 0; i < 62; ++i) r.obs[i] = obs[i];
    for (int i = 0; i < 12; ++i) {
      r.action[i] = action[i];
      r.target[i] = target[i];
      r.q[i] = q[i];
      r.dq[i] = dq[i];
      r.tau_est[i] = tau[i];
    }
    for (int i = 0; i < 5; ++i) r.est[i] = est[i], r.cmd[i] = cmd_[i];
    for (int i = 0; i < 4; ++i) r.foot_raw[i] = raw[i];
    r.rpy[0] = roll, r.rpy[1] = pitch, r.rpy[2] = ls.imu_state.rpy[2];
    log_.write(reinterpret_cast<const char*>(&r), sizeof(r));
    max_dt_ = std::max(max_dt_, dt);
    if (++n_ % 50 == 0) {
      float err = 0;
      for (int i = 0; i < 12; ++i) err = std::max(err, std::abs(target[i] - q[i]));
      RCLCPP_INFO(get_logger(),
                  "%s t=%.1f dt_max=%.1fms age=%.1fms cmd=[%.2f %.2f %.2f %.2f %.2f] est v=(%.2f %.2f) h=%.3f "
                  "|tgt-q|max=%.3f contact=%d%d%d%d",
                  kStateName[state_], t, max_dt_ * 1e3, age * 1e3, cmd_[0], cmd_[1], cmd_[2], cmd_[3], cmd_[4],
                  est[0], est[1], est[3], err, (int)contact[0], (int)contact[1], (int)contact[2], (int)contact[3]);
      max_dt_ = 0;
    }
    prev_ = rc;
  }

  void finish() {
    if (!running_) return;
    RCLCPP_WARN(get_logger(), "exit: robot is %s, node shutting down", kStateName[state_]);
    running_ = false;
    rclcpp::shutdown();
  }

  void start_sit(const float* q, double t) {
    begin_ramp(q, kSitMidQ, sit_s_, DOWN, t);
    sit_phase2_ = true;
  }

  void begin_ramp(const float* q, const std::array<float, 12>& to, double dur, State after, double t) {
    std::copy(q, q + 12, ramp_from_.begin());
    ramp_to_ = to;
    ramp_dur_ = dur;
    ramp_t0_ = t;
    after_ramp_ = after;
    state_ = RAMP;
  }

  void begin_policy(const float* q) {
    pol_->reset();
    for (int i = 0; i < 12; ++i) last_action_[i] = (q[i] - pol_->offset[i]) / pol_->scale[i];
  }

  // params
  std::string weights_, log_path_;
  bool live_ = false, cmd_topic_ = false, consistent_ = false;
  double contact_thr_, h_default_, ramp_s_, max_tilt_, state_timeout_, damp_kd_, stand_kp_, stand_kd_, blend_s_, sit_s_, sit2_s_, back_s_;
  int rt_prio_;
  std::vector<double> vmax_;
  // ros
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr cmd_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr exit_sub_;
  std::atomic<bool> exit_req_{false};
  double down_t0_ = -1;
  rclcpp::Publisher<unitree_go::msg::LowCmd>::SharedPtr pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr dbg_pub_;
  // shared
  std::mutex mu_;
  unitree_go::msg::LowState ls_;
  double ls_time_ = 0, topic_cmd_time_ = -1e9;
  bool have_ls_ = false;
  std::array<float, 5> topic_cmd_{};
  // control
  std::unique_ptr<hipan::Policy> pol_;
  std::array<float, 5> cmd_{};
  std::array<float, 12> last_action_{}, ramp_from_{}, ramp_to_{};
  double ramp_dur_ = 1.0;
  State after_ramp_ = HOLD;
  bool sit_phase2_ = false;
  bool manual_ = false;
  std::array<float, 4> thr_{}, calib_sum_{};
  int calib_n_ = 0;
  double contact_margin_ = 5.0;
  State state_ = WAIT;
  double ramp_t0_ = 0, policy_t0_ = 0, max_dt_ = 0;
  long n_ = 0;
  Remote prev_;
  std::ofstream log_;
  std::thread th_;
  std::atomic<bool> running_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HipanLowNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
