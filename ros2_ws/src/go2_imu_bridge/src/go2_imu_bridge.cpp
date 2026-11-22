// Go2 /lowstate (unitree_go/LowState, ~500 Hz) -> /go2/imu (sensor_msgs/Imu)
// - stamped with this host's clock on receipt (lowstate has no header)
// - quaternion in LowState is [w, x, y, z]
// - accelerometer is specific force in m/s^2 (includes gravity, ~+9.8 on z at rest)
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <unitree_go/msg/low_state.hpp>

class Go2ImuBridge : public rclcpp::Node {
public:
  Go2ImuBridge() : Node("go2_imu_bridge") {
    frame_id_ = declare_parameter<std::string>("frame_id", "go2_imu");
    // reliable: FAST-LIO subscribes RELIABLE; best-effort subscribers still match
    pub_ = create_publisher<sensor_msgs::msg::Imu>("/go2/imu", rclcpp::QoS(rclcpp::KeepLast(200)).reliable());
    sub_ = create_subscription<unitree_go::msg::LowState>(
        "/lowstate", rclcpp::SensorDataQoS(),
        [this](unitree_go::msg::LowState::ConstSharedPtr m) { cb(*m); });
  }

private:
  void cb(const unitree_go::msg::LowState &m) {
    sensor_msgs::msg::Imu out;
    out.header.stamp = now();
    out.header.frame_id = frame_id_;
    const auto &s = m.imu_state;
    out.orientation.w = s.quaternion[0];
    out.orientation.x = s.quaternion[1];
    out.orientation.y = s.quaternion[2];
    out.orientation.z = s.quaternion[3];
    out.angular_velocity.x = s.gyroscope[0];
    out.angular_velocity.y = s.gyroscope[1];
    out.angular_velocity.z = s.gyroscope[2];
    out.linear_acceleration.x = s.accelerometer[0];
    out.linear_acceleration.y = s.accelerometer[1];
    out.linear_acceleration.z = s.accelerometer[2];
    out.orientation_covariance[0] = out.orientation_covariance[4] = out.orientation_covariance[8] = 1e-3;
    out.angular_velocity_covariance[0] = out.angular_velocity_covariance[4] = out.angular_velocity_covariance[8] = 1e-4;
    out.linear_acceleration_covariance[0] = out.linear_acceleration_covariance[4] = out.linear_acceleration_covariance[8] = 1e-2;
    pub_->publish(out);
  }

  std::string frame_id_;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr sub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Go2ImuBridge>());
  rclcpp::shutdown();
  return 0;
}
