// Voxel-hashed global map from FAST-LIO /cloud_registered (world frame).
//  - keeps one point per save_voxel (default 5 cm) -> saved as PCD on request
//  - keeps one point per display_voxel (default 10 cm) -> /map_cloud every display_period s
//  - services: ~/save_map (std_srvs/Trigger), ~/clear_map (std_srvs/Trigger)
// Replaces FAST-LIO's ROS2 map_save (which writes an always-empty buffer and crashes)
// and keeps rviz from accumulating every scan in memory.
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <unordered_map>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>

using PointT = pcl::PointXYZI;

namespace {
inline std::uint64_t voxel_key(float x, float y, float z, float inv) {
  // 21 bits per axis (+-1M voxels) -> plenty for indoor/outdoor maps at 5 cm
  auto q = [inv](float v) { return static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(v * inv)) + (1 << 20)) & 0x1FFFFF; };
  return (q(x) << 42) | (q(y) << 21) | q(z);
}
}  // namespace

class MapAccumulator : public rclcpp::Node {
public:
  MapAccumulator() : Node("map_accumulator") {
    save_voxel_ = declare_parameter<double>("save_voxel", 0.05);
    display_voxel_ = declare_parameter<double>("display_voxel", 0.10);
    save_dir_ = declare_parameter<std::string>("save_dir", std::string(std::getenv("HOME")) + "/Desktop/3DOG/maps");
    frame_id_ = declare_parameter<std::string>("frame_id", "camera_init");
    const double period = declare_parameter<double>("display_period", 2.0);

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        declare_parameter<std::string>("input_topic", "/cloud_registered"), rclcpp::QoS(20),
        [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_cloud(*m); });
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("/map_cloud", rclcpp::QoS(1).transient_local());
    timer_ = create_wall_timer(std::chrono::duration<double>(period), [this] { publish_display(); });

    save_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/save_map", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> res) { save(*res); });
    clear_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/clear_map", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                              std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
          std::lock_guard<std::mutex> lk(mtx_);
          save_map_.clear();
          disp_map_.clear();
          res->success = true;
          res->message = "cleared";
        });
    RCLCPP_INFO(get_logger(), "map_accumulator: save voxel %.2f m, display voxel %.2f m, save dir %s",
                save_voxel_, display_voxel_, save_dir_.c_str());
  }

private:
  void on_cloud(const sensor_msgs::msg::PointCloud2 &msg) {
    pcl::PointCloud<PointT> cloud;
    pcl::fromROSMsg(msg, cloud);
    const float inv_s = 1.0f / save_voxel_, inv_d = 1.0f / display_voxel_;
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto &p : cloud.points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      save_map_.emplace(voxel_key(p.x, p.y, p.z, inv_s), p);
      if (disp_map_.emplace(voxel_key(p.x, p.y, p.z, inv_d), p).second) disp_dirty_ = true;
    }
  }

  void publish_display() {
    pcl::PointCloud<PointT> out;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      if (!disp_dirty_) return;
      out.reserve(disp_map_.size());
      for (const auto &kv : disp_map_) out.push_back(kv.second);
      disp_dirty_ = false;
    }
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(out, msg);
    msg.header.frame_id = frame_id_;
    msg.header.stamp = now();
    pub_->publish(msg);
  }

  void save(std_srvs::srv::Trigger::Response &res) {
    pcl::PointCloud<PointT> out;
    {
      std::lock_guard<std::mutex> lk(mtx_);
      out.reserve(save_map_.size());
      for (const auto &kv : save_map_) out.push_back(kv.second);
    }
    if (out.empty()) {
      res.success = false;
      res.message = "map is empty (is FAST-LIO publishing /cloud_registered?)";
      return;
    }
    out.width = out.size();
    out.height = 1;
    out.is_dense = true;
    std::filesystem::create_directories(save_dir_);
    char stamp[32];
    std::time_t t = std::time(nullptr);
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&t));
    const std::string path = save_dir_ + "/map_" + stamp + ".pcd";
    try {
      pcl::io::savePCDFileBinary(path, out);
    } catch (const std::exception &e) {
      res.success = false;
      res.message = std::string("save failed: ") + e.what();
      return;
    }
    res.success = true;
    res.message = "saved " + std::to_string(out.size()) + " points to " + path;
    RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
  }

  double save_voxel_, display_voxel_;
  std::string save_dir_, frame_id_;
  std::mutex mtx_;
  std::unordered_map<std::uint64_t, PointT> save_map_, disp_map_;
  bool disp_dirty_ = false;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_, clear_srv_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MapAccumulator>());
  rclcpp::shutdown();
  return 0;
}
