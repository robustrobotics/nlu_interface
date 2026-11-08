#ifndef NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_
#define NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_

// STL
#include <array>
#include <optional>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>

// ROS
#include "rclcpp/rclcpp.hpp"
#include <nlu_interface_rviz/msg/manipulation_approval_request.hpp>
#include <nlu_interface_rviz/msg/manipulation_approval_response.hpp>
#include <omniplanner_msgs/msg/goto_points_goal_msg.hpp>
#include <omniplanner_msgs/msg/language_goal_msg.hpp>
#include <ros_system_monitor_msgs/msg/node_info_msg.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>

// Qt

#include <QCheckBox>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QPen>
#include <QPushButton>
#include <QResizeEvent>
#include <QSet>
#include <QString>
#include <QTimer>

namespace nlu_interface_rviz {
class ScaledClickableLabel : public QLabel {
  Q_OBJECT

public:
  using QLabel::QLabel;

  void setOriginalPixmap(QPixmap const &pmap) {
    original_pixmap_ = pmap;
    updateScaledPixmap();
    return;
  }

signals:
  void pixelClicked(int x, int y);
  void pixelHovered(int x, int y);
  void hoverLeft();

protected:
  void resizeEvent(QResizeEvent *event) override {
    QLabel::resizeEvent(event);
    updateScaledPixmap();
    return;
  }

  // Map a widget position to a full-resolution image pixel; false if outside.
  bool toImagePixel(QPoint const &pos, int &image_x, int &image_y) const {
    if (original_pixmap_.isNull())
      return false;
    double scale = std::min(static_cast<double>(this->size().width()) /
                                original_pixmap_.size().width(),
                            static_cast<double>(this->size().height()) /
                                original_pixmap_.size().height());
    double offset_x =
        (this->size().width() - original_pixmap_.size().width() * scale) / 2.0;
    double offset_y =
        (this->size().height() - original_pixmap_.size().height() * scale) /
        2.0;
    image_x = static_cast<int>((pos.x() - offset_x) / scale);
    image_y = static_cast<int>((pos.y() - offset_y) / scale);
    return image_x >= 0 && image_x < original_pixmap_.width() && image_y >= 0 &&
           image_y < original_pixmap_.height();
  }

  void mousePressEvent(QMouseEvent *event) override {
    int image_x = 0, image_y = 0;
    if (event->button() == Qt::LeftButton &&
        toImagePixel(event->pos(), image_x, image_y)) {
      emit pixelClicked(image_x, image_y);
    }
    QLabel::mousePressEvent(event);
    return;
  }

  void mouseMoveEvent(QMouseEvent *event) override {
    int image_x = 0, image_y = 0;
    if (toImagePixel(event->pos(), image_x, image_y)) {
      emit pixelHovered(image_x, image_y);
    } else {
      emit hoverLeft();
    }
    QLabel::mouseMoveEvent(event);
  }

  void leaveEvent(QEvent *event) override {
    emit hoverLeft();
    QLabel::leaveEvent(event);
  }

private:
  void updateScaledPixmap() {
    if (!original_pixmap_.isNull()) {
      setAlignment(Qt::AlignCenter);
      QPixmap scaled_pixmap = original_pixmap_.scaled(
          size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
      QLabel::setPixmap(scaled_pixmap);
    }
  }

  QPixmap original_pixmap_;
}; // class ScaledClickableLabel

class ManipulationApprovalPanel : public rviz_common::Panel {
  Q_OBJECT
public:
  explicit ManipulationApprovalPanel(QWidget *parent = 0);
  ~ManipulationApprovalPanel() override = default;
  void onInitialize() override;

protected:
  void publishManipulationResponse(
      nlu_interface_rviz::msg::ManipulationApprovalResponse const &msg);
  void showCurrentImage();

  // ROS callbacks
  void handleManipulationRequest(
      nlu_interface_rviz::msg::ManipulationApprovalRequest::ConstSharedPtr msg,
      std::string const &robot_id);
  // Data members
  QSet<QString> robot_ids_;
  std::vector<QPixmap> candidate_pixmaps_;
  std::vector<std::string> image_source_names_;
  bool received_first_message_ = false;
  bool has_detection_ = false;
  int current_image_index_ = 0;
  int detection_image_index_ = 0;
  int detection_x_ = 0;
  int detection_y_ = 0;
  int selected_image_index_ = 0;
  std::optional<std::pair<int, int>> selected_pixel_;
  // Optional heading control for walk-to: the ground pixel Spot should stand
  // on, facing the selected pixel. Only offered when the rviz node parameter
  // `manipulation_walk_to_heading_selection` is true (off by default).
  bool heading_selection_enabled_ = false;
  int stand_image_index_ = 0;
  std::optional<std::pair<int, int>> stand_pixel_;

  // Depth registered to each candidate image (CV_16UC1, empty when absent),
  // its raw units per meter, and a colourised overlay for display only.
  std::vector<cv::Mat> depth_images_;
  std::vector<double> depth_scales_;
  std::vector<QPixmap> depth_overlays_;

  // Geometry behind the robot's walk-to estimate (vision frame), per image.
  struct CameraGeometry {
    bool valid = false;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::array<double, 3> origin{};
    std::array<double, 9> rotation{}; // row-major vision_R_camera
  };
  std::vector<CameraGeometry> cameras_;
  bool has_ground_ = false;
  std::array<double, 3> ground_point_{};
  std::array<double, 3> ground_normal_{};
  bool has_robot_ = false;
  double robot_x_ = 0, robot_y_ = 0;
  double walk_standoff_ = 0, walk_max_distance_ = 0;

  struct GroundEstimate {
    bool available = false; // geometry present for this image
    bool hit = false;       // ray meets the floor within range
    double distance = 0;    // horizontal robot -> target (m)
    double walk = 0;        // planned move (m); 0 = turn only
    bool capped = false;
  };
  GroundEstimate groundEstimate(int index, int x, int y) const;
  std::optional<double> depthAt(int index, int x, int y) const;
  bool estimateUnreliable(int index) const;
  std::string describePoint(int index, int x, int y) const;
  void updateWalkButtonLabel();

  // ROS2 member variables
  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface>
      p_node_abstraction_;
  rclcpp::Publisher<ros_system_monitor_msgs::msg::NodeInfoMsg>::SharedPtr
      system_monitor_publisher_;
  // map of robot ids to manipulation request subscriptions and approval
  // publishers
  std::map<std::string,
           rclcpp::Subscription<
               nlu_interface_rviz::msg::ManipulationApprovalRequest>::SharedPtr>
      manipulation_request_subscriptions_;
  std::map<
      std::string,
      rclcpp::Publisher<
          nlu_interface_rviz::msg::ManipulationApprovalResponse>::SharedPtr>
      manipulation_approval_publishers_;

  // GUI member variables
  QLabel *p_manipulation_robot_id_label_;
  ScaledClickableLabel *p_manipulation_image_label_;
  QPixmap *p_base_pixmap_ = nullptr;
  QLabel *p_selected_pixels_label_;
  QPushButton *p_manipulation_publish_push_button_;
  QPushButton *p_manipulation_reject_push_button_;
  QPushButton *p_manipulation_walk_to_push_button_;
  QPushButton *p_manipulation_reset_push_button_;
  QPushButton *p_manipulation_set_detection_toggle_button_;
  QPushButton *p_manipulation_set_stand_point_toggle_button_;
  QPushButton *p_prev_image_button_;
  QPushButton *p_next_image_button_;
  QLabel *p_image_index_label_;
  QLabel *p_status_label_;
  QCheckBox *p_show_depth_checkbox_;
  QLabel *p_cursor_label_;
  QLabel *p_estimate_warning_label_;
  QLabel *p_walk_note_label_;
  QTimer *p_timer_;

private Q_SLOTS:
  void handlePixelClick(int x, int y);
  void publishManipulationApproval(void);
  void publishManipulationRejection(void);
  void publishWalkTo(void);
  void resetSelection(void);
  void publishSystemMonitor(void);
  void handlePixelHover(int x, int y);
  void handleHoverLeft(void);
  void showPrevImage(void);
  void showNextImage(void);
}; // class ManipulationApprovalPanel
} // namespace nlu_interface_rviz

#endif // NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_
