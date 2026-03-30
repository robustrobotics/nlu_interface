#ifndef NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_
#define NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_

// STL
#include <optional>

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
#include <QComboBox>
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

protected:
  void resizeEvent(QResizeEvent *event) override {
    QLabel::resizeEvent(event);
    updateScaledPixmap();
    return;
  }

  void mousePressEvent(QMouseEvent *event) override {
    std::cout << "In mousePressEvent()" << std::endl;
    if (event->button() == Qt::LeftButton) {
      auto label_x = event->pos().x();
      auto label_y = event->pos().y();

      double scale = std::min(static_cast<double>(this->size().width()) /
                                  original_pixmap_.size().width(),
                              static_cast<double>(this->size().height()) /
                                  original_pixmap_.size().height());

      double offset_x =
          (this->size().width() - original_pixmap_.size().width() * scale) /
          2.0;
      double offset_y =
          (this->size().height() - original_pixmap_.size().height() * scale) /
          2.0;

      int image_x = static_cast<int>((label_x - offset_x) / scale);
      int image_y = static_cast<int>((label_y - offset_y) / scale);

      std::cout << "Left Clicked Location: " << label_x << "," << label_y
                << std::endl;
      std::cout << "Scale: " << scale << " Offset: " << offset_x << ","
                << offset_y << std::endl;
      std::cout << "Left Clicked Pixel: " << image_x << "," << image_y
                << std::endl;

      // Only emit if click is within the image bounds
      if (image_x < 0 || image_x >= original_pixmap_.width() || image_y < 0 ||
          image_y >= original_pixmap_.height()) {
        return;
      }

      emit pixelClicked(image_x, image_y);
    }
    QLabel::mousePressEvent(event);
    return;
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
  bool received_first_message_ = false;
  bool has_detection_ = false;
  int current_image_index_ = 0;
  int detection_image_index_ = 0;
  int detection_x_ = 0;
  int detection_y_ = 0;
  int selected_image_index_ = 0;
  std::optional<std::pair<int, int>> selected_pixel_;

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
  QComboBox *p_manipulation_robot_id_combo_box_;
  ScaledClickableLabel *p_manipulation_image_label_;
  QPixmap *p_base_pixmap_;
  QLabel *p_selected_pixels_label_;
  QPushButton *p_manipulation_publish_push_button_;
  QPushButton *p_manipulation_reject_push_button_;
  QPushButton *p_manipulation_reset_push_button_;
  QPushButton *p_manipulation_set_detection_toggle_button_;
  QPushButton *p_prev_image_button_;
  QPushButton *p_next_image_button_;
  QLabel *p_image_index_label_;
  QTimer *p_timer_;

private Q_SLOTS:
  void handlePixelClick(int x, int y);
  void publishManipulationApproval(void);
  void publishManipulationRejection(void);
  void resetSelection(void);
  void publishSystemMonitor(void);
  void showPrevImage(void);
  void showNextImage(void);
}; // class ManipulationApprovalPanel
} // namespace nlu_interface_rviz

#endif // NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_
