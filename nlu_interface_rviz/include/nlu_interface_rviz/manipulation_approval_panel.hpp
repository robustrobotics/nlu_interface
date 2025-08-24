#ifndef NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_
#define NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_

// ROS
#include "rclcpp/rclcpp.hpp"
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

      double scale_x = static_cast<double>(this->size().width()) /
                       original_pixmap_.size().width();
      double scale_y = static_cast<double>(this->size().height()) /
                       original_pixmap_.size().height();

      int image_x = label_x / scale_x;
      int image_y = label_y / scale_y;

      std::cout << "Left Clicked Location: " << label_x << "," << label_y
                << std::endl;
      std::cout << "Scaling Factors:" << scale_x << "," << scale_y << std::endl;
      std::cout << "Left Clicked Pixel: " << image_x << "," << image_y
                << std::endl;
    }
    QLabel::mousePressEvent(event);
    return;
  }

private:
  void updateScaledPixmap() {
    if (!original_pixmap_.isNull()) {
      QPixmap scaled_pixmap = original_pixmap_.scaled(
          size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
      QLabel::setPixmap(scaled_pixmap);
    }
  }

  QPixmap original_pixmap_;
};

class ManipulationApprovalPanel : public rviz_common::Panel {
  Q_OBJECT
public:
  explicit ManipulationApprovalPanel(QWidget *parent = 0);
  ~ManipulationApprovalPanel() override = default;
  void onInitialize() override;

protected:
  void publishManipulationResponse(bool const approve);

  // ROS callbacks
  void handleManipulationRequest(sensor_msgs::msg::Image::ConstSharedPtr msg,
                                 std::string const &robot_id);
  // Data members
  QSet<QString> robot_ids_;

  // ROS2 member variables
  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface>
      p_node_abstraction_;
  rclcpp::Publisher<ros_system_monitor_msgs::msg::NodeInfoMsg>::SharedPtr
      system_monitor_publisher_;
  // map of robot ids to manipulation request subscriptions and approval
  // publishers
  std::map<std::string,
           rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr>
      manipulation_request_subscriptions_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr>
      manipulation_approval_publishers_;

  // GUI member variables
  QComboBox *p_manipulation_robot_id_combo_box_;
  // QLabel *p_manipulation_image_label_; // TODO: Display a sensor_msgs/Image
  // as a
  ScaledClickableLabel
      *p_manipulation_image_label_; // TODO: Display a sensor_msgs/Image as a
                                    // QLabel
  QPushButton *p_manipulation_approve_push_button_;
  QPushButton *p_manipulation_reject_push_button_;
  QTimer *p_timer_;

private Q_SLOTS:
  void publishManipulationApproval(void);
  void publishManipulationRejection(void);
  void publishSystemMonitor(void);
}; // class ManipulationApprovalPanel
} // namespace nlu_interface_rviz

#endif // NLU_INTERFACE_RVIZ__MANIPULATION_APPROVAL_PANEL_HPP_
