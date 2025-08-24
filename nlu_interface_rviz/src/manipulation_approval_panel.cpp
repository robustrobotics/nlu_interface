#include <QImage>
#include <QVBoxLayout>
#include <cv_bridge/cv_bridge.hpp>
#include <nlu_interface_rviz/manipulation_approval_panel.hpp>
#include <rviz_common/display_context.hpp>

namespace nlu_interface_rviz {
ManipulationApprovalPanel::ManipulationApprovalPanel(QWidget *parent)
    : Panel(parent) {
  // Create the layout for a display of a manipulation request image
  QVBoxLayout *p_manipulation_request_layout = new QVBoxLayout;
  p_manipulation_request_layout->addWidget(
      new QLabel("Manipulation Approval Request"));
  p_manipulation_image_label_ = new QLabel("Manipulation Image Placeholder");
  p_manipulation_request_layout->addWidget(p_manipulation_image_label_);

  // Create the layout for the approval robot id
  QHBoxLayout *p_manipulation_robot_id_combo_box_layout = new QHBoxLayout;
  p_manipulation_robot_id_combo_box_layout->addWidget(new QLabel("Robot ID"));
  p_manipulation_robot_id_combo_box_ = new QComboBox;
  p_manipulation_robot_id_combo_box_layout->addWidget(
      p_manipulation_robot_id_combo_box_);

  QHBoxLayout *p_manipulation_approval_layout = new QHBoxLayout;
  p_manipulation_approve_push_button_ = new QPushButton("Approve");
  p_manipulation_approval_layout->addWidget(
      p_manipulation_approve_push_button_);
  p_manipulation_reject_push_button_ = new QPushButton("Reject");
  p_manipulation_approval_layout->addWidget(p_manipulation_reject_push_button_);

  // Organize the layouts vertically
  QVBoxLayout *p_layout = new QVBoxLayout;
  p_layout->addLayout(p_manipulation_request_layout);
  p_layout->addLayout(p_manipulation_robot_id_combo_box_layout);
  p_layout->addLayout(p_manipulation_approval_layout);
  setLayout(p_layout);

  // Create a timer to regularly publish to the system monitor
  QTimer *p_timer_ = new QTimer(this);

  // Create Qt connections
  QObject::connect(p_manipulation_approve_push_button_, SIGNAL(clicked()), this,
                   SLOT(publishManipulationApproval()));
  QObject::connect(p_manipulation_reject_push_button_, SIGNAL(clicked()), this,
                   SLOT(publishManipulationRejection()));
  QObject::connect(p_timer_, &QTimer::timeout, this,
                   &ManipulationApprovalPanel::publishSystemMonitor);
  p_timer_->start(500);
  return;
}

void ManipulationApprovalPanel::onInitialize() {
  p_node_abstraction_ = getDisplayContext()->getRosNodeAbstraction().lock();
  rclcpp::Node::SharedPtr node = p_node_abstraction_->get_raw_node();

  // Get the robot ids & populate the relevant combo boxes
  if (!node->has_parameter("robot_ids")) {
    node->declare_parameter<std::vector<std::string>>("robot_ids", {"euclid"});
  }
  auto param_robot_ids = node->get_parameter("robot_ids");

  for (auto &s : param_robot_ids.as_string_array()) {
    robot_ids_.insert(s.c_str());
  }

  p_manipulation_robot_id_combo_box_->addItems(
      QList<QString>(robot_ids_.begin(), robot_ids_.end()));

  // Create the publishers
  system_monitor_publisher_ =
      node->create_publisher<ros_system_monitor_msgs::msg::NodeInfoMsg>(
          "~/node_status", 1);
  // Create a manipulation approval publisher and request subscriptions
  // namespaced for each robot
  for (auto const &q_robot_id : robot_ids_) {
    auto const robot_id = q_robot_id.toStdString();
    auto const request_topic =
        "/" + robot_id + "/spot_executor_node/annotated_image";
    manipulation_request_subscriptions_.emplace(
        robot_id,
        node->create_subscription<sensor_msgs::msg::Image>(
            request_topic, 10,
            [this, robot_id](sensor_msgs::msg::Image::ConstSharedPtr msg) {
              this->handleManipulationRequest(msg, robot_id);
            }));
    auto const approval_topic =
        "/" + robot_id + "/spot_executor_node/pick_confirmation";
    manipulation_approval_publishers_.emplace(
        robot_id,
        node->create_publisher<std_msgs::msg::Bool>(approval_topic, 1));
  }
  return;
}

void ManipulationApprovalPanel::handleManipulationRequest(
    sensor_msgs::msg::Image::ConstSharedPtr msg, std::string const &robot_id) {

  // Set the combo box to the robot id
  p_manipulation_robot_id_combo_box_->setCurrentText(robot_id.c_str());

  // Convert msg to cv::Mat
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception &e) {
    throw std::runtime_error("cv_bridge exception: " + std::string(e.what()));
  }
  auto cv_mat = cv_ptr->image;

  // Convert cv::Mat to QImage
  auto qimage = QImage(cv_mat.data, cv_mat.cols, cv_mat.rows, cv_mat.step,
                       QImage::Format_BGR888);

  // Convert QImage to QPixmap
  auto qpixmap = QPixmap::fromImage(qimage);
  qpixmap = qpixmap.scaled(320, 240, Qt::KeepAspectRatio);

  // Set the QLabel's pixmap
  p_manipulation_image_label_->setPixmap(qpixmap);
  return;
}

void ManipulationApprovalPanel::publishManipulationApproval(void) {
  publishManipulationResponse(true);
  return;
}

void ManipulationApprovalPanel::publishManipulationRejection(void) {
  publishManipulationResponse(false);
  return;
}

void ManipulationApprovalPanel::publishManipulationResponse(
    bool const approve) {
  // Construct the approval message (bool)
  auto msg = std_msgs::msg::Bool();
  msg.data = approve;
  // Get the publisher for the selected robot id
  auto it_publisher = manipulation_approval_publishers_.find(
      p_manipulation_robot_id_combo_box_->currentText().toStdString());
  assert(it_publisher != manipulation_approval_publishers_.end());
  // Publish the approval message
  it_publisher->second->publish(msg);
  return;
}

void ManipulationApprovalPanel::publishSystemMonitor(void) {
  // Get the ROS2 node
  auto p_node = p_node_abstraction_->get_raw_node();

  // Construct the message
  auto msg = ros_system_monitor_msgs::msg::NodeInfoMsg();
  msg.nickname = "nlu_rviz_manipulation_panel";
  msg.node_name = p_node->get_fully_qualified_name();
  msg.status = ros_system_monitor_msgs::msg::NodeInfoMsg::NOMINAL;
  msg.notes = "You gonna give me orders?";
  system_monitor_publisher_->publish(msg);
  return;
}

} // namespace nlu_interface_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(nlu_interface_rviz::ManipulationApprovalPanel,
                       rviz_common::Panel)
