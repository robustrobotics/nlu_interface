#include <QImage>
#include <QVBoxLayout>
#include <cv_bridge/cv_bridge.hpp>
#include <nlu_interface_rviz/manipulation_approval_panel.hpp>
#include <rviz_common/display_context.hpp>

namespace nlu_interface_rviz {

QPixmap image_msg_to_pixmap(sensor_msgs::msg::Image::ConstSharedPtr msg) {
  // Convert msg to cv::Mat
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
  } catch (cv_bridge::Exception &e) {
    throw std::runtime_error("cv_bridge exception: " + std::string(e.what()));
  }
  auto cv_mat = cv_ptr->image;

  // Convert cv::Mat to QImage
  auto qimage = QImage(cv_mat.data, cv_mat.cols, cv_mat.rows, cv_mat.step,
                       QImage::Format_BGR888);

  // Convert QImage to QPixmap
  auto qpixmap = QPixmap::fromImage(qimage);
  return qpixmap;
}

ManipulationApprovalPanel::ManipulationApprovalPanel(QWidget *parent)
    : Panel(parent) {
  // Create the layout for a display of a manipulation request image
  QVBoxLayout *p_manipulation_request_layout = new QVBoxLayout;
  p_manipulation_request_layout->setContentsMargins(0, 0, 0, 0);
  p_manipulation_request_layout->setSpacing(0);
  p_manipulation_request_layout->setSizeConstraint(
      QLayout::SetDefaultConstraint);

  p_manipulation_image_label_ =
      new ScaledClickableLabel("Manipulation Image Placeholder");
  p_manipulation_image_label_->setMinimumSize(0, 0);
  p_manipulation_request_layout->addWidget(p_manipulation_image_label_, 1);

  // Create the layout for the image cycling buttons
  QHBoxLayout *p_image_cycle_layout = new QHBoxLayout;
  p_prev_image_button_ = new QPushButton("<");
  p_image_index_label_ = new QLabel("Image - / -");
  p_image_index_label_->setAlignment(Qt::AlignCenter);
  p_next_image_button_ = new QPushButton(">");
  p_image_cycle_layout->addWidget(p_prev_image_button_);
  p_image_cycle_layout->addWidget(p_image_index_label_, 1);
  p_image_cycle_layout->addWidget(p_next_image_button_);
  p_manipulation_request_layout->addLayout(p_image_cycle_layout);

  // Create the layout for the override toggle button
  QHBoxLayout *p_manipulation_override_toggle_layout = new QHBoxLayout;
  p_manipulation_set_detection_toggle_button_ =
      new QPushButton("Select New Detection Pixel");
  p_manipulation_set_detection_toggle_button_->setCheckable(
      true); // make it toggle
  p_manipulation_override_toggle_layout->addWidget(
      p_manipulation_set_detection_toggle_button_);

  // Create the layout for the approval robot id
  QHBoxLayout *p_manipulation_robot_id_combo_box_layout = new QHBoxLayout;
  p_manipulation_robot_id_combo_box_layout->addWidget(new QLabel("Robot ID"));
  p_manipulation_robot_id_combo_box_ = new QComboBox;
  p_manipulation_robot_id_combo_box_layout->addWidget(
      p_manipulation_robot_id_combo_box_);

  // Create the layout for displaying the selected pixels
  QHBoxLayout *p_manipulation_selected_pixels_view_layout = new QHBoxLayout;
  p_selected_pixels_label_ = new QLabel("Selected Pixels: N/A");
  p_manipulation_selected_pixels_view_layout->addWidget(
      p_selected_pixels_label_);

  // Create the layout for the publish/reject/reset buttons
  QHBoxLayout *p_manipulation_approval_layout = new QHBoxLayout;
  p_manipulation_publish_push_button_ = new QPushButton("Publish");
  p_manipulation_approval_layout->addWidget(
      p_manipulation_publish_push_button_);
  p_manipulation_reject_push_button_ = new QPushButton("Reject");
  p_manipulation_approval_layout->addWidget(p_manipulation_reject_push_button_);
  p_manipulation_reset_push_button_ = new QPushButton("Reset");
  p_manipulation_approval_layout->addWidget(p_manipulation_reset_push_button_);

  // Organize the layouts vertically
  QVBoxLayout *p_layout = new QVBoxLayout;
  p_layout->addLayout(p_manipulation_request_layout);
  p_layout->addLayout(p_manipulation_robot_id_combo_box_layout);
  p_layout->addLayout(p_manipulation_selected_pixels_view_layout);
  p_layout->addLayout(p_manipulation_override_toggle_layout);
  p_layout->addLayout(p_manipulation_approval_layout);
  setLayout(p_layout);
  setMinimumSize(0, 0);

  // Create a timer to regularly publish to the system monitor
  QTimer *p_timer_ = new QTimer(this);

  // Create Qt connections
  QObject::connect(p_manipulation_image_label_,
                   &ScaledClickableLabel::pixelClicked, this,
                   &ManipulationApprovalPanel::handlePixelClick);
  QObject::connect(p_prev_image_button_, &QPushButton::clicked, this,
                   &ManipulationApprovalPanel::showPrevImage);
  QObject::connect(p_next_image_button_, &QPushButton::clicked, this,
                   &ManipulationApprovalPanel::showNextImage);
  QObject::connect(p_manipulation_publish_push_button_, &QPushButton::clicked,
                   this,
                   &ManipulationApprovalPanel::publishManipulationApproval);
  QObject::connect(p_manipulation_reject_push_button_, &QPushButton::clicked,
                   this,
                   &ManipulationApprovalPanel::publishManipulationRejection);
  QObject::connect(p_manipulation_reset_push_button_, &QPushButton::clicked,
                   this, &ManipulationApprovalPanel::resetSelection);
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
        "/" + robot_id + "/spot_executor_node/manipulation_request";
    manipulation_request_subscriptions_.emplace(
        robot_id, node->create_subscription<
                      nlu_interface_rviz::msg::ManipulationApprovalRequest>(
                      request_topic, rclcpp::QoS(10).transient_local(),
                      [this, robot_id](
                          nlu_interface_rviz::msg::ManipulationApprovalRequest::
                              ConstSharedPtr msg) {
                        this->handleManipulationRequest(msg, robot_id);
                      }));
    auto const approval_topic =
        "/" + robot_id + "/spot_executor_node/pick_confirmation";
    manipulation_approval_publishers_.emplace(
        robot_id, node->create_publisher<
                      nlu_interface_rviz::msg::ManipulationApprovalResponse>(
                      approval_topic, 1));
  }
  return;
}

void ManipulationApprovalPanel::handleManipulationRequest(
    nlu_interface_rviz::msg::ManipulationApprovalRequest::ConstSharedPtr msg,
    std::string const &robot_id) {

  // Set the combo box to the robot id
  p_manipulation_robot_id_combo_box_->setCurrentText(robot_id.c_str());

  // Convert all images to pixmaps
  candidate_pixmaps_.clear();
  for (auto const &image_msg : msg->images) {
    auto img = std::make_shared<sensor_msgs::msg::Image>(image_msg);
    candidate_pixmaps_.push_back(image_msg_to_pixmap(img));
  }

  // On first message, resize the panel to a reasonable default
  if (!received_first_message_) {
    received_first_message_ = true;
    resize(480, 600);
  }

  // Store detection metadata and reset selection to detection
  has_detection_ = msg->has_detection;
  detection_image_index_ = static_cast<int>(msg->detection_image_index);
  detection_x_ = msg->image_x;
  detection_y_ = msg->image_y;
  if (has_detection_) {
    selected_image_index_ = detection_image_index_;
    selected_pixel_ = std::make_pair(detection_x_, detection_y_);
    current_image_index_ = detection_image_index_;
  } else {
    selected_image_index_ = 0;
    selected_pixel_ = std::nullopt;
    current_image_index_ = 0;
  }

  showCurrentImage();
  return;
}

void ManipulationApprovalPanel::showCurrentImage() {
  if (candidate_pixmaps_.empty())
    return;

  // Start from the clean base pixmap
  p_base_pixmap_ = new QPixmap(candidate_pixmaps_[current_image_index_]);

  // Draw selection marker whenever viewing the selected image
  if (current_image_index_ == selected_image_index_ &&
      selected_pixel_.has_value()) {
    auto const [sx, sy] = selected_pixel_.value();
    QPixmap annotated = *p_base_pixmap_;
    QPainter painter(&annotated);
    painter.setPen(QPen(Qt::red, 3));
    int r = 15;
    painter.drawEllipse(QPoint(sx, sy), r, r);
    painter.drawLine(sx - r, sy, sx + r, sy);
    painter.drawLine(sx, sy - r, sx, sy + r);
    painter.end();
    p_manipulation_image_label_->setOriginalPixmap(annotated);
  } else {
    p_manipulation_image_label_->setOriginalPixmap(*p_base_pixmap_);
  }

  // Always show current selection
  if (selected_pixel_.has_value()) {
    std::ostringstream oss;
    oss << "Selected Pixels: (" << selected_pixel_->first << ", "
        << selected_pixel_->second << ") [Image " << (selected_image_index_ + 1)
        << "]";
    p_selected_pixels_label_->setText(oss.str().c_str());
  } else {
    p_selected_pixels_label_->setText("Selected Pixels: N/A");
  }

  // Update image index label
  std::ostringstream idx_oss;
  idx_oss << "Image " << (current_image_index_ + 1) << " / "
          << candidate_pixmaps_.size();
  p_image_index_label_->setText(idx_oss.str().c_str());

  // Green when viewing the selected image
  if (current_image_index_ == selected_image_index_) {
    p_image_index_label_->setStyleSheet(
        "background-color: green; color: white; font-weight: bold;");
  } else {
    p_image_index_label_->setStyleSheet("");
  }

  return;
}

void ManipulationApprovalPanel::showPrevImage() {
  if (candidate_pixmaps_.empty())
    return;
  current_image_index_ =
      (current_image_index_ - 1 + static_cast<int>(candidate_pixmaps_.size())) %
      static_cast<int>(candidate_pixmaps_.size());
  showCurrentImage();
  return;
}

void ManipulationApprovalPanel::showNextImage() {
  if (candidate_pixmaps_.empty())
    return;
  current_image_index_ =
      (current_image_index_ + 1) % static_cast<int>(candidate_pixmaps_.size());
  showCurrentImage();
  return;
}

void ManipulationApprovalPanel::handlePixelClick(int x, int y) {
  {
    std::ostringstream oss;
    oss << "Clicked pixel: (" << x << "," << y << "). Pixel Override State: "
        << p_manipulation_set_detection_toggle_button_->isChecked();
    RCLCPP_INFO(rclcpp::get_logger("rviz_panel"), "%s", oss.str().c_str());
  }
  if (p_manipulation_set_detection_toggle_button_->isChecked()) {
    selected_image_index_ = current_image_index_;
    selected_pixel_ = std::pair<int, int>(x, y);
    p_manipulation_set_detection_toggle_button_->setChecked(false);

    if (p_base_pixmap_ != nullptr) {
      QPixmap annotated = *p_base_pixmap_;
      QPainter painter(&annotated);
      painter.setPen(QPen(Qt::red, 3));
      int r = 15;
      painter.drawEllipse(QPoint(x, y), r, r);
      painter.drawLine(x - r, y, x + r, y);
      painter.drawLine(x, y - r, x, y + r);
      painter.end();
      p_manipulation_image_label_->setOriginalPixmap(annotated);
    }

    showCurrentImage();
  }
  {
    std::ostringstream oss;
    oss << "Updated selected_pixel_: ("
        << (selected_pixel_ ? selected_pixel_->first : -1) << ", "
        << (selected_pixel_ ? selected_pixel_->second : -1) << ") on image "
        << selected_image_index_ << std::endl;
    RCLCPP_INFO(rclcpp::get_logger("rviz_panel"), "%s", oss.str().c_str());
  }
  return;
}

void ManipulationApprovalPanel::publishManipulationApproval(void) {
  auto msg = nlu_interface_rviz::msg::ManipulationApprovalResponse();
  msg.approve = true;
  msg.image_index = static_cast<uint32_t>(selected_image_index_);
  msg.image_x = selected_pixel_ ? selected_pixel_->first : 0;
  msg.image_y = selected_pixel_ ? selected_pixel_->second : 0;
  publishManipulationResponse(msg);
  return;
}

void ManipulationApprovalPanel::publishManipulationRejection(void) {
  auto msg = nlu_interface_rviz::msg::ManipulationApprovalResponse();
  msg.approve = false;
  msg.image_index = 0;
  msg.image_x = 0;
  msg.image_y = 0;
  publishManipulationResponse(msg);
  return;
}

void ManipulationApprovalPanel::resetSelection(void) {
  if (has_detection_) {
    selected_image_index_ = detection_image_index_;
    selected_pixel_ = std::make_pair(detection_x_, detection_y_);
    current_image_index_ = selected_image_index_;
  } else {
    selected_image_index_ = 0;
    selected_pixel_ = std::nullopt;
    current_image_index_ = 0;
  }
  showCurrentImage();
  return;
}

void ManipulationApprovalPanel::publishManipulationResponse(
    nlu_interface_rviz::msg::ManipulationApprovalResponse const &msg) {
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
  msg.notes = "All I do is Pick-n-Place";
  system_monitor_publisher_->publish(msg);
  return;
}

} // namespace nlu_interface_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(nlu_interface_rviz::ManipulationApprovalPanel,
                       rviz_common::Panel)
