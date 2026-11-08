#include <QImage>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <cv_bridge/cv_bridge.hpp>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
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

// Depth overlay colour scale, matching the web console: warm (near) to cool
// (far) over this range; pixels without a measurement stay transparent.
constexpr double kDepthNearM = 0.3;
constexpr double kDepthFarM = 6.0;

QPixmap bgr_mat_to_pixmap(cv::Mat const &bgr) {
  QImage image(bgr.data, bgr.cols, bgr.rows, static_cast<int>(bgr.step),
               QImage::Format_BGR888);
  return QPixmap::fromImage(image.copy());
}

bool depth_valid(uint16_t value) { return value != 0 && value != 65535; }

QPixmap depth_overlay_pixmap(cv::Mat const &depth, double scale) {
  static const double stops[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
  static const double colors[5][3] = {{255, 48, 48},
                                      {255, 160, 0},
                                      {235, 235, 70},
                                      {70, 200, 255},
                                      {70, 80, 235}};
  QImage overlay(depth.cols, depth.rows, QImage::Format_RGBA8888);
  overlay.fill(Qt::transparent);
  for (int row = 0; row < depth.rows; ++row) {
    auto const *values = depth.ptr<uint16_t>(row);
    auto *out = overlay.scanLine(row);
    for (int col = 0; col < depth.cols; ++col) {
      if (!depth_valid(values[col]))
        continue;
      double t = (values[col] / scale - kDepthNearM) / (kDepthFarM - kDepthNearM);
      t = std::clamp(t, 0.0, 1.0);
      int k = std::min(3, static_cast<int>(t / 0.25));
      double f = (t - stops[k]) / (stops[k + 1] - stops[k]);
      for (int c = 0; c < 3; ++c) {
        out[4 * col + c] = static_cast<uchar>(colors[k][c] +
                                              f * (colors[k + 1][c] - colors[k][c]));
      }
      out[4 * col + 3] = 255;
    }
  }
  return QPixmap::fromImage(overlay);
}

// Row-major rotation from a (w, x, y, z) quaternion; false if degenerate.
bool quaternion_rotation(double w, double x, double y, double z,
                         std::array<double, 9> &r) {
  double n = std::sqrt(w * w + x * x + y * y + z * z);
  if (n < 0.5 || !std::isfinite(n))
    return false;
  w /= n; x /= n; y /= n; z /= n;
  r = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
       2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
       2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)};
  return true;
}

std::string fixed2(double value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << value;
  return oss.str();
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

  // Create the layout for the override toggle buttons. The selected pixel is
  // either the grasp point (Publish) or the ground/object point to walk
  // toward (Walk To). The stand-point toggle is the optional heading control
  // and stays hidden unless enabled by parameter.
  QHBoxLayout *p_manipulation_override_toggle_layout = new QHBoxLayout;
  p_manipulation_set_detection_toggle_button_ =
      new QPushButton("Select Pixel (grasp / walk-to target)");
  p_manipulation_set_detection_toggle_button_->setCheckable(
      true); // make it toggle
  p_manipulation_override_toggle_layout->addWidget(
      p_manipulation_set_detection_toggle_button_);
  p_manipulation_set_stand_point_toggle_button_ =
      new QPushButton("Select Stand Point (heading)");
  p_manipulation_set_stand_point_toggle_button_->setCheckable(true);
  p_manipulation_set_stand_point_toggle_button_->setToolTip(
      "Walk To only: click the ground pixel Spot should stand on. It will "
      "face the selected target pixel from there.");
  p_manipulation_set_stand_point_toggle_button_->setVisible(false);
  p_manipulation_override_toggle_layout->addWidget(
      p_manipulation_set_stand_point_toggle_button_);

  // Create the layout for the approval robot id
  QHBoxLayout *p_manipulation_robot_id_layout = new QHBoxLayout;
  p_manipulation_robot_id_layout->addWidget(new QLabel("Robot ID:"));
  p_manipulation_robot_id_label_ = new QLabel("N/A");
  p_manipulation_robot_id_layout->addWidget(p_manipulation_robot_id_label_);

  // Create the layout for displaying the selected pixels
  QHBoxLayout *p_manipulation_selected_pixels_view_layout = new QHBoxLayout;
  p_selected_pixels_label_ = new QLabel("Selected Pixels: N/A");
  p_manipulation_selected_pixels_view_layout->addWidget(
      p_selected_pixels_label_);

  // Create the layout for the status line
  QHBoxLayout *p_status_layout = new QHBoxLayout;
  p_status_label_ = new QLabel("Waiting for a manipulation request.");
  p_status_label_->setWordWrap(true);
  p_status_layout->addWidget(p_status_label_);

  // Create the layout for the publish/walk-to/reject/reset buttons
  QHBoxLayout *p_manipulation_approval_layout = new QHBoxLayout;
  p_manipulation_publish_push_button_ = new QPushButton("Publish");
  p_manipulation_publish_push_button_->setToolTip(
      "Approve: grasp at the selected pixel.");
  p_manipulation_approval_layout->addWidget(
      p_manipulation_publish_push_button_);
  p_manipulation_walk_to_push_button_ = new QPushButton("Walk To");
  p_manipulation_walk_to_push_button_->setToolTip(
      "Do not grasp yet: walk toward the selected ground/object pixel, then "
      "take new images and ask again.");
  p_manipulation_approval_layout->addWidget(
      p_manipulation_walk_to_push_button_);
  p_manipulation_reject_push_button_ = new QPushButton("Reject");
  p_manipulation_approval_layout->addWidget(p_manipulation_reject_push_button_);
  p_manipulation_reset_push_button_ = new QPushButton("Reset");
  p_manipulation_approval_layout->addWidget(p_manipulation_reset_push_button_);

  // Depth / walk-estimate readout under the image
  QHBoxLayout *p_depth_layout = new QHBoxLayout;
  p_show_depth_checkbox_ = new QCheckBox("Show depth");
  p_show_depth_checkbox_->setChecked(true);
  p_show_depth_checkbox_->setToolTip(
      "Overlay the depth registered to this camera: red near (0.3 m) to blue "
      "far (6 m). Display only; Walk To does not use it.");
  p_depth_layout->addWidget(p_show_depth_checkbox_);
  p_cursor_label_ = new QLabel("Hover the image for distances");
  p_cursor_label_->setWordWrap(true);
  p_depth_layout->addWidget(p_cursor_label_, 1);
  p_estimate_warning_label_ = new QLabel;
  p_estimate_warning_label_->setWordWrap(true);
  p_estimate_warning_label_->setStyleSheet("color: #ff6b5b; font-weight: bold;");
  p_estimate_warning_label_->setVisible(false);
  p_walk_note_label_ = new QLabel;
  p_walk_note_label_->setWordWrap(true);
  p_walk_note_label_->setStyleSheet("color: gray;");
  p_walk_note_label_->setVisible(false);

  // Organize the layouts vertically
  QVBoxLayout *p_layout = new QVBoxLayout;
  p_layout->addLayout(p_manipulation_request_layout);
  p_layout->addLayout(p_depth_layout);
  p_layout->addWidget(p_estimate_warning_label_);
  p_layout->addWidget(p_walk_note_label_);
  p_layout->addLayout(p_manipulation_robot_id_layout);
  p_layout->addLayout(p_manipulation_selected_pixels_view_layout);
  p_layout->addLayout(p_status_layout);
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
  p_manipulation_image_label_->setMouseTracking(true);
  QObject::connect(p_manipulation_image_label_,
                   &ScaledClickableLabel::pixelHovered, this,
                   &ManipulationApprovalPanel::handlePixelHover);
  QObject::connect(p_manipulation_image_label_, &ScaledClickableLabel::hoverLeft,
                   this, &ManipulationApprovalPanel::handleHoverLeft);
  QObject::connect(p_show_depth_checkbox_, &QCheckBox::toggled, this,
                   [this](bool) { showCurrentImage(); });
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
  QObject::connect(p_manipulation_walk_to_push_button_, &QPushButton::clicked,
                   this, &ManipulationApprovalPanel::publishWalkTo);
  QObject::connect(p_manipulation_reset_push_button_, &QPushButton::clicked,
                   this, &ManipulationApprovalPanel::resetSelection);
  // The two pixel-selection toggles are mutually exclusive.
  QObject::connect(p_manipulation_set_detection_toggle_button_,
                   &QPushButton::toggled, this, [this](bool checked) {
                     if (checked)
                       p_manipulation_set_stand_point_toggle_button_->setChecked(
                           false);
                   });
  QObject::connect(p_manipulation_set_stand_point_toggle_button_,
                   &QPushButton::toggled, this, [this](bool checked) {
                     if (checked)
                       p_manipulation_set_detection_toggle_button_->setChecked(
                           false);
                   });
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

  // Heading control for Walk To is a feature flag, off by default.
  if (!node->has_parameter("manipulation_walk_to_heading_selection")) {
    node->declare_parameter<bool>("manipulation_walk_to_heading_selection",
                                  false);
  }
  heading_selection_enabled_ =
      node->get_parameter("manipulation_walk_to_heading_selection").as_bool();
  p_manipulation_set_stand_point_toggle_button_->setVisible(
      heading_selection_enabled_);

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

  // Set the robot id label
  p_manipulation_robot_id_label_->setText(robot_id.c_str());

  // Convert all images to pixmaps: raw images, or JPEGs from the compressed
  // transport (the executor's default).
  candidate_pixmaps_.clear();
  for (auto const &image_msg : msg->images) {
    auto img = std::make_shared<sensor_msgs::msg::Image>(image_msg);
    candidate_pixmaps_.push_back(image_msg_to_pixmap(img));
  }
  if (msg->images.empty()) {
    for (auto const &compressed : msg->compressed_images) {
      cv::Mat bgr = cv::imdecode(compressed.data, cv::IMREAD_COLOR);
      candidate_pixmaps_.push_back(bgr.empty() ? QPixmap() : bgr_mat_to_pixmap(bgr));
    }
  }
  int const count = static_cast<int>(candidate_pixmaps_.size());

  // Depth registered to each candidate (16-bit PNG), for display only.
  depth_images_.assign(count, cv::Mat());
  depth_scales_.assign(count, 0.0);
  depth_overlays_.assign(count, QPixmap());
  for (int i = 0; i < count && i < static_cast<int>(msg->depth_images.size()) &&
                  i < static_cast<int>(msg->depth_scales.size());
       ++i) {
    if (msg->depth_images[i].data.empty() || !(msg->depth_scales[i] > 0))
      continue;
    cv::Mat depth = cv::imdecode(msg->depth_images[i].data, cv::IMREAD_UNCHANGED);
    if (depth.empty() || depth.type() != CV_16UC1)
      continue;
    depth_images_[i] = depth;
    depth_scales_[i] = msg->depth_scales[i];
    depth_overlays_[i] = depth_overlay_pixmap(depth, msg->depth_scales[i]);
  }

  // Geometry for the walk-to ground-plane estimate (same math as the robot).
  cameras_.assign(count, CameraGeometry());
  for (int i = 0; i < count; ++i) {
    if (msg->camera_intrinsics.size() < 4u * (i + 1) ||
        msg->camera_poses.size() < 7u * (i + 1))
      break;
    auto const *k = &msg->camera_intrinsics[4 * i];
    auto const *pose = &msg->camera_poses[7 * i];
    CameraGeometry camera;
    if (k[0] <= 0 || k[1] <= 0 ||
        !quaternion_rotation(pose[3], pose[4], pose[5], pose[6], camera.rotation))
      continue;
    camera.valid = true;
    camera.fx = k[0]; camera.fy = k[1]; camera.cx = k[2]; camera.cy = k[3];
    camera.origin = {pose[0], pose[1], pose[2]};
    cameras_[i] = camera;
  }
  has_ground_ = msg->ground_plane.size() == 6;
  if (has_ground_) {
    auto const &g = msg->ground_plane;
    double n = std::sqrt(g[3] * g[3] + g[4] * g[4] + g[5] * g[5]);
    has_ground_ = n > 0.5;
    ground_point_ = {g[0], g[1], g[2]};
    if (has_ground_)
      ground_normal_ = {g[3] / n, g[4] / n, g[5] / n};
  }
  has_robot_ = msg->robot_pose.size() == 3;
  if (has_robot_) {
    robot_x_ = msg->robot_pose[0];
    robot_y_ = msg->robot_pose[1];
  }
  walk_standoff_ = msg->walk_to_standoff;
  walk_max_distance_ = msg->walk_to_max_distance;
  p_cursor_label_->setText("Hover the image for distances");

  // Store image source names
  image_source_names_.clear();
  for (auto const &name : msg->image_source_names) {
    image_source_names_.push_back(name);
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
  // A new request (e.g. after a walk-to) always starts without a stand point.
  stand_pixel_ = std::nullopt;
  stand_image_index_ = 0;
  p_manipulation_set_detection_toggle_button_->setChecked(false);
  p_manipulation_set_stand_point_toggle_button_->setChecked(false);
  p_status_label_->setText(
      has_detection_
          ? "New request: review the detection, then Publish, Walk To or Reject."
          : "New request: no detection. Select a pixel, then Publish, Walk To "
            "or Reject.");

  showCurrentImage();
  return;
}

void ManipulationApprovalPanel::showCurrentImage() {
  if (candidate_pixmaps_.empty())
    return;

  // Start from the clean base pixmap
  delete p_base_pixmap_;
  p_base_pixmap_ = new QPixmap(candidate_pixmaps_[current_image_index_]);

  // Draw the depth overlay (display only), the selection (red crosshair) and,
  // if set, the stand point (blue square) whenever viewing their image.
  QPixmap annotated = *p_base_pixmap_;
  bool drew = false;
  QPainter painter(&annotated);
  bool const has_depth =
      current_image_index_ < static_cast<int>(depth_overlays_.size()) &&
      !depth_overlays_[current_image_index_].isNull();
  p_show_depth_checkbox_->setEnabled(has_depth);
  if (has_depth && p_show_depth_checkbox_->isChecked()) {
    painter.setOpacity(0.5);
    painter.drawPixmap(annotated.rect(), depth_overlays_[current_image_index_]);
    painter.setOpacity(1.0);
    drew = true;
  }
  if (current_image_index_ == selected_image_index_ &&
      selected_pixel_.has_value()) {
    auto const [sx, sy] = selected_pixel_.value();
    painter.setPen(QPen(Qt::red, 3));
    int r = 15;
    painter.drawEllipse(QPoint(sx, sy), r, r);
    painter.drawLine(sx - r, sy, sx + r, sy);
    painter.drawLine(sx, sy - r, sx, sy + r);
    drew = true;
  }
  if (current_image_index_ == stand_image_index_ && stand_pixel_.has_value()) {
    auto const [px, py] = stand_pixel_.value();
    painter.setPen(QPen(Qt::blue, 3));
    int r = 12;
    painter.drawRect(px - r, py - r, 2 * r, 2 * r);
    if (current_image_index_ == selected_image_index_ &&
        selected_pixel_.has_value()) {
      painter.drawLine(px, py, selected_pixel_->first, selected_pixel_->second);
    }
    drew = true;
  }
  painter.end();
  p_manipulation_image_label_->setOriginalPixmap(drew ? annotated
                                                      : *p_base_pixmap_);

  // Always show current selection
  if (selected_pixel_.has_value()) {
    std::ostringstream oss;
    oss << "Selected Pixels: (" << selected_pixel_->first << ", "
        << selected_pixel_->second << ") [";
    if (selected_image_index_ < static_cast<int>(image_source_names_.size())) {
      oss << image_source_names_[selected_image_index_];
    } else {
      oss << "Image " << (selected_image_index_ + 1);
    }
    oss << "]";
    if (stand_pixel_.has_value()) {
      oss << "  Stand: (" << stand_pixel_->first << ", " << stand_pixel_->second
          << ")";
      if (stand_image_index_ != selected_image_index_) {
        oss << " [other image: ignored]";
      }
    }
    auto const point = describePoint(selected_image_index_, selected_pixel_->first,
                                     selected_pixel_->second);
    if (!point.empty())
      oss << "\n" << point;
    p_selected_pixels_label_->setText(oss.str().c_str());
  } else {
    p_selected_pixels_label_->setText("Selected Pixels: N/A");
  }

  // Update image index label
  std::ostringstream idx_oss;
  if (current_image_index_ < static_cast<int>(image_source_names_.size())) {
    idx_oss << image_source_names_[current_image_index_] << " ("
            << (current_image_index_ + 1) << " / " << candidate_pixmaps_.size()
            << ")";
  } else {
    idx_oss << "Image " << (current_image_index_ + 1) << " / "
            << candidate_pixmaps_.size();
  }
  p_image_index_label_->setText(idx_oss.str().c_str());

  // Hand-camera warning and the reminder of what Walk To actually uses.
  bool const estimate_here =
      current_image_index_ < static_cast<int>(cameras_.size()) &&
      cameras_[current_image_index_].valid && has_ground_ && has_robot_;
  p_estimate_warning_label_->setVisible(estimate_here &&
                                        estimateUnreliable(current_image_index_));
  p_estimate_warning_label_->setText(
      "Hand camera: walk distances from this camera are unreliable. Pick "
      "walk-to points in a body camera.");
  p_walk_note_label_->setVisible(estimate_here);
  p_walk_note_label_->setText(QString::fromStdString(
      "Walk To plans only from the ground-plane estimate (assumes the point is "
      "on the floor; unreliable near the horizon). Depth is for reference and "
      "is not used. The body center stops " +
      fixed2(walk_standoff_).substr(0, fixed2(walk_standoff_).size() - 1) +
      " m short and moves at most " +
      fixed2(walk_max_distance_).substr(0, fixed2(walk_max_distance_).size() - 1) +
      " m per walk."));
  updateWalkButtonLabel();

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
    showCurrentImage();
  } else if (p_manipulation_set_stand_point_toggle_button_->isChecked()) {
    stand_image_index_ = current_image_index_;
    stand_pixel_ = std::pair<int, int>(x, y);
    p_manipulation_set_stand_point_toggle_button_->setChecked(false);
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
  if (!selected_pixel_.has_value()) {
    p_status_label_->setText(
        "Publish needs a grasp pixel: select one or Reject.");
    return;
  }
  auto msg = nlu_interface_rviz::msg::ManipulationApprovalResponse();
  msg.approve = true;
  msg.walk_to = false;
  msg.image_index = static_cast<uint32_t>(selected_image_index_);
  msg.image_x = selected_pixel_->first;
  msg.image_y = selected_pixel_->second;
  publishManipulationResponse(msg);
  p_status_label_->setText("Approval sent; the robot is grasping.");
  return;
}

void ManipulationApprovalPanel::publishManipulationRejection(void) {
  auto msg = nlu_interface_rviz::msg::ManipulationApprovalResponse();
  msg.approve = false;
  msg.walk_to = false;
  msg.image_index = 0;
  msg.image_x = 0;
  msg.image_y = 0;
  publishManipulationResponse(msg);
  p_status_label_->setText("Rejection sent; the grasp attempt is abandoned.");
  return;
}

void ManipulationApprovalPanel::publishWalkTo(void) {
  if (!selected_pixel_.has_value()) {
    p_status_label_->setText(
        "Walk To needs a target: select the ground/object pixel first.");
    return;
  }
  auto msg = nlu_interface_rviz::msg::ManipulationApprovalResponse();
  msg.approve = false;
  msg.walk_to = true;
  msg.image_index = static_cast<uint32_t>(selected_image_index_);
  msg.image_x = selected_pixel_->first;
  msg.image_y = selected_pixel_->second;
  msg.has_stand_point = false;
  if (heading_selection_enabled_ && stand_pixel_.has_value()) {
    if (stand_image_index_ == selected_image_index_) {
      msg.has_stand_point = true;
      msg.stand_x = stand_pixel_->first;
      msg.stand_y = stand_pixel_->second;
    } else {
      RCLCPP_WARN(rclcpp::get_logger("rviz_panel"),
                  "Stand point is on a different image than the target; "
                  "sending walk-to with the default standoff instead.");
    }
  }
  publishManipulationResponse(msg);
  p_status_label_->setText(
      msg.has_stand_point
          ? "Walk-to sent (with stand point); waiting for new images..."
          : "Walk-to sent; waiting for new images...");
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
  stand_pixel_ = std::nullopt;
  stand_image_index_ = 0;
  p_manipulation_set_detection_toggle_button_->setChecked(false);
  p_manipulation_set_stand_point_toggle_button_->setChecked(false);
  showCurrentImage();
  return;
}

void ManipulationApprovalPanel::publishManipulationResponse(
    nlu_interface_rviz::msg::ManipulationApprovalResponse const &msg) {
  // Get the publisher for the selected robot id
  auto it_publisher = manipulation_approval_publishers_.find(
      p_manipulation_robot_id_label_->text().toStdString());
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

namespace nlu_interface_rviz {

bool ManipulationApprovalPanel::estimateUnreliable(int index) const {
  return index >= 0 && index < static_cast<int>(image_source_names_.size()) &&
         image_source_names_[index].rfind("hand", 0) == 0;
}

// Reproduces spot_tools walk_to_utils.pixel_to_ground_point + plan_walk.
ManipulationApprovalPanel::GroundEstimate
ManipulationApprovalPanel::groundEstimate(int index, int x, int y) const {
  GroundEstimate estimate;
  if (index < 0 || index >= static_cast<int>(cameras_.size()) ||
      !cameras_[index].valid || !has_ground_ || !has_robot_)
    return estimate;
  estimate.available = true;
  auto const &c = cameras_[index];
  double ray_cam[3] = {(x - c.cx) / c.fx, (y - c.cy) / c.fy, 1.0};
  double norm = std::sqrt(ray_cam[0] * ray_cam[0] + ray_cam[1] * ray_cam[1] + 1.0);
  double ray[3];
  for (int r = 0; r < 3; ++r) {
    ray[r] = (c.rotation[3 * r] * ray_cam[0] + c.rotation[3 * r + 1] * ray_cam[1] +
              c.rotation[3 * r + 2] * ray_cam[2]) /
             norm;
  }
  double denominator = 0, numerator = 0;
  for (int r = 0; r < 3; ++r) {
    denominator += ray[r] * ground_normal_[r];
    numerator += (ground_point_[r] - c.origin[r]) * ground_normal_[r];
  }
  if (std::abs(denominator) < 1e-6)
    return estimate;
  double t = numerator / denominator;
  if (t < 0.2 || t > 15.0)
    return estimate;
  estimate.hit = true;
  double hx = c.origin[0] + t * ray[0], hy = c.origin[1] + t * ray[1];
  estimate.distance = std::hypot(hx - robot_x_, hy - robot_y_);
  if (estimate.distance - walk_standoff_ >= 0.05) {
    estimate.walk = estimate.distance - walk_standoff_;
    if (walk_max_distance_ > 0 && estimate.walk > walk_max_distance_) {
      estimate.walk = walk_max_distance_;
      estimate.capped = true;
    }
  }
  return estimate;
}

std::optional<double> ManipulationApprovalPanel::depthAt(int index, int x,
                                                         int y) const {
  if (index < 0 || index >= static_cast<int>(depth_images_.size()) ||
      depth_images_[index].empty() ||
      index >= static_cast<int>(candidate_pixmaps_.size()))
    return std::nullopt;
  auto const &depth = depth_images_[index];
  auto const &image = candidate_pixmaps_[index];
  int dx = std::clamp(static_cast<int>((x + 0.5) * depth.cols / image.width()), 0,
                      depth.cols - 1);
  int dy = std::clamp(static_cast<int>((y + 0.5) * depth.rows / image.height()), 0,
                      depth.rows - 1);
  std::vector<uint16_t> values;
  for (int row = std::max(0, dy - 2); row <= std::min(depth.rows - 1, dy + 2); ++row)
    for (int col = std::max(0, dx - 2); col <= std::min(depth.cols - 1, dx + 2); ++col)
      if (depth_valid(depth.at<uint16_t>(row, col)))
        values.push_back(depth.at<uint16_t>(row, col));
  if (values.empty())
    return std::nullopt;
  std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
  return values[values.size() / 2] / depth_scales_[index];
}

std::string ManipulationApprovalPanel::describePoint(int index, int x,
                                                     int y) const {
  std::vector<std::string> parts;
  auto const estimate = groundEstimate(index, x, y);
  if (estimate.available) {
    std::string text = "Walk To (ground plane): ";
    if (!estimate.hit) {
      text += "ray misses the floor";
    } else {
      text += fixed2(estimate.distance) + " m away -> ";
      text += estimate.walk > 0
                  ? "walks " + fixed2(estimate.walk) + " m" +
                        (estimate.capped ? " (capped)" : "")
                  : std::string("turns only");
    }
    if (estimateUnreliable(index))
      text += " (unreliable for this camera)";
    parts.push_back(text);
  }
  if (index >= 0 && index < static_cast<int>(depth_images_.size()) &&
      !depth_images_[index].empty()) {
    auto const depth = depthAt(index, x, y);
    parts.push_back(depth ? "measured depth " + fixed2(*depth) + " m (display only)"
                          : std::string("measured depth: none"));
  }
  std::string joined;
  for (auto const &part : parts)
    joined += (joined.empty() ? "" : " | ") + part;
  return joined;
}

void ManipulationApprovalPanel::updateWalkButtonLabel() {
  QString label = "Walk To";
  if (selected_pixel_.has_value()) {
    auto const estimate = groundEstimate(selected_image_index_, selected_pixel_->first,
                                         selected_pixel_->second);
    if (estimate.available) {
      std::string detail = !estimate.hit ? "no floor hit"
                           : estimate.walk > 0
                               ? fixed2(estimate.walk).substr(0, fixed2(estimate.walk).size() - 1) +
                                     " m, ground plane"
                               : std::string("turn only, ground plane");
      if (estimate.hit && estimateUnreliable(selected_image_index_))
        detail += ", unreliable";
      label += QString::fromStdString(" (" + detail + ")");
    }
  }
  p_manipulation_walk_to_push_button_->setText(label);
}

void ManipulationApprovalPanel::handlePixelHover(int x, int y) {
  auto const text = describePoint(current_image_index_, x, y);
  p_cursor_label_->setText(
      text.empty() ? QString("No depth or walk estimate for this image")
                   : QString::fromStdString("Cursor (" + std::to_string(x) + ", " +
                                            std::to_string(y) + "): " + text));
}

void ManipulationApprovalPanel::handleHoverLeft(void) {
  p_cursor_label_->setText("Hover the image for distances");
}

} // namespace nlu_interface_rviz
