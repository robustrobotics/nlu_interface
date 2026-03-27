#include <QImage>
#include <QVBoxLayout>
#include <cv_bridge/cv_bridge.hpp>
#include <nlu_interface_rviz/instruction_panel.hpp>
#include <rviz_common/display_context.hpp>

namespace nlu_interface_rviz {
InstructionPanel::InstructionPanel(QWidget *parent) : Panel(parent) {
  // Create the layout for the domain type combo box
  QHBoxLayout *p_domain_type_combo_box_layout = new QHBoxLayout;
  p_domain_type_combo_box_layout->addWidget(new QLabel("Domain Type"));
  p_domain_type_combo_box_ = new QComboBox;
  p_domain_type_combo_box_->addItems({"default", "Pddl", "goto_points"});
  p_domain_type_combo_box_layout->addWidget(p_domain_type_combo_box_);

  // Create the layout for the robot id combo box
  QHBoxLayout *p_robot_id_combo_box_layout = new QHBoxLayout;
  p_robot_id_combo_box_layout->addWidget(new QLabel("Robot ID"));
  p_robot_id_combo_box_ = new QComboBox;
  p_robot_id_combo_box_layout->addWidget(p_robot_id_combo_box_);

  // Create the layout for the instruction field
  QHBoxLayout *p_instruction_layout = new QHBoxLayout;
  p_instruction_layout->addWidget(new QLabel("Instruction:"));
  p_instruction_editor_ = new QLineEdit;
  p_instruction_layout->addWidget(p_instruction_editor_);

  // Create the layout for a display of the previously published instruction
  QHBoxLayout *p_prev_instruction_layout = new QHBoxLayout;
  p_prev_instruction_layout->addWidget(new QLabel("Published Instruction:"));
  p_prev_instruction_editor_ = new QLineEdit;
  p_prev_instruction_editor_->setReadOnly(true);
  p_prev_instruction_layout->addWidget(p_prev_instruction_editor_);

  // Create the layout for a display of the response from the LLM
  QHBoxLayout *p_llm_response_layout = new QHBoxLayout;
  p_llm_response_layout->addWidget(new QLabel("LLM Response"));
  p_llm_response_textbox_ = new QTextEdit;
  p_llm_response_textbox_->setReadOnly(true);
  p_llm_response_layout->addWidget(p_llm_response_textbox_);

  // Organize the layouts vertically
  QVBoxLayout *p_layout = new QVBoxLayout;
  p_layout->addLayout(p_domain_type_combo_box_layout);
  p_layout->addLayout(p_robot_id_combo_box_layout);
  p_layout->addLayout(p_instruction_layout);
  p_layout->addLayout(p_prev_instruction_layout);
  p_layout->addLayout(p_llm_response_layout);
  setLayout(p_layout);

  // Create a timer to regularly publish to the system monitor
  QTimer *p_timer_ = new QTimer(this);

  // Create Qt connections
  QObject::connect(p_instruction_editor_, SIGNAL(returnPressed()), this,
                   SLOT(publishInstruction()));
  QObject::connect(p_timer_, &QTimer::timeout, this,
                   &InstructionPanel::publishSystemMonitor);
  p_timer_->start(500);
  return;
}

void InstructionPanel::onInitialize() {
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

  p_robot_id_combo_box_->addItems(
      QList<QString>(robot_ids_.begin(), robot_ids_.end()));

  // Create the publishers
  instruction_publisher_ =
      node->create_publisher<omniplanner_msgs::msg::LanguageGoalMsg>(
          "omniplanner_node/language_planner/language_goal", 10);
  system_monitor_publisher_ =
      node->create_publisher<ros_system_monitor_msgs::msg::NodeInfoMsg>(
          "~/node_status", 1);
  // Create the subscriptions
  llm_response_subscription_ = node->create_subscription<std_msgs::msg::String>(
      "~/llm_response", 10, [this](std_msgs::msg::String::ConstSharedPtr msg) {
        this->handleLLMResponse(msg);
      });
  return;
}

void InstructionPanel::handleLLMResponse(
    std_msgs::msg::String::ConstSharedPtr msg) {
  p_llm_response_textbox_->setText(msg->data.c_str());
  return;
}

void InstructionPanel::publishInstruction(void) {
  // Construct and publish the instruction text as a String message
  auto msg = omniplanner_msgs::msg::LanguageGoalMsg();
  msg.robot_id = p_robot_id_combo_box_->currentText().toStdString();
  msg.command = p_instruction_editor_->text().toStdString();
  msg.domain_type = p_domain_type_combo_box_->currentText().toStdString();
  instruction_publisher_->publish(msg);

  // Update the display for the previous instruction
  p_prev_instruction_editor_->setText(p_instruction_editor_->text());
  return;
}

void InstructionPanel::publishSystemMonitor(void) {
  // Get the ROS2 node
  auto p_node = p_node_abstraction_->get_raw_node();

  // Construct the message
  auto msg = ros_system_monitor_msgs::msg::NodeInfoMsg();
  msg.nickname = "nlu_rviz_instruction_panel";
  msg.node_name = p_node->get_fully_qualified_name();
  msg.status = ros_system_monitor_msgs::msg::NodeInfoMsg::NOMINAL;
  msg.notes = "You gonna give me orders?";
  system_monitor_publisher_->publish(msg);
  return;
}

} // namespace nlu_interface_rviz

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(nlu_interface_rviz::InstructionPanel, rviz_common::Panel)
