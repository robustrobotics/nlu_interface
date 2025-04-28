#include <nlu_interface/instruction_panel.hpp>
#include <QVBoxLayout>
#include <rviz_common/display_context.hpp>

namespace nlu_interface {
InstructionPanel::InstructionPanel(QWidget* parent) : Panel(parent) {
  // Create the layout for the instruction field
  QHBoxLayout * p_instruction_layout = new QHBoxLayout;
  p_instruction_layout->addWidget( new QLabel("Instruction:"));
  p_instruction_editor_ = new QLineEdit;
  p_instruction_layout->addWidget( p_instruction_editor_ );

  // Create the layout for a display of the previously published instruction
  QHBoxLayout * p_prev_instruction_layout = new QHBoxLayout;
  p_prev_instruction_layout->addWidget( new QLabel("Published Instruction:"));
  p_prev_instruction_editor_ = new QLineEdit;
  p_prev_instruction_editor_->setReadOnly(true);
  p_prev_instruction_layout->addWidget( p_prev_instruction_editor_ );

  // Create the layout for a display of the response from the LLM
  QHBoxLayout * p_llm_response_layout = new QHBoxLayout;
  p_llm_response_layout->addWidget( new QLabel("LLM Response"));
  p_llm_response_textbox_ = new QTextEdit;
  p_llm_response_textbox_->setReadOnly(true);
  p_llm_response_layout->addWidget( p_llm_response_textbox_ );

  // Organize the layouts vertically
  QVBoxLayout * p_layout = new QVBoxLayout;
  p_layout->addLayout( p_instruction_layout );
  p_layout->addLayout( p_prev_instruction_layout );
  p_layout->addLayout( p_llm_response_layout );
  setLayout( p_layout );

  // Create a timer to regularly publish to the system monitor
  QTimer * p_timer_ = new QTimer(this);

  // Create Qt connections
  QObject::connect( p_instruction_editor_, SIGNAL(returnPressed()), this, SLOT(publishInstruction()) );
  QObject::connect( p_timer_, &QTimer::timeout, this, &InstructionPanel::publishSystemMonitor );
  p_timer_->start(500);
  return;
}

void InstructionPanel::onInitialize(){
  p_node_abstraction_ = getDisplayContext()->getRosNodeAbstraction().lock();
  rclcpp::Node::SharedPtr node = p_node_abstraction_->get_raw_node();

  // Create the publishers & subscriptions
  instruction_publisher_ = node->create_publisher<omniplanner_msgs::msg::LanguageGoalMsg>("/omniplanner_node/language_planner/language_goal", 10);
  system_monitor_publisher_ = node->create_publisher<ros_system_monitor_msgs::msg::NodeInfoMsg>("~/node_status", 1);
  llm_response_subscription_ = node->create_subscription<std_msgs::msg::String>("~/llm_response", 10, std::bind(&InstructionPanel::handleLLMResponse, this, std::placeholders::_1));
  return;
}

void InstructionPanel::handleLLMResponse(std_msgs::msg::String const & msg ){
  p_llm_response_textbox_->setText( msg.data.c_str() ); 
  return;
}

void InstructionPanel::publishInstruction( void ){
  // Construct and publish the instruction text as a String message 
  auto msg = omniplanner_msgs::msg::LanguageGoalMsg();
  msg.robot_id = "NO ONE";
  msg.command = p_instruction_editor_->text().toStdString();
  instruction_publisher_->publish( msg );

  // Update the display for the previous instruction
  p_prev_instruction_editor_->setText( p_instruction_editor_->text() );
  return;
}

void InstructionPanel::publishSystemMonitor( void ){
  // Get the ROS2 node
  auto p_node = p_node_abstraction_->get_raw_node();

  // Construct the message
  auto msg = ros_system_monitor_msgs::msg::NodeInfoMsg();
  msg.nickname = "nlu_rviz_panel";
  msg.node_name = p_node->get_fully_qualified_name();
  msg.status = ros_system_monitor_msgs::msg::NodeInfoMsg::NOMINAL;
  msg.notes = "You gonna give me orders?";
  system_monitor_publisher_->publish( msg );
  return;
}

}  // namespace nlu_interface

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(nlu_interface::InstructionPanel, rviz_common::Panel)
