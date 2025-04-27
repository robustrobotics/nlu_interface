#ifndef NLU_INTERFACE__INSTRUCTION_PANEL_HPP_
#define NLU_INTERFACE__INSTRUCTION_PANEL_HPP_

// ROS
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <std_msgs/msg/string.hpp>
#include <ros_system_monitor_msgs/msg/node_info_msg.hpp>

// Qt
#include <QLabel>
#include <QTextEdit>
#include <QLineEdit>
#include <QTimer>

namespace nlu_interface {
class InstructionPanel : public rviz_common::Panel
{
  Q_OBJECT
public:
  explicit InstructionPanel(QWidget * parent = 0);
  ~InstructionPanel() override = default;
  void onInitialize() override;

protected:

  //void topicCallback(const std_msgs::msg::String & msg);
  void handleLLMResponse( std_msgs::msg::String const & msg );


  // ROS2 member variables
  std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> node_ptr_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr instruction_publisher_;
  rclcpp::Publisher<ros_system_monitor_msgs::msg::NodeInfoMsg>::SharedPtr system_monitor_publisher_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr llm_response_subscription_;

  // GUI member variables
  QTextEdit * p_llm_response_textbox_;
  QLineEdit * p_prev_instruction_editor_;
  QLineEdit * p_instruction_editor_;
  QTimer * p_timer_;

private Q_SLOTS:
  void publishInstruction( void );
  void publishSystemMonitor( void );
}; // class InstructionPanel
}  // namespace nlu_interface

#endif  // NLU_INTERFACE__INSTRUCTION_PANEL_HPP_
