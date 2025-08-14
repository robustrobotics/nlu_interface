#ifndef NLU_INTERFACE_RVIZ__INSTRUCTION_PANEL_HPP_
#define NLU_INTERFACE_RVIZ__INSTRUCTION_PANEL_HPP_

// ROS
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <ros_system_monitor_msgs/msg/node_info_msg.hpp>
#include <omniplanner_msgs/msg/language_goal_msg.hpp>
#include <omniplanner_msgs/msg/goto_points_goal_msg.hpp>
//#include <nlu_interface_rviz/msg/manipulation_approval_request.hpp>

// Qt
#include <QLabel>
#include <QTextEdit>
#include <QLineEdit>
#include <QComboBox>
#include <QRadioButton>
#include <QPushButton>
#include <QTimer>
#include <QSet>
#include <QString>

namespace nlu_interface_rviz {
class InstructionPanel : public rviz_common::Panel
{
    Q_OBJECT
public:
    explicit InstructionPanel(QWidget * parent = 0);
    ~InstructionPanel() override = default;
    void onInitialize() override;

protected:

    void publishManipulationResponse( bool const approve );

    // ROS callbacks
    void handleLLMResponse( std_msgs::msg::String const & msg );
    void handleManipulationRequest( std_msgs::msg::String const & msg ); // TODO: Use ManipulationApprovalRequest message
    // Data members
    QSet<QString> robot_ids_;


    // ROS2 member variables
    std::shared_ptr<rviz_common::ros_integration::RosNodeAbstractionIface> p_node_abstraction_;
    rclcpp::Publisher<omniplanner_msgs::msg::LanguageGoalMsg>::SharedPtr instruction_publisher_;
    rclcpp::Publisher<ros_system_monitor_msgs::msg::NodeInfoMsg>::SharedPtr system_monitor_publisher_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr llm_response_subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr manipulation_request_subscription_;
    // map of robot ids to ros publishers
    std::map< std::string, rclcpp::Publisher< std_msgs::msg::Bool >::SharedPtr> manipulation_approval_publishers_;

    // GUI member variables
    QTextEdit * p_llm_response_textbox_;
    QLineEdit * p_prev_instruction_editor_;
    QLineEdit * p_instruction_editor_;
    QComboBox * p_domain_type_combo_box_;
    QComboBox * p_robot_id_combo_box_;
    QComboBox * p_manipulation_robot_id_combo_box_;
    QLabel * p_manipulation_image_label_; //TODO: Display a sensor_msgs/Image as a QLabel
    QPushButton * p_manipulation_approve_push_button_;
    QPushButton * p_manipulation_reject_push_button_;
    QTimer * p_timer_;

private Q_SLOTS:
    void publishInstruction( void );
    void publishManipulationApproval( void );
    void publishManipulationRejection( void );
    void publishSystemMonitor( void );
}; // class InstructionPanel
}  // namespace nlu_interface_rviz

#endif  // NLU_INTERFACE_RVIZ__INSTRUCTION_PANEL_HPP_
