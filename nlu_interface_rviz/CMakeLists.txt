cmake_minimum_required(VERSION 3.8)
project(nlu_interface_rviz)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(ament_cmake_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rviz_common REQUIRED)
find_package(ros_system_monitor_msgs REQUIRED)
find_package(omniplanner_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ManipulationApprovalRequest.msg"
  "msg/ManipulationApprovalResponse.msg"
  DEPENDENCIES sensor_msgs
)

set(CMAKE_AUTOMOC ON)
qt5_wrap_cpp(MOC_FILES
  include/nlu_interface_rviz/instruction_panel.hpp
  include/nlu_interface_rviz/manipulation_approval_panel.hpp
)

add_library(nlu_rviz_panels 
    src/instruction_panel.cpp 
    src/manipulation_approval_panel.cpp 
    ${MOC_FILES})

rosidl_get_typesupport_target(
  cpp_typesupport_target
  ${PROJECT_NAME}
  "rosidl_typesupport_cpp"
)
target_link_libraries(nlu_rviz_panels
  ${cpp_typesupport_target}
)

target_include_directories(nlu_rviz_panels PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(nlu_rviz_panels
  pluginlib
  rviz_common
  cv_bridge
  ros_system_monitor_msgs
  omniplanner_msgs
)
install(TARGETS nlu_rviz_panels
        EXPORT export_nlu_interface_rviz
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)


install(DIRECTORY include/
        DESTINATION include
)
install(FILES rviz_common_plugins.xml
        DESTINATION share/${PROJECT_NAME}
)
ament_export_include_directories(include)
ament_export_targets(export_nlu_interface_rviz)
pluginlib_export_plugin_description_file(rviz_common rviz_common_plugins.xml)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
