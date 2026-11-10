string robot_id
# Raw candidate images, one per camera. Empty when the sender uses the
# compressed transport below (the executor's default).
sensor_msgs/Image[] images
string[] image_source_names
bool has_detection
uint32 detection_image_index
int32 image_x
int32 image_y

# Compressed transport: same order and pixel grid as image_source_names.
# JPEG of the candidate image (format "jpeg").
sensor_msgs/CompressedImage[] compressed_images
# Depth registered to each candidate image's pixels (e.g. Spot's
# <camera>_depth_in_visual_frame / hand_depth_in_hand_color_frame), as a
# lossless 16-bit single-channel PNG (format "png"). Raw values: 0 and 65535
# mean no measurement. An entry with empty data means no depth for that camera.
sensor_msgs/CompressedImage[] depth_images
# Raw depth units per meter for each depth_images entry (Spot: usually 1000).
float32[] depth_scales

# Walk-to preview: what the robot needs to turn a clicked pixel into a walk
# goal, so a viewer can show the same ground-plane estimate before the operator
# answers. Everything is in the robot's fixed "vision" frame at capture time.
# Per candidate image (same order), 4 values: fx, fy, cx, cy. Zeros = unknown.
float64[] camera_intrinsics
# Per candidate image, 7 values: sensor position x y z, then orientation
# qw qx qy qz (camera axes: +z forward, +x right, +y down). Zeros = unknown.
float64[] camera_poses
# Ground plane estimate: a point x y z, then its unit normal x y z. Empty = unknown.
float64[] ground_plane
# Robot body x, y, yaw. Empty = unknown.
float64[] robot_pose
# The walk the robot will plan: stop this far short of the target (m), and
# never translate farther than this in one walk (m).
float64 walk_to_standoff
float64 walk_to_max_distance
