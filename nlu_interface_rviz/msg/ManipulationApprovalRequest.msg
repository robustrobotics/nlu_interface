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
