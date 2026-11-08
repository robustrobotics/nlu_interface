# Operator decision for a ManipulationApprovalRequest.
#
# approve=true: grasp at (image_x, image_y) in images[image_index].
# approve=false, walk_to=false: abandon the grasp attempt.
# approve=false, walk_to=true: do not grasp yet. Walk toward the ground/object
#   point (image_x, image_y) in images[image_index], then capture new images
#   and send a fresh request.
bool approve
int32 image_x
int32 image_y
uint32 image_index

bool walk_to

# Optional heading control for walk_to: stand at this ground pixel (same image
# as image_index) facing the target point. When has_stand_point is false the
# robot stops a configured standoff distance short of the target, facing it.
bool has_stand_point
int32 stand_x
int32 stand_y
