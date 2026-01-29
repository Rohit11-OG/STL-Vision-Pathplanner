# Detection.msg
# Single object detection with 3D position and orientation

# Header with timestamp and frame
std_msgs/Header header

# Detection class information
string class_name
int32 class_id
float32 confidence

# 2D bounding box in pixels [x1, y1, x2, y2]
int32[4] bbox

# Center point in pixels
int32 center_x
int32 center_y

# 3D position in camera frame (meters)
geometry_msgs/Point position

# Object orientation (degrees) - roll, pitch, yaw
geometry_msgs/Vector3 orientation

# Distance from camera (meters)
float32 distance
