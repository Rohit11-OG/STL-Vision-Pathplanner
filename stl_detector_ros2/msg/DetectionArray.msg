# DetectionArray.msg
# Array of detections with header

# Header with timestamp and frame
std_msgs/Header header

# Array of detections
stl_detector_ros2/Detection[] detections

# Number of detections
int32 count
