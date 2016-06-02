# Set up Workspace
# This is following the description on https://github.com/tum-vision/lsd_slam

# mkdir -rf ~/ros_slam/src
# cd ~/ros_slam/src
# catkin_init_workspace
# source /opt/ros/indigo/setup.bash
# source ~/ros_slam/devel/setup.bash
# export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/ros_slam/src

# git clone https://github.com/ros-drivers/gscam.git
# git clone https://github.com/tum-vision/lsd_slam.git
# git clone https://github.com/ros-perception/image_pipeline.git
# cd lsd_slam
# rosmake lsd_slam
# cd ~/ros_slam
# catkin_make

# To install openFABMap for loop closure opencv must be built from scratch and installed. If in problems, consider these pages:
# http://stackoverflow.com/questions/28010399/build-opencv-with-cuda-support
# 

# Run live SLAM

# Start droidcam client
# droidcam

# start roscore
# roscore &
gnome-terminal --geometry=40x10 --tab -e "bash -c 'roscore'" 

# start gscam
# cd ~/ros_slam/src/gscam
# export GSCAM_CONFIG="v4l2src device=/dev/video0 ! video/x-raw-rgb,framerate=50/1 ! ffmpegcolorspace"
# rosrun gscam gscam &


# (Optional: calibrate camera)
# rosrun camera_calibration cameracalibrator.py --size 7x7 --square 0.41 image:=/camera/image_raw camera:=/

# start lsd_slam with camera
# rosrun lsd_slam_core live_slam /image:=/camera/image_raw _calib:=lsd_slam_core/calib/lenovo_phone.cfg

# alternatively from bag video file
#rosrun lsd_slam_core live_slam image:=/image_raw camera_info:=/camera_info
gnome-terminal --geometry=40x10 --tab -e "bash -c 'rosrun lsd_slam_core live_slam image:=/image_raw camera_info:=/camera_info'" 

# play bagfile
#rosbag play src/lsd_slam/LSD_foodcourt.bag
gnome-terminal --geometry=40x10 --tab -e "bash -c 'rosbag play src/lsd_slam/LSD_foodcourt.bag'" 

# start slam viewer
# rosrun lsd_slam_viewer viewer
gnome-terminal --geometry=40x10 --tab -e "bash -c 'rosrun lsd_slam_viewer viewer'" 

# Also view raw image data
# rosrun image_view image_view image:=/image_raw
gnome-terminal --geometry=40x10 --tab -e "bash -c 'rosrun image_view image_view image:=/image_raw'" 

