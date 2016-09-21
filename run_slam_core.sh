#!/bin/bash
# Only allow the container with the ID (hostname) of ros_slam_core to control my X-Server
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_slam_core` 
docker run \
--rm \
-it \
--workdir="//opt/semslam" \
--link roscore_indigo \
--name ros_slam_core \
-e ROS_MASTER_URI=http://roscore_indigo:11311/ \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
meppe78/lsd_slam \
bash -c 'rosrun lsd_slam_core live_slam image:=/image_raw camera_info:=/camera_info'
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_slam_core`