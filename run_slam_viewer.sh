#!/bin/bash
# Only allow the container with the ID (hostname) of ros_slam_core to control my X-Server
# xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_slam_viewer` 
nvidia-docker run \
-it \
--env="DISPLAY" \
--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
--rm \
--workdir="//opt/semslam" \
--link roscore_indigo \
--name ros_slam_viewer \
meppe78/lsd_slam-nvidia \
bash -c 'rosrun lsd_slam_viewer viewer'
# xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_slam_viewer`