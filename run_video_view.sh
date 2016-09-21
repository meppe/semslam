#!/bin/bash
# Only allow the container with the ID (hostname) of ros_video_view to control my X-Server
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_view` 
docker run \
--rm \
-v /$(pwd):/opt/semslam \
-it \
--workdir="//opt/semslam" \
--link roscore_indigo \
--name ros_video_view \
-e ROS_MASTER_URI=http://roscore_indigo:11311/ \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
meppe78/ros-video-view \
bash -c 'rosrun image_view image_view image:=/image_raw'
xhost -local:`docker inspect --format='{{ .Config.Hostname }}' ros_video_view`