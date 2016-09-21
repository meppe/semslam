#!/bin/bash
docker run \
--rm \
-v /$(pwd):/opt/semslam \
-it \
--workdir="//opt/semslam" \
--link roscore_indigo \
-e ROS_MASTER_URI=http://roscore_indigo:11311/ \
--name ros_video_stream \
meppe78/roscore-indigo \
bash -c 'rosbag play -l video_bags/LSD_foodcourt.bag'
