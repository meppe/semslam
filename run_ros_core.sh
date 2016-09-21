#!/bin/bash
docker run \
--rm \
-v /$(pwd):/opt/semslam \
-it \
--workdir="//opt/semslam" \
--name roscore_indigo \
meppe78/roscore-indigo \
roscore 
