#! /bin/bash
current_dir=`pwd`
mkdir  src/lsd_slam/lsd_slam_core/bin/
mkdir  src/lsd_slam/lsd_slam_viewer/bin/
mkdir  src/gscam/bin/
ln -s  $current_dir/devel/lib/lsd_slam_core/* src/lsd_slam/lsd_slam_core/bin/
ln -s  $current_dir/devel/lib/lsd_slam_viewer/viewer src/lsd_slam/lsd_slam_viewer/bin/viewer
ln -s  $current_dir/devel/lib/gscam/gscam src/gscam/bin/gscam

