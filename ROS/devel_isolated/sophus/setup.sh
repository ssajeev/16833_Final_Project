#!/usr/bin/env sh
# generated from catkin.builder Python module

# remember type of shell if not already set
if [ -z "$CATKIN_SHELL" ]; then
  CATKIN_SHELL=sh
fi
. "/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/rqt_svo/setup.$CATKIN_SHELL"

# detect if running on Darwin platform
_UNAME=`uname -s`
IS_DARWIN=0
if [ "$_UNAME" = "Darwin" ]; then
  IS_DARWIN=1
fi

# Prepend to the environment
export CMAKE_PREFIX_PATH="/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus:$CMAKE_PREFIX_PATH"
if [ $IS_DARWIN -eq 0 ]; then
  export LD_LIBRARY_PATH="/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus/lib:/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
else
  export DYLD_LIBRARY_PATH="/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus/lib:/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus/lib/x86_64-linux-gnu:$DYLD_LIBRARY_PATH"
fi
export PATH="/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus/bin:$PATH"
export PKG_CONFIG_PATH="/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus/lib/pkgconfig:/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH"
export PYTHONPATH="/home/advaith/Documents/16833_Final_Project/ROS/devel_isolated/sophus/lib/python2.7/dist-packages:$PYTHONPATH"
