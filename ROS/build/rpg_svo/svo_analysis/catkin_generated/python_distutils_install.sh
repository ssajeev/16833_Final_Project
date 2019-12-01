#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/advaith/Documents/16833_Final_Project/ROS/src/rpg_svo/svo_analysis"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/advaith/Documents/16833_Final_Project/ROS/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/advaith/Documents/16833_Final_Project/ROS/install/lib/python2.7/dist-packages:/home/advaith/Documents/16833_Final_Project/ROS/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/advaith/Documents/16833_Final_Project/ROS/build" \
    "/usr/bin/python2" \
    "/home/advaith/Documents/16833_Final_Project/ROS/src/rpg_svo/svo_analysis/setup.py" \
    build --build-base "/home/advaith/Documents/16833_Final_Project/ROS/build/rpg_svo/svo_analysis" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/home/advaith/Documents/16833_Final_Project/ROS/install" --install-scripts="/home/advaith/Documents/16833_Final_Project/ROS/install/bin"
