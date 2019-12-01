execute_process(COMMAND "/home/advaith/Documents/16833_Final_Project/ROS/build/rpg_vikit/vikit_py/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/advaith/Documents/16833_Final_Project/ROS/build/rpg_vikit/vikit_py/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
