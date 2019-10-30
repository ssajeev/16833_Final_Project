#!/usr/bin/env python

import rospy
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class img_publisher:

    def __init__(self, path_to_video):
        self.path_to_video = path_to_video
        self.cap = cv2.VideoCapture(path_to_video)
        if(not self.cap.isOpened()):
            print("Error in video file path")
        self.bridge = CvBridge()

    def pull_loop(self):
        if(not self.cap.isOpened()):
            print("Error in video file path")


