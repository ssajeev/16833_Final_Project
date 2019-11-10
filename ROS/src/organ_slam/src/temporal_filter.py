#!/usr/bin/env python

import rospy
import sys
import cv2
import math
import numpy as np
from sklearn.preprocessing import normalize
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class temporal_filter:

    def __init__(self, model_path = ""):
        print("temporal")
        rospy.sleep(2.)
        #self.model = load_thingy
        self.bridge = CvBridge()
        self.disp_map_raw_feed = rospy.Subscriber("disp_map_raw", Image, self.get_disp_map)
        self.disp_map_smoothed_pub = rospy.Publisher("disp_map_smooth", Image, queue_size=10)
        self.disp_map_raw = None
        self.first_flag = False

    def get_disp_map(self, data):
        self.disp_map_raw = self.bridge.imgmsg_to_cv2(data)
        self.first_flag = True

    #TODO add the filtering code here
    def filter_disp_map(self):
        while not rospy.is_shutdown():
            if self.first_flag:
                self.disp_map_smoothed_pub.publish(self.bridge.cv2_to_imgmsg(self.disp_map_raw, "8UC1"))

def main():
    rospy.init_node('temporal_filter', anonymous=False)
    t_pub = temporal_filter()
    rospy.loginfo("Temporal Filter Initialized")
    t_pub.filter_disp_map()

main()


