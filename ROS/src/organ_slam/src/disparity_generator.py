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
from inference import *

l = 10000
sigma = 1.1
visual_mult = 1.0

class disparity_generator:

    def __init__(self, frame_rate, model_path = ""):
        print("Sleeping for 2.0 seconds")
        rospy.sleep(2.)
        #self.model = load_thingy
        self.bridge = CvBridge()
        self.image_feed_left = rospy.Subscriber("left_stereo", Image, self.get_left)
        self.image_feed_right = rospy.Subscriber("right_stereo", Image, self.get_right)
        self.disparity_pub = rospy.Publisher("disp_map_raw", Image, queue_size=10)
        self.model = load_model(model_path)
        self.rate = rospy.Rate(frame_rate)
        self.left_img = None
        self.right_img = None
        self.l_flag = False
        self.r_flag = False

    def get_left(self, data):

        self.left_img = self.bridge.imgmsg_to_cv2(data)
        self.l_flag = True

    def get_right(self, data):

        self.right_img = self.bridge.imgmsg_to_cv2(data)
        self.r_flag = True

    def generate_geometric_disp_map(self):
        while not rospy.is_shutdown():
            if(not self.l_flag or not self.r_flag):
                continue
            else:
                left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY)
                right_img = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY)

                cv2.waitKey(33)
                l_stereo = cv2.StereoSGBM_create(minDisparity=0,
                                                 numDisparities=16,
                                                 blockSize=3,
                                                 uniquenessRatio=15,
                                                 speckleWindowSize=0,
                                                 speckleRange=2,
                                                 preFilterCap=20,
                                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
                r_stereo = cv2.ximgproc.createRightMatcher(l_stereo)
                wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=l_stereo)
                wls_filter.setLambda(l)
                wls_filter.setSigmaColor(sigma)
                d_l = l_stereo.compute(left_img, right_img)
                d_r = r_stereo.compute(right_img, left_img)
                d_l = np.int16(d_l)
                d_r = np.int16(d_r)
                filtered = wls_filter.filter(d_l, left_img, None, d_r)
                filtered = cv2.normalize(src=filtered, dst=filtered, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
                filtered = np.uint8(filtered)
                #filtered = cv2.applyColorMap(filtered, cv2.COLORMAP_JET)
                #disparity = cv2.convertScaleAbs(stereo.compute(left_img, right_img))
                self.disparity_pub.publish(self.bridge.cv2_to_imgmsg(filtered, "8UC1"))

    def generate_smart_disparity_map(self):
        while not rospy.is_shutdown():
            depth_inference = inference(self.model, self.left_img, self.right_img)
            self.disparity_pub.publish(self.bridge.cv2_to_imgmsg(depth_inference, "8UC1"))



def main():
    print("Began Disp Generator Node")
    rospy.init_node('disparity_generator', anonymous=False)
    d_pub = disparity_generator(24, '/home/advaith/Documents/16833_Final_Project/ROS/src/organ_slam/src/saved_model_10-31-2019-21_41_41.pt');
    d_pub.generate_smart_disp_map()

main()


