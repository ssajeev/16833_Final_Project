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

l = 10000
sigma = 1.1
visual_mult = 1.0

class disparity_generator:

    def __init__(self, path_to_video, frame_rate):
        self.image_feed_left = rospy.Subscriber("left_stereo", Image, self.get_left)
        self.image_feed_right = rospy.Subscriber("right_stereo", Image, self.get_right)
        self.disparity_pub = rospy.Publisher("disp_map_raw", Image, queue_size=10)
        self.path_to_video = path_to_video
        self.cap = cv2.VideoCapture(self.path_to_video)
        if(not self.cap.isOpened()):
            print("Error in video file path")
        self.bridge = CvBridge()
        self.rate = rospy.Rate(frame_rate)
        self.left_img = None
        self.right_img = None

    def get_left(self, data):
        while(self.left_flag):
            pass
        print("got1")
        self.left_img = self.bridge.imgmsg_to_cv2(data)

    def get_right(self, data):
        while(self.right_flag):
            pass
        print("got2")
        self.right_img = self.bridge.imgmsg_to_cv2(data)

    def generate_disp_map(self):
        while(1):
            if(self.left_img != None or self.right_img != None):
                continue
            left_img = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY)
            right_img = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY)
            l_stereo = cv2.StereoSGBM_create(minDisparity=6,
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
            filtered = cv2.applyColorMap(filtered, cv2.COLORMAP_JET)
            #disparity = cv2.convertScaleAbs(stereo.compute(left_img, right_img))
            self.disparity_pub.publish(self.bridge.cv2_to_imgmsg(filtered, "bgr8"))





def main():
    print("Began Disp Generator Node")
    rospy.init_node('disparity_generator', anonymous=False)
    d_pub = disparity_generator("/home/advaith/Downloads/hamlyn_vids/stereo.avi", 24);
    d_pub.generate_disp_map()

main()


