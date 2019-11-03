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


class visual_odometry:

    def __init__(self, model_path = ""):
        print("vo")
        rospy.sleep(2.)
        #self.model = load_thingy
        self.bridge = CvBridge()
        self.image_feed_left = rospy.Subscriber("left_stereo", Image, self.get_left)
        self.image_feed_right = rospy.Subscriber("right_stereo", Image, self.get_right)
        self.l_odom_pub = rospy.Publisher("/camera/left/image_raw", Image, queue_size=10)
        self.r_odom_pub = rospy.Publisher("/camera/right/image_raw", Image, queue_size=10)
        self.disp_map_raw = None
        self.first_flag = False

    def get_left(self, data):
        self.left_img = self.bridge.imgmsg_to_cv2(data)
        self.l_odom_pub.publish(self.bridge.cv2_to_imgmsg(cv2.adaptiveThreshold(cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                            cv2.THRESH_BINARY, 7, 5), "8UC1"))
        self.first_flag = True

    def get_right(self, data):
        self.right_img = self.bridge.imgmsg_to_cv2(data)
        self.r_odom_pub.publish(self.bridge.cv2_to_imgmsg(cv2.adaptiveThreshold(cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 7, 5), "8UC1"))
        self.first_flag = True


    #TODO add the filtering code here
    def visual_odom_find(self):
        feature_params = dict( maxCorners = 10000,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 15 )
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 5,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        color = np.random.randint(0,255,(10000,3))
        while not self.first_flag:
            pass
        old_gray = cv2.adaptiveThreshold(cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 7, 3)

        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        mask = np.zeros_like(old_gray)
        while not rospy.is_shutdown():
            if self.first_flag:
                frame_gray = cv2.adaptiveThreshold(cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 7, 3)


                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]

                # draw the tracks
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame_gray,(a,b),5,color[i].tolist(),-1)
                img = cv2.add(frame_gray,mask)

                cv2.imshow('VO',img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1,1,2)

                self.visual_odom_pub.publish(self.bridge.cv2_to_imgmsg(img, "8UC1"))

def main():
    rospy.init_node('temporal_filter', anonymous=False)
    d_pub = visual_odometry()
    rospy.spin()
    #d_pub.visual_odom_find()

main()


