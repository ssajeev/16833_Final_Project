#!/usr/bin/env python

import rospy
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class img_publisher:

    def __init__(self, path_to_video, frame_rate):
        self.image_pub_left = rospy.Publisher("left_stereo", Image, queue_size=10)
        self.image_pub_right = rospy.Publisher("right_stereo", Image, queue_size=10)
        self.path_to_video = path_to_video
        self.cap = cv2.VideoCapture(self.path_to_video)
        if(not self.cap.isOpened()):
            print("Error in video file path")
        self.bridge = CvBridge()
        self.rate = rospy.Rate(frame_rate)


    def pull_loop(self):
        print("Sleeping for 2 seconds")

        if(not self.cap.isOpened()):
            print("Error in video file path")

        while(self.cap.isOpened() and not rospy.is_shutdown()):
            try:
                ret, frame = self.cap.read()
                if(ret):
                    split_width = frame.shape[1]//2
                    left_img, right_img = frame[:, 0:split_width, :], frame[:, split_width-1:-1, :]
                    self.image_pub_left.publish(self.bridge.cv2_to_imgmsg(left_img, "bgr8"))
                    self.image_pub_right.publish(self.bridge.cv2_to_imgmsg(right_img, "bgr8"))
                    self.rate.sleep()
            except KeyboardInterrupt:
                cv2.destroyAllWindows()

def main():
    print("Began Image Publisher Node")
    rospy.init_node('img_publisher', anonymous=False)
    ig_pub = img_publisher("/home/advaith/Downloads/hamlyn_vids/stereo.avi", 24);
    ig_pub.pull_loop()

main()


