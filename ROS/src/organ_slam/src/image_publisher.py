#!/usr/bin/env python

import rospy
import sys
import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import tf2_ros as tf2
import tf

class img_publisher:

    def __init__(self, path_to_video, frame_rate):
        self.image_pub_left = rospy.Publisher("left_stereo", Image, queue_size=10)
        self.image_pub_right = rospy.Publisher("right_stereo", Image, queue_size=10)
        self.image_info_pub_left = rospy.Publisher("stereo/left/camera_info", CameraInfo, queue_size=10)
        self.image_info_pub_right = rospy.Publisher("stereo/right/camera_info", CameraInfo, queue_size=10)
        self.br = tf.TransformBroadcaster()
        self.path_to_video = path_to_video
        self.cap = cv2.VideoCapture(self.path_to_video)
        if(not self.cap.isOpened()):
            print("Error in video file path")
        self.bridge = CvBridge()
        self.rate = rospy.Rate(frame_rate)

    def make_camera_msg(self, cam):
        camera_info_msg = CameraInfo()
        width, height = cam[0], cam[1]
        fx, fy = cam[2], cam[3]
        cx, cy = cam[4], cam[5]
        camera_info_msg.width = width
        camera_info_msg.height = height
        camera_info_msg.K = [fx, 0, cx,
                             0, fy, cy,
                             0, 0, 1]

        camera_info_msg.D = [cam[6], cam[7], cam[8], cam[9]]

        camera_info_msg.P = [fx, 0, cx, 0,
                             0, fy, cy, 0,
                             0, 0, 1, 0]
        return camera_info_msg

    def pull_loop(self):

        if(not self.cap.isOpened()):
            print("Error in video file path")

        while(self.cap.isOpened() and not rospy.is_shutdown()):
            try:
                ret, frame = self.cap.read()
                if(ret):
                    sz = np.shape(frame)
                    l_info = self.make_camera_msg([sz[0], sz[1], 381.914307,
                                                   383.797882, 168.108963, 126.979446,
                                                   -0.333236, 0.925076, 0.003847, 0.000916])
                    r_info = self.make_camera_msg([sz[0], sz[1], 381.670013,
                                                   382.582397, 129.929291, 120.092186,
                                                   -0.329342, 0.699034, 0.004927, 0.008194])

                    self.br.sendTransform((-19.5, 0, 0),
                                          tf.transformations.quaternion_from_euler(0, 0, 0),
                                          rospy.Time.now(),
                                          "left_cam",
                                          "base_link"
                                          )
                    self.br.sendTransform((19.5, 0, 0),
                                          tf.transformations.quaternion_from_euler(0, 0, 0),
                                          rospy.Time.now(),
                                          "right_cam",
                                          "base_link"
                                          )
                    split_width = frame.shape[1]//2
                    left_img, right_img = frame[:, 0:split_width, :], frame[:, split_width-1:-1, :]
                    l_img = self.bridge.cv2_to_imgmsg(left_img, "bgr8")
                    r_img = self.bridge.cv2_to_imgmsg(right_img, "bgr8")
                    l_img.header.frame_id = "left_cam"
                    r_img.header.frame_id = "right_cam"
                    self.image_pub_left.publish(l_img)
                    self.image_pub_right.publish(r_img)
                    self.image_info_pub_left.publish(l_info)
                    self.image_info_pub_right.publish(r_info)
                    self.rate.sleep()
            except KeyboardInterrupt:
                cv2.destroyAllWindows()

def main():

    rospy.init_node('img_publisher', anonymous=False)
    ig_pub = img_publisher("/home/advaith/Downloads/hamlyn_vids/stereo.avi", 10)
    rospy.loginfo("Image Publish Initialized")
    ig_pub.pull_loop()

main()


