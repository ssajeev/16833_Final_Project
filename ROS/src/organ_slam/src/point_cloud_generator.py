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
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2


class point_cloud_generator:

    def __init__(self, model_path = ""):
        print("Sleeping for 2.0 seconds")
        rospy.sleep(2.)
        #self.model = load_thingy
        self.bridge = CvBridge()
        self.disp_map_smooth_feed = rospy.Subscriber("disp_map_smooth", Image, self.get_disp_map)
        self.point_cloud_pub = rospy.Publisher("point_cloud_smooth", PointCloud2, queue_size=2)
        self.disp_map_smooth = None
        self.first_flag = False

    def get_disp_map(self, data):
        self.disp_map_smooth = self.bridge.imgmsg_to_cv2(data)
        self.first_flag = True

    def generate_point_cloud(self):
        while not rospy.is_shutdown():
            if self.first_flag:
                # cv2.imshow("d", self.disp_map_smooth)
                # cv2.waitKey(33)
                img_size = np.shape(self.disp_map_smooth)

                inds = np.indices(img_size)
                self.disp_map_smooth = self.disp_map_smooth / 255.0
                inds = np.reshape(inds, (2, img_size[0]*img_size[1])).astype(float)
                inds[0] = inds[0]/float(img_size[0])#reshape the indices
                inds[1] = inds[1]/float(img_size[1])*float(img_size[1])/float(img_size[0])

                #print inds
                depths = np.reshape(self.disp_map_smooth, (1, img_size[0]*img_size[1]))
                #print np.shape(depths)
                #print np.shape(inds)
                p_data = np.concatenate((inds.T, depths.T), axis=1)


                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'map'
                pcla = pcl2.create_cloud_xyz32(header, p_data)
                self.point_cloud_pub.publish(pcla)




def main():
    rospy.init_node('point_cloud_generator', anonymous=False)
    d_pub = point_cloud_generator()
    d_pub.generate_point_cloud()

main()


