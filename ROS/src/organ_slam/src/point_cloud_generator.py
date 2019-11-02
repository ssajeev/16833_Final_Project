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
                cv2.imshow("d", self.disp_map_smooth)
                cv2.waitKey(33)
                img_size = np.shape(self.disp_map_smooth)
                print "img_size", img_size
                inds = np.indices(img_size)

                inds = np.reshape(inds, (2, img_size[0]*img_size[1])).astype(float)
                inds[0] = inds[0]/float(img_size[0])*10.0#reshape the indices
                inds[1] = inds[1]/float(img_size[1])*10.0*float(img_size[1])/float(img_size[0])

            #print inds
                depths = np.reshape(self.disp_map_smooth, (1, img_size[0]*img_size[1]))
                #print np.shape(depths)
                #print np.shape(inds)
                p_data = np.concatenate((inds.T, depths.T), axis=1)
                #print np.shape(p_data)
                p_data = np.reshape(p_data, (1, 3*img_size[0]*img_size[1]))
                #print p_data

                msg = PointCloud2()
                msg.header.frame_id = "map"
                msg.height = p_data.shape[0]
                msg.width = p_data.shape[1]
                msg.fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1)]
                msg.is_bigendian = False
                msg.point_step = 12
                msg.row_step = 4*img_size[0]*img_size[1]
                msg.is_dense = True
                msg.data = np.asarray(p_data, np.float32).tostring()
                self.point_cloud_pub.publish(msg)



def main():
    rospy.init_node('point_cloud_generator', anonymous=False)
    d_pub = point_cloud_generator()
    d_pub.generate_point_cloud()

main()


