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
import open3d as o3d


class point_cloud_generator:

    def __init__(self, model_path = ""):
        np.set_printoptions(threshold=sys.maxsize)
        print("pcl")
        rospy.sleep(2.)
        #self.model = load_thingy
        self.bridge = CvBridge()
        self.disp_map_smooth_feed = rospy.Subscriber("disp_map_smooth", Image, self.get_disp_map)
        self.rgb_feed = rospy.Subscriber("left_stereo", Image, self.get_rgb_img)
        self.point_cloud_pub = rospy.Publisher("point_cloud_smooth", PointCloud2, queue_size=2)
        self.disp_map_smooth = None
        self.rgb_img = None
        self.first_flag = False
        self.first_flag_rgb = False

    def get_disp_map(self, data):
        self.disp_map_smooth = self.bridge.imgmsg_to_cv2(data)
        self.first_flag = True

    def get_rgb_img(self, data):
        self.rgb_img = self.bridge.imgmsg_to_cv2(data)
        self.first_flag_rgb = True

    def generate_point_cloud(self):
        f = open("data_log.txt", "w+")
        while not rospy.is_shutdown():
            if self.first_flag and self.first_flag_rgb:
                # cv2.imshow("d", self.disp_map_smooth)
                # cv2.waitKey(33)
                img_size = np.shape(self.disp_map_smooth)

                rgb_frame = np.reshape(self.rgb_img,
                                       (np.shape(self.rgb_img)[0]*np.shape(self.rgb_img)[1],
                                       3))



                inds = np.indices(img_size)
                # f.write("map:")
                # f.write(str(self.disp_map_smooth))
                self.disp_map_smooth = np.reshape(self.disp_map_smooth, img_size).astype(float)
                # f.write("map_s:")
                # f.write(str(self.disp_map_smooth))
                inds = np.reshape(inds, (2, img_size[0]*img_size[1])).astype(float)

                inds[0] = inds[0]/float(img_size[0])#reshape the indices
                inds[1] = inds[1]/float(img_size[1])*float(img_size[1])/float(img_size[0])

                #print inds
                depths = np.reshape(self.disp_map_smooth, (1, img_size[0]*img_size[1])) / 255.0
                # f.write("resae:")
                # f.write(str(depths))
                #print np.shape(depths)
                #print np.shape(inds)

                ###Pointcloud Msg Header
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "map"

                p_data = np.concatenate((inds.T, depths.T), axis=1)

                xyzrgb_data = np.concatenate((p_data, rgb_frame), axis=1)

                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('r', 12, PointField.FLOAT32, 1),
                    PointField('g', 16, PointField.FLOAT32, 1),
                    PointField('b', 20, PointField.FLOAT32, 1)
                ]
                pc2 = point_cloud2.create_cloud(header, fields, xyzrgb_data)
                # f.write("p_data:")
                # f.write(str(p_data))
                # header = Header()
                # header.stamp = rospy.Time.now()
                # header.frame_id = 'map'
                # pcla = pcl2.create_cloud_xyz32(header, p_data)
                self.point_cloud_pub.publish(pc2)




def main():

    rospy.init_node('point_cloud_generator', anonymous=False)
    d_pub = point_cloud_generator()
    rospy.loginfo("Point Cloud Generator Initialized")
    d_pub.generate_point_cloud()

main()


