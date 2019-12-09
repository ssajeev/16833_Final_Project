#!/usr/bin/env python
# point_cloud_generator.py
#
# Node creates dense pointclouds from disparity map images
#
#
# Author: Sandra Sajeev, Advaith Sethuraman 2019

import rospy
import sys
import cv2
import math
import numpy as np
from sklearn.preprocessing import normalize
import rospy
import cv2
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import *
from tf.transformations import *
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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pose = PoseStamped()
        self.rgb_img = None
        self.first_flag = False
        self.first_flag_rgb = False
        self.prev_cloud = PointCloud2()
        self.righty = TransformStamped()
        self.righty.header.frame_id = "base_link"
        self.righty.child_frame_id = "base_link"
        quat = quaternion_from_euler(0.0, 0.0, 1.57)
        self.righty.transform.rotation.x = quat[0]
        self.righty.transform.rotation.y = quat[1]
        self.righty.transform.rotation.z = quat[2]
        self.righty.transform.rotation.w = quat[3]
        self.transl = TransformStamped()
        self.transl.header.frame_id = "base_link"
        self.transl.child_frame_id = "base_link"
        quat = quaternion_from_euler(0.0, 0.0, 0.0)
        self.transl.transform.rotation.x = quat[0]
        self.transl.transform.rotation.y = quat[1]
        self.transl.transform.rotation.z = quat[2]
        self.transl.transform.rotation.w = quat[3]
        self.transl.transform.translation = Vector3(-.5, -0.5, 0.0)
        self.global_pct = None
        self.init_global = True


    def get_disp_map(self, data):
        self.disp_map_smooth = self.bridge.imgmsg_to_cv2(data)
        self.first_flag = True

    def get_rgb_img(self, data):
        self.rgb_img = self.bridge.imgmsg_to_cv2(data)
        self.first_flag_rgb = True

    #generate point cloud from depth map topic
    def generate_point_cloud(self):
        f = open("data_log.txt", "w+")
        while not rospy.is_shutdown():
            if self.first_flag and self.first_flag_rgb:
                img_size = np.shape(self.disp_map_smooth)

                rgb_frame = np.reshape(self.rgb_img / 255.0,
                                       (np.shape(self.rgb_img)[0]*np.shape(self.rgb_img)[1],
                                       3))
                inds = np.indices(img_size)
                self.disp_map_smooth = np.reshape(self.disp_map_smooth, img_size).astype(float)
                inds = np.reshape(inds, (2, img_size[0]*img_size[1])).astype(float)
                tmp = np.copy(inds[0])
                inds[0] = inds[1]/float(img_size[1])*float(img_size[1])/float(img_size[0]) - float(img_size[1])/float(img_size[0])/2.0
                      #reshape the indices
                inds[1] = -(tmp/float(img_size[0]) - 1.0/2.0)
                depths = -1.0*np.reshape(255 - self.disp_map_smooth, (1, img_size[0]*img_size[1])) / 255.0
                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "base_link"
                p_data = np.concatenate((inds.T, depths.T), axis=1)
                xyzrgb_data = np.concatenate((p_data, rgb_frame), axis=1)
                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('b', 12, PointField.FLOAT32, 1),
                    PointField('g', 16, PointField.FLOAT32, 1),
                    PointField('r', 20, PointField.FLOAT32, 1)
                ]
                pc2 = point_cloud2.create_cloud(header, fields, xyzrgb_data)
                try:
                    trans = self.tf_buffer.lookup_transform("base_link", "map", rospy.Time(0))
                    pc2 = do_transform_cloud(pc2, trans)
                except tf2.LookupException as ex:
                    rospy.logwarn(ex)
                except tf2.ExtrapolationException as ex:
                    rospy.logwarn(ex)
                pts = pcl2.read_points(pc2, field_names=("x", "y", "z", "b", "g", "r"), skip_nans=True)
                pts = np.asarray(list(pts))
                print(np.shape(pts))
                if(self.init_global):
                    self.global_pct = pts
                    self.init_global = False
                else:
                    self.global_pct = pts
                    #np.concatenate([self.global_pct, pts], axis = 0)
                    print(np.shape(self.global_pct))
                header2 = Header()
                header2.stamp = rospy.Time.now()
                header2.frame_id = "base_link"
                global_ros_pointcloud = point_cloud2.create_cloud(header2, fields, self.global_pct)
                self.point_cloud_pub.publish(global_ros_pointcloud)

def main():
    rospy.init_node('point_cloud_generator', anonymous=False)
    d_pub = point_cloud_generator()
    rospy.loginfo("Point Cloud Generator Initialized")
    d_pub.generate_point_cloud()

main()


