#!/usr/bin/env python

import rospy
import sys
import cv2
import math
import numpy as np
from sklearn.preprocessing import normalize
import rospy
import cv2
from sensor_msgs.msg import Images
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pcl2
import pcl
import open3d as o3d
from utils import *

class point_cloud_registration:

    def __init__(self):
        self.point_cloud_sub = rospy.Subscriber("point_cloud_smooth", PointCloud2, self.generate_point_cloud)
        self.global_point_cloud_pub = rospy.Publisher("point_cloud", PointCloud2, queue_size=2)
        self.global_fusion = None
        pass

    def fuse_point_cloud(self, source):

        pcl_source = ros_to_pcl(source)

        inlier_final = pcl.PointXYZRGB()

        model_s = pcl.SampleConsensusModelSphere(pcl_source)
        ransac = pcl.RandomSampleConsensus(model_s)
        ransac.set_DistanceThreshold (0.01)
        ransac.computeModel()
        inliers = ransac.get_Inliers()

        if len(inliers) != 0:
            finalpoints = np.zeros((len(inliers), 6), dtype=np.float32)

            for i in range(0, len(inliers)):
                finalpoints[i][0] = cloud[inliers[i]][0]
                finalpoints[i][1] = cloud[inliers[i]][1]
                finalpoints[i][2] = cloud[inliers[i]][2]
                finalpoints[i][3] = cloud[inliers[i]][3]
                finalpoints[i][4] = cloud[inliers[i]][4]
                finalpoints[i][5] = cloud[inliers[i]][5]

            inlier_final.from_array(finalpoints)


            if(self.global_fusion == None):
                self.global_fusion = inlier_final;
            else:
                self.global_fusion = self.global_fusion + inlier_final

    def get_global_point_cloud(self):
        if(self.global_fusion != None):
            ros_target = pcl_to_ros(self.global_fusion)
            self.global_point_cloud_pub.publish(ros_target)
        pass
