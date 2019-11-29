#!/usr/bin/env python

import rospy
import sys
import cv2
import math
import numpy as np
from sklearn.preprocessing import normalize
import rospy
import cv2
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
        self.point_cloud_sub = rospy.Subscriber("point_cloud_smooth", PointCloud2, self.get_point_cloud)
        self.global_point_cloud_pub = rospy.Publisher("point_cloud_global", PointCloud2, queue_size=2)
        self.global_fusion = None
        self.source = None
        pass



    def get_point_cloud(self, data):
        rospy.loginfo("Got a pointcloud")
        self.source = data
    
    def concatenate_point_cloud(self, global_model, new_model):
        
        new_model_pts = (new_model.size)
        global_model_pts = (global_model.size)
        finalpoints = np.zeros((new_model_pts + global_model_pts, 3), dtype=np.float32)

        offset = 0
        for i in range(0, global_model_pts):
            finalpoints[i][0] = global_model[i][0]
            finalpoints[i][1] = global_model[i][1]
            finalpoints[i][2] = global_model[i][2]
            offset = i

        for j in range(0, new_model_pts):
            finalpoints[j+offset][0] = new_model[j][0]
            finalpoints[j+offset][1] = new_model[j][1]
            finalpoints[j+offset][2] = new_model[j][2]

        final_model = pcl.PointCloud()
        final_model.from_array(finalpoints)

        return final_model
         

    def fuse_point_cloud(self):
        if(self.source == None):
            return
        source = self.source
        pcl_source = ros_to_pcl(source)

        inlier_final = pcl.PointCloud()

        model_s = pcl.SampleConsensusModelSphere(pcl_source)
        ransac = pcl.RandomSampleConsensus(model_s)
        ransac.set_DistanceThreshold (0.01)
        ransac.computeModel()
        inliers = ransac.get_Inliers()

        if len(inliers) != 0:
            finalpoints = np.zeros((len(inliers), 3), dtype=np.float32)

            for i in range(0, len(inliers)):
                finalpoints[i][0] = pcl_source[inliers[i]][0]
                finalpoints[i][1] = pcl_source[inliers[i]][1]
                finalpoints[i][2] = pcl_source[inliers[i]][2]
            
            print(finalpoints)

            inlier_final.from_array(finalpoints)
           

            if(self.global_fusion == None):
                rospy.loginfo("Set inlier model")
                self.global_fusion = inlier_final;
            else:
                rospy.loginfo("Fused points")
                
                self.global_fusion = self.concatenate_point_cloud(self.global_fusion,inlier_final)

    def get_global_point_cloud(self):
        while not rospy.is_shutdown():
            self.fuse_point_cloud()
            if(self.global_fusion != None):
                ros_target = pcl_to_ros(self.global_fusion)
                self.global_point_cloud_pub.publish(ros_target)
        pass

def main():

    rospy.init_node('point_fusion', anonymous=False)
    global_pub = point_cloud_registration()
    rospy.loginfo("Point Cloud Basic Fusion")
    global_pub.get_global_point_cloud()

main()


