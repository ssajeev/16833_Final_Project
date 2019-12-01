
import rospy
import sys
import cv2
import math
import numpy as np
from sklearn.preprocessing import normalize
import rospy
import cv2
import struct
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import pcl
from random import randint
import open3d as o3d


def random_color_gen():
    """ Generates a random color
    
        Args: None
        
        Returns: 
            list: 3 elements, R, G, and B
    """
    r = randint(0, 255)
    g = randint(0, 255)
    b = randint(0, 255)
    return [r, g, b]

def pcl_to_ros(pcl_array):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB
    
        Args:
            pcl_array (PointCloud_PointXYZRGB): A PCL XYZRGB point cloud
            
        Returns:
            PointCloud2: A ROS point cloud
    """
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "global"
    
    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]

    pc2_cloud = point_cloud2.create_cloud(header, fields, pcl_array.to_array())
    return pc2_cloud

def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB
    
        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message
            
        Returns:
            pcl.PointCloud_PointXYZRGB: PCL XYZRGB point cloud
    """
    points_list = []

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        points_list.append([data[0], data[1], data[2]])

    pcl_data = pcl.PointCloud()
    pcl_data.from_list(points_list)

    return pcl_data 


