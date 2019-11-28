#! /usr/bin/env python

 
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_grabber:
	def __init__(self, path_to_video,rate):
		self.image_left_publisher = rospy.Publisher("/camera/left/image_raw", Image)
		self.image_right_publisher = rospy.Publisher("/camera/right/image_raw", Image)
		self.bridge = CvBridge()
		self.cap = cv2.VideoCapture(path_to_video)
		self.rate = rospy.Rate(rate)
		print("created cap")

	def grab_and_send(self):
		print("in grab and send")
		if(not self.cap.isOpened()):
			print("Error in video file path")
		while(self.cap.isOpened()):
			ret, frame = self.cap.read()
			split_width = frame.shape[1]//2
			left_img, right_img = frame[:, 0:split_width, :], frame[:, split_width:-1, :]
			# = frame[frame.shape[1]/2+1:][:]
			#print right_img.shape
			if(ret):
				try:
					self.image_left_publisher.publish(self.bridge.cv2_to_imgmsg(left_img, "bgr8"))
					self.image_right_publisher.publish(self.bridge.cv2_to_imgmsg(right_img, "bgr8"))
				except CvBridgeError as e:
					print(e)
			self.rate.sleep()



def main():
	rospy.init_node('image_grabber')
	ig = image_grabber('/home/advaith/Downloads/stereo.avi', 24)
	print("init node")
	print("created grabber")
	ig.grab_and_send()
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting image_grabber down")
	cv2.destroyAllWindows()


main()





