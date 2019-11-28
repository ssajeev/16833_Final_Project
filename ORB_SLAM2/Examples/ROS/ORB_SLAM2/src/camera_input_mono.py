#! /usr/bin/env python

 
import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_grabber:
	def __init__(self, rate, path = ""):
		self.image_publisher = rospy.Publisher("/camera/image_raw", Image, queue_size=10)
		self.bridge = CvBridge()
		if(path == ""):
			self.cap = cv2.VideoCapture(2)
		else:
			self.cap = cv2.VideoCapture(path)
		self.rate = rospy.Rate(rate)
		print("created cap")

	def grab_and_send(self):
		print("in grab and send")
		if(not self.cap.isOpened()):
			print("Error in video file path")
		while(self.cap.isOpened()):
			ret, frame = self.cap.read()
			if(ret):
				try:
					self.image_publisher.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
					cv2.imshow("Camera", frame)
				except CvBridgeError as e:
					print(e)



def main():
	rospy.init_node('camera_streamer')
	ig = image_grabber(24, "/home/advaith/Desktop/robosub_vid/right_log1.avi")
	print("init node")
	print("created grabber")
	#ig.grab_and_send()
	print ig.cap.isOpened()
	while(ig.cap.isOpened()):
		try:
			ret, frame = ig.cap.read()
			if(ret):
				try:
					ig.image_publisher.publish(ig.bridge.cv2_to_imgmsg(frame, "bgr8"))
					cv2.imshow("Camera", frame)
					cv2.waitKey(1)
				except CvBridgeError as e:
					print(e)
			ig.rate.sleep()
		except KeyboardInterrupt:
			ig.cap.release()
			cv2.destroyAllWindows()
			print("Shutting image_grabber down")
	


main()





