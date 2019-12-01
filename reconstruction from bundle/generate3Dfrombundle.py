#!/usr/bin/env python
# coding: utf-8

# In[58]:


import cv2
vidcap = cv2.VideoCapture('stereo_1.avi')
success,image = vidcap.read()
count = 0
success = True
l_image = image[:,:320,:]
r_image = image[:,320:,:]
print(l_image.shape,r_image.shape)


# In[59]:


while success:
  success,image = vidcap.read()
  print(success)
  l_image = image[:,:320,:]
  r_image = image[:,320:,:]
  if count % 10 == 0:
      cv2.imwrite("left/frame%d.jpg" % count, l_image)     # save frame as JPEG file
      cv2.imwrite("right/frame%d.jpg" % count, r_image)     # save frame as JPEG file
      print(count/10)
  #if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      #break
  count += 1


# In[66]:


import numpy as np
from numpy.linalg import inv, pinv
import cv2
from time import sleep
import timeit

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def featureTracking(img_1, img_2, p1):

    lk_params = dict( winSize  = (11,11),
                      maxLevel = 3,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p2, st, err = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None, **lk_params)
    st = st.reshape(st.shape[0])
    ##find good one
    p1 = p1[st==1]
    p2 = p2[st==1]

    return p1,p2

def featureDetection():
    thresh = dict(threshold=25, nonmaxSuppression=True);
    fast = cv2.FastFeatureDetector_create(**thresh)
    return fast

def write_ply(fn, verts, colors):
    print(verts.shape, colors.shape)
    verts = np.hstack([verts.T, colors.T])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')

def calculate_disparity(left_img, right_img):

        window_size = 11
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
                                        numDisparities=num_disp,
                                        blockSize = 16,
                                        P1 = 8*3*window_size**2,
                                        P2 = 32*3*window_size**2,
                                        disp12MaxDiff = 1,
                                        uniquenessRatio = 10,
                                        speckleWindowSize = 100,
                                        speckleRange = 32
                                )
        disparity = stereo.compute(imgL,imgR).astype(np.float32) / 16.0
        return disparity

def  Transformation(p_C_points, intensities,R,t):#, f, frame_id):
    # Transform from camera frame to world frame
        R_C_frame = np.array([[0, -1, 0],[0, 0, -1], [1, 0, 0]])
        xlims = [7, 20]
        ylims = [-6, 10]
        zlims = [-5, 5]
        p_F_points = inv(R_C_frame).dot(p_C_points)

        mask = (
            (p_F_points[0, :] > xlims[0]) & (p_F_points[0, :] < xlims[1]) &
            (p_F_points[1, :] > ylims[0]) & (p_F_points[1, :] < ylims[1]) &
            (p_F_points[2, :] > zlims[0]) & (p_F_points[2, :] < zlims[1])
        )

        p_F_points = p_F_points[:,mask]
        intensities = intensities[mask]

        #ss = f[frame_id].strip().split()
        #T_W_C = [float(ele) for ele in ss]
        #print(T_W_C)
        #T_W_C = np.array(T_W_C).reshape(3,4)#3*4

        #print(T_W_C)
        P_matrix = np.hstack((R_C_frame, np.zeros((3,1))))#3*4
        P_matrix = np.vstack((P_matrix, np.zeros((1,4))))#4*4
        P_matrix[3,3] = 1
        
        
        T_W_C = np.hstack((R,t))
        T_W_F = T_W_C.dot(P_matrix)
        #print(T_W_F)
        points = T_W_F[:,0:3].dot(p_F_points) + T_W_F[:,3].reshape(3,1)

        return points, intensities

def disparityToPointCloud(disp_img, K, baseline, left_img):
        # points should be 3xN and intensities 1xN, where N is the amount of pixels
        # which have a valid disparity. I.e., only return points and intensities
        # for pixels of left_img which have a valid disparity estimate! The i-th
        # intensity should correspond to the i-th point.
        h, w = disp_img.shape

        h_ = np.linspace(1, h, h)
        w_ = np.linspace(1, w, w)
        [X,Y] = np.meshgrid(w_,h_);

        px_left =  np.stack((Y.reshape(-1), X.reshape(-1), np.ones((h*w))))

        disp_im = disp_img.reshape(-1)

        px_left = px_left[:, disp_im > 0]
        temp  = disp_im[disp_im > 0]

        # Switch from (row, col, 1) to (u, v, 1)
        px_left[0:2, :] = np.flipud(px_left[0:2, :])

        bv_left = inv(K).dot(px_left)

        f = K[0,0]

        x = f*baseline/temp
        points = bv_left*x

        intensities = left_img.reshape(-1)[disp_im > 0]

        return points, intensities


#change to your data path
baseline = 0.54
left_imgs = '../left/frame%d.jpg'
rhgt_imgs = '../right/frame%d.jpg'
#poses 	  = '../data/poses.txt'

K = np.array([[381.914307, 0, 168.108963],
             [0.000000,383.914307,126.979446],
             [0, 0, 1]]
             )
fc = 381.914307
pp = (168.108963, 126.979446)
fc = fc/2
pp = pp/2
#ground_truth = None
#with open(poses) as f:
  #ground_truth = f.readlines()

K[0:2, :] = K[0:2, :] / 2;

min_disp = 4
num_disp = 52 - min_disp
window_size = 4
image0_L =  cv2.resize(cv2.imread("left/frame0.jpg" ,0), (0,0), fx = 0.5, fy = 0.5)
image0_R =  cv2.resize(cv2.imread("right/frame0.jpg" ,0), (0,0), fx = 0.5, fy = 0.5)
disparity = calculate_disparity(image0_L, image0_R)
h, w = image0_L.shape
points, colors = disparityToPointCloud(disparity, K, baseline, image0_L)
if len(imgL.shape) == 2:
    colors = np.dstack([colors]*3)
if (len(colors.shape) == 3):
    colors = np.squeeze(colors, axis=0).T
dis = disparity.reshape(-1)
all_points = points
all_colors = colors
disp = (disparity - min_disp)/num_disp
write_ply(out_fn, all_points, all_colors)
print('Done')

image1_L =  cv2.resize(cv2.imread("left/frame10.jpg" ,0), (0,0), fx = 0.5, fy = 0.5)
#image1_R =  cv2.resize(cv2.imread("right/frame0.jpg" ,0), (0,0), fx = 0.5, fy = 0.5)
detector = featureDetection()
MIN_NUM_FEAT  = 1000
kp1      = detector.detect(image0_L)
p1       = np.array([ele.pt for ele in kp1],dtype='float32')
p1, p2   = featureTracking(image0_L, image1_L, p1)
E, mask = cv2.findEssentialMat(p2, p1, fc, pp, cv2.RANSAC,0.999,1.0)
_, R, t, mask = cv2.recoverPose(E, p2, p1,focal=fc, pp = pp)
preFeature = p2
preImage   = image1_L 
R_f = R
t_f = t
print('generating 3d point cloud...',)
out_fn = 'out.ply'


# In[67]:


maxNum = 15
for i in range(2, maxNum):

    imgL = cv2.resize(cv2.imread("left/frame%d.jpg" %int(i*10),0), (0,0), fx = 0.5, fy = 0.5)
    imgR = cv2.resize(cv2.imread("right/frame%d.jpg" %int(i*10),0), (0,0), fx = 0.5, fy = 0.5)
    disparity = calculate_disparity(imgL, imgR)
    h, w = imgL.shape
    points, colors = disparityToPointCloud(disparity, K, baseline, imgL)
    
    #detect keypoint
    kp1 = detector.detect(imgL)
    if (len(preFeature) < MIN_NUM_FEAT):
        feature   = detector.detect(preImage)
        preFeature = np.array([ele.pt for ele in feature],dtype='float32')
        
    preFeature, curFeature = featureTracking(preImage, imgL, preFeature)
    E, mask = cv2.findEssentialMat(curFeature, preFeature, fc, pp, cv2.RANSAC,0.999,1.0)
    _, R, t, mask = cv2.recoverPose(E, curFeature, preFeature, focal=fc, pp = pp)
    points, colors = Transformation(points, colors,R,t)

    if len(imgL.shape) == 2:
        colors = np.dstack([colors]*3)
    if (len(colors.shape) == 3):
        colors = np.squeeze(colors, axis=0).T
    #colors = colors.reshape(3, -1)

    dis = disparity.reshape(-1)
    all_points = np.hstack((all_points, points))
    all_colors = np.hstack((all_colors, colors))
    #print(all_points.shape)

    disp = (disparity - min_disp)/num_disp
    preImage = imgL
    preFeature = curFeature


write_ply(out_fn, all_points, all_colors)
print('Done')


# In[ ]:




