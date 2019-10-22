
import torch.nn as nn
import torch
import cv2

##
# Bilinear Sampling from from https://github.com/alwynmathew/bilinear-sampler-pytorch/blob/master/bilinear.py
##
def image_warp(img, depth, padding_mode='zeros'):
  # img: the source image (where to sample pixels) -- [B, 3, H, W]
  # depth: depth map of the target image -- [B, 1, H, W]
  # Returns: Source image warped to the target image

  b, _, h, w = depth.size()
  i_range = torch.autograd.Variable(torch.linspace(-1.0, 1.0).view(1, h, 1).expand(1, h, w),
                                    requires_grad=False)  # [1, H, W]  copy 0-height for w times : y coord
  j_range = torch.autograd.Variable(torch.linspace(-1.0, 1.0).view(1, 1, w).expand(1, h, w),
                                    requires_grad=False)  # [1, H, W]  copy 0-width for h times  : x coord

  pixel_coords = torch.stack((j_range, i_range), dim=1).float().cuda()  # [1, 2, H, W]
  batch_pixel_coords = pixel_coords[:, :, :, :].expand(b, 2, h, w).contiguous().view(b, 2, -1)  # [B, 2, H*W]

  X = batch_pixel_coords[:, 0, :] + depth.contiguous().view(b, -1)  # [B, H*W]
  Y = batch_pixel_coords[:, 1, :]

  X_norm = X
  Y_norm = Y

  pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
  pixel_coords = pixel_coords.view(b, h, w, 2)  # [B, H, W, 2]

  projected_img = torch.nn.functional.grid_sample(img, pixel_coords, padding_mode=padding_mode)

  return projected_img

class DepthModel(nn.Module):
  def __init__(self):
    super(DepthModel, self).__init__()

    self.identity = nn.Identity()
    #Encoder
    self.conv1_1 = nn.Conv2d(3, 64, 3, 1)
    self.relu1_1 = nn.ReLU()
    self.batchnorm_1_1 = nn.BatchNorm2d(64, 1e-3)

    self.conv1_2 = nn.Conv2d(64, 64, 3, 1)
    self.relu1_2 = nn.ReLU()
    self.batchnorm_1_2 = nn.BatchNorm2d(64, 1e-3)

    self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

    self.conv2_1 = nn.Conv2d(64, 128, 3, 1)
    self.relu2_1 = nn.ReLU()
    self.batchnorm_2_1 = nn.BatchNorm2d(128, 1e-3)

    self.conv2_2 = nn.Conv2d(128, 128, 3, 1)
    self.relu2_2 = nn.ReLU()
    self.batchnorm_2_2 = nn.BatchNorm2d(128, 1e-3)

    self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

    self.conv3_1 = nn.Conv2d(128, 256, 3, 1)
    self.relu3_1 = nn.ReLU()
    self.batchnorm_3_1 = nn.BatchNorm2d(256, 1e-3)

    self.conv3_2 = nn.Conv2d(256, 256, 3, 1)
    self.relu3_2 = nn.ReLU()
    self.batchnorm_3_2 = nn.BatchNorm2d(256, 1e-3)

    self.conv3_3 = nn.Conv2d(256, 256, 3, 1)
    self.relu3_3 = nn.ReLU()
    self.batchnorm_3_3 = nn.BatchNorm2d(256, 1e-3)

    self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

    self.conv4_1 = nn.Conv2d(256, 512, 3, 1)
    self.relu4_1 = nn.ReLU()
    self.batchnorm_4_1 = nn.BatchNorm2d(512, 1e-3)

    self.conv4_2 = nn.Conv2d(512, 512, 3, 1)
    self.relu4_2 = nn.ReLU()
    self.batchnorm_4_2 = nn.BatchNorm2d(512, 1e-3)

    self.conv4_3 = nn.Conv2d(512, 512, 3, 1)
    self.relu4_3 = nn.ReLU()
    self.batchnorm_4_3 = nn.BatchNorm2d(512, 1e-3)

    self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

    #Decoder
    self.deconv5_1 = nn.ConvTranspose2d(512, 512, 3, 1)
    self.relud_5_1 = nn.ReLU()
    self.batchnormd_5_1= nn.BatchNorm2d(512, 1e-3)

    self.unpool4 = nn.MaxUnpool2d(2, 2)
    self.deconv4_1 = nn.ConvTranspose2d(1024, 512, 3, 1)
    self.relud_4_1 = nn.ReLU()
    self.batchnormd_4_1 = nn.BatchNorm2d(512, 1e-3)

    self.deconv4_2 = nn.ConvTranspose2d(512, 512, 3, 1)
    self.relud_4_2 = nn.ReLU()
    self.batchnormd_4_2 = nn.BatchNorm2d(512, 1e-3)

    self.deconv4_3 = nn.ConvTranspose2d(512, 256, 3, 1)
    self.relud_4_3 = nn.ReLU()
    self.batchnormd_4_3 = nn.BatchNorm2d(256, 1e-3)

    self.unpool3 = nn.MaxUnpool2d(2, 2)
    self.deconv3_1 = nn.ConvTranspose2d(512, 256, 3, 1)
    self.relud_3_1 = nn.ReLU()
    self.batchnormd_3_1 = nn.BatchNorm2d(256, 1e-3)

    self.deconv3_2 = nn.ConvTranspose2d(256, 256, 3, 1)
    self.relud_3_2 = nn.ReLU()
    self.batchnormd_3_2 = nn.BatchNorm2d(256, 1e-3)

    self.deconv3_3 = nn.ConvTranspose2d(256, 128, 3, 1)
    self.relud_3_3 = nn.ReLU()
    self.batchnormd_3_3 = nn.BatchNorm2d(128, 1e-3)

    self.unpool2 = nn.MaxUnpool2d(2, 2)
    self.deconv2_1 = nn.ConvTranspose2d(256, 128, 3, 1)
    self.relud_2_1 = nn.ReLU()
    self.batchnormd_2_1 = nn.BatchNorm2d(128, 1e-3)

    self.deconv2_2 = nn.ConvTranspose2d(128, 64, 3, 1)
    self.relud_2_2 = nn.ReLU()
    self.batchnormd_2_2 = nn.BatchNorm2d(64, 1e-3)

    self.unpool1 = nn.MaxUnpool2d(2, 2)
    self.deconv1_1 = nn.ConvTranspose2d(128, 64, 3, 1)
    self.relud_1_1 = nn.ReLU()
    self.batchnormd_1_1 = nn.BatchNorm2d(128, 1e-3)

    self.deconv1_2 = nn.ConvTranspose2d(64, 3, 3, 1)
    self.relud_1_2 = nn.ReLU()
    self.batchnormd_1_2 = nn.BatchNorm2d(3, 1e-3)

    self.disp2_deconv = nn.ConvTranspose2d(64, 3, 3, 1)
    self.disp2_relu = nn.ReLU()
    self.disp2_batchnorm = nn.BatchNorm2d(3, 1e-3)

    self.disp1_conv = nn.Conv2d(3, 1, 3, 1)
    self.disp2_conv = nn.Conv2d(3, 1, 3, 1)
    self.disp1_sigmoid = nn.Sigmoid()
    self.disp2_sigmoid = nn.Sigmoid()


  def forward(self, x):

    #Encoder
    x = self.identity(x)
    conv1_1_output = self.batchnorm_1_1(self.relu1_1(self.conv1_1(x)))
    conv1_2_output = self.batchnorm_1_2(self.relu1_2(self.conv1_2(conv1_1_output)))
    pool1_output, indices1 = self.pool1(conv1_2_output)
    conv2_1_output = self.batchnorm_2_1(self.relu2_1(self.conv2_1(pool1_output)))
    conv2_2_output = self.batchnorm_2_2(self.relu2_2(self.conv2_2(conv2_1_output)))
    pool2_output, indices2 = self.pool2(conv2_2_output)

    conv3_1_output = self.batchnorm_3_1(self.relu3_1(self.conv3_1(pool2_output)))
    conv3_2_output = self.batchnorm_3_2(self.relu3_2(self.conv3_2(conv3_1_output)))
    conv3_3_output = self.batchnorm_3_3(self.relu3_3(self.conv3_3(conv3_2_output)))

    pool3_output, indices3 = self.pool3(conv3_3_output)

    conv4_1_output = self.batchnorm_4_1(self.relu4_1(self.conv4_1(pool3_output)))
    conv4_2_output = self.batchnorm_4_2(self.relu4_2(self.conv4_2(conv4_1_output)))
    conv4_3_output = self.batchnorm_4_3(self.relu4_3(self.conv4_3(conv4_2_output)))

    pool4_output, indices4 = self.pool4(conv4_3_output)

    #Decoder
    deconv_5_1_output = self.batchnormd_5_1(self.relud_5_1(self.deconv5_1(pool4_output)))

    unpool4_output = self.unpool4(deconv_5_1_output, indices4)
    join4_output = torch.cat([unpool4_output, conv4_3_output], 2)

    deconv4_1_output = self.batchnormd_4_1(self.relud_4_1(self.deconv4_1(join4_output)))
    deconv4_2_output = self.batchnormd_4_2(self.relud_4_2(self.deconv4_2(deconv4_1_output)))
    deconv4_3_output = self.batchnormd_4_3(self.relud_4_1(self.deconv4_3(deconv4_2_output)))

    unpool3_output = self.unpool3(deconv4_3_output, indices3)
    join3_output = torch.cat([unpool3_output, conv3_3_output], 2)

    deconv3_1_output = self.batchnormd_3_1(self.relud_3_1(self.deconv3_1(join3_output)))
    deconv3_2_output = self.batchnormd_3_2(self.relud_3_2(self.deconv3_2(deconv3_1_output)))
    deconv3_3_output = self.batchnormd_3_3(self.relud_3_3(self.deconv3_3(deconv3_2_output)))

    unpool2_output = self.unpool2(deconv3_3_output, indices2)
    join2_output = torch.cat([unpool2_output, conv2_2_output], 2)

    deconv2_1_output = self.batchnormd_2_1(self.relud_2_1(self.deconv2_1(join2_output)))
    deconv2_2_output = self.batchnormd_2_2(self.relud_2_2(self.deconv2_2(deconv2_1_output)))

    # deconv2_2_output_disp2 = self.disp2_batchnorm(self.disp2_relu(self.disp2_deconv(deconv2_2_output)))
    # disp2 = self.disp2_sigmoid(self.disp2_conv(deconv2_2_output_disp2))

    unpool1_output = self.unpool1(deconv2_2_output, indices1)
    join1_output = torch.cat([unpool1_output, conv1_2_output], 2)

    deconv1_1_output = self.batchnorm_1_1(self.relu1_1(self.deconv1_1(join1_output)))
    deconv1_2_output = self.batchnorm_1_2(self.relu1_2(self.deconv1_2(deconv1_1_output)))

    disp1 = self.disp1_sigmoid(self.disp1_conv(deconv1_2_output))

    return disp1


class SiameseDepthModel(nn.Module):
  def __init__(self, width, height, focal_length, baseline):
    super(SiameseDepthModel, self).__init__()
    self.depth_model = DepthModel()
    self.width = width
    self.heigt = height
    self.focal_length = focal_length
    self.baseline = baseline

  def forward(self, l_image, r_image):
    disp1_l = -1 * self.depth_model.forward(l_image)
    disp1_r = self.depth_model.forward(r_image)

    projected_img_l = image_warp(r_image, disp1_l)
    projected_img_r = image_warp(l_image, disp1_r)

    self.depth_l = self.focal_length * self.baseline / disp1_l
    self.depth_r = self.focal_length * self.baseline / disp1_r

    return [projected_img_l, projected_img_r]

  def get_depth_imgs(self):
    depth_l = torch.Tensor.cpu(self.depth_l).detach().numpy()[:,:,:,-1]
    depth_r = torch.Tensor.cpu(self.depth_r).detach().numpy()[:,:,:,-1]
    depth_l_img = cv2.normalize(depth_l, depth_l, 0, 255, cv.NORM_MINMAX)
    depth_r_img = cv2.normalize(depth_r, depth_r, 0, 255, cv.NORM_MINMAX)
    return  depth_l_img, depth_r_img

# width = 384
# height = 192
# focal_length = ((373.47833252)**2 + (373.47833252)**2)**0.5
# baseline = -5.63117313
# model = SiameseDepthModel(width, height, focal_length, base)
# print(model.state_dict())
# print(len(model.state_dict()))