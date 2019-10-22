
import torch.nn as nn
import torch
import cv2
import numpy as np
from torch.nn.functional import pad

def apply_disparity(input_images, x_offset, wrap_mode='border', tensor_type='torch.cuda.FloatTensor'):
    num_batch, num_channels, height, width = input_images.size()

    # Handle both texture border types
    edge_size = 0
    if wrap_mode == 'border':
      edge_size = 1
      # Pad last and second-to-last dimensions by 1 from both sides
      input_images = pad(input_images, (1, 1, 1, 1))
    elif wrap_mode == 'edge':
      edge_size = 0
    else:
      return None

    # Put channels to slowest dimension and flatten batch with respect to others
    input_images = input_images.permute(1, 0, 2, 3).contiguous()
    im_flat = input_images.view(num_channels, -1)

    # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
    # meshgrid function)
    x = torch.linspace(0, width - 1, width).repeat(height, 1)  #.type(tensor_type).to(opt.gpu_ids)
    y = torch.linspace(0, height - 1, height).repeat(width, 1) #.transpose(0, 1)  #.type(tensor_type).to(opt.gpu_ids)
    # Take padding into account
    x = x + edge_size
    y = y + edge_size
    # Flatten and repeat for each image in the batch
    x = x.view(-1).repeat(1, num_batch)
    y = y.view(-1).repeat(1, num_batch)

    # Now we want to sample pixels with indicies shifted by disparity in X direction
    # For that we convert disparity from % to pixels and add to X indicies
    x = x + x_offset.contiguous().view(-1) * width
    # Make sure we don't go outside of image
    x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
    # Round disparity to sample from integer-valued pixel grid
    y0 = torch.floor(y)
    # In X direction round both down and up to apply linear interpolation
    # between them later
    x0 = torch.floor(x)
    x1 = x0 + 1
    # After rounding up we might go outside the image boundaries again
    x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

    # Calculate indices to draw from flattened version of image batch
    dim2 = (width + 2 * edge_size)
    dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
    # Set offsets for each image in the batch
    base = dim1 * torch.arange(num_batch)  #.type(tensor_type).to(opt.gpu_ids)
    base = base.view(-1, 1).repeat(1, height * width).view(-1)
    # One pixel shift in Y  direction equals dim2 shift in flattened array
    base_y0 = base + y0 * dim2
    # Add two versions of shifts in X direction separately
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    # Sample pixels from images
    pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
    pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

    # Apply linear interpolation to account for fractional offsets
    weight_l = x1 - x
    weight_r = x - x0
    output = weight_l * pix_l + weight_r * pix_r

    # Reshape back into image batch and permute back to (N,C,H,W) shape
    output = output.view(num_channels, num_batch, height, width).permute(1, 0, 2, 3)

    return output

class DepthModel(nn.Module):
  def __init__(self):
    super(DepthModel, self).__init__()

    self.identity = nn.Identity()
    #Encoder
    self.conv1_1 = nn.Conv2d(3, 64, 3, 1)
    self.relu1_1 = nn.ReLU(True)
    self.batchnorm_1_1 = nn.BatchNorm2d(64, 1e-3)

    self.conv1_2 = nn.Conv2d(64, 64, 3, 1)
    self.relu1_2 = nn.ReLU(True)
    self.batchnorm_1_2 = nn.BatchNorm2d(64, 1e-3)

    self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

    self.conv2_1 = nn.Conv2d(64, 128, 3, 1)
    self.relu2_1 = nn.ReLU(True)
    self.batchnorm_2_1 = nn.BatchNorm2d(128, 1e-3)

    self.conv2_2 = nn.Conv2d(128, 128, 3, 1)
    self.relu2_2 = nn.ReLU(True)
    self.batchnorm_2_2 = nn.BatchNorm2d(128, 1e-3)

    self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

    self.conv3_1 = nn.Conv2d(128, 256, 3, 1)
    self.relu3_1 = nn.ReLU(True)
    self.batchnorm_3_1 = nn.BatchNorm2d(256, 1e-3)

    self.conv3_2 = nn.Conv2d(256, 256, 3, 1)
    self.relu3_2 = nn.ReLU(True)
    self.batchnorm_3_2 = nn.BatchNorm2d(256, 1e-3)

    self.conv3_3 = nn.Conv2d(256, 256, 3, 1)
    self.relu3_3 = nn.ReLU(True)
    self.batchnorm_3_3 = nn.BatchNorm2d(256, 1e-3)

    self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

    self.conv4_1 = nn.Conv2d(256, 512, 3, 1)
    self.relu4_1 = nn.ReLU(True)
    self.batchnorm_4_1 = nn.BatchNorm2d(512, 1e-3)

    self.conv4_2 = nn.Conv2d(512, 512, 3, 1)
    self.relu4_2 = nn.ReLU(True)
    self.batchnorm_4_2 = nn.BatchNorm2d(512, 1e-3)

    self.conv4_3 = nn.Conv2d(512, 512, 3, 1)
    self.relu4_3 = nn.ReLU(True)
    self.batchnorm_4_3 = nn.BatchNorm2d(512, 1e-3)

    self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

    #Decoder
    self.deconv5_1 = nn.Conv2d(512, 512, 1, 1)
    self.relud_5_1 = nn.ReLU(True)
    self.batchnormd_5_1= nn.BatchNorm2d(512, 1e-3)

    self.unpool4 = nn.MaxUnpool2d(2, 2)
    self.deconv4_1 = nn.ConvTranspose2d(1024, 512, 3, 1)
    self.relud_4_1 = nn.ReLU(True)
    self.batchnormd_4_1 = nn.BatchNorm2d(512, 1e-3)

    self.deconv4_2 = nn.ConvTranspose2d(512, 512, 3, 1)
    self.relud_4_2 = nn.ReLU(True)
    self.batchnormd_4_2 = nn.BatchNorm2d(512, 1e-3)

    self.deconv4_3 = nn.ConvTranspose2d(512, 256, 3, 1)
    self.relud_4_3 = nn.ReLU(True)
    self.batchnormd_4_3 = nn.BatchNorm2d(256, 1e-3)

    self.unpool3 = nn.MaxUnpool2d(2, 2)
    self.deconv3_1 = nn.ConvTranspose2d(512, 256, 3, 1)
    self.relud_3_1 = nn.ReLU(True)
    self.batchnormd_3_1 = nn.BatchNorm2d(256, 1e-3)

    self.deconv3_2 = nn.ConvTranspose2d(256, 256, 3, 1)
    self.relud_3_2 = nn.ReLU(True)
    self.batchnormd_3_2 = nn.BatchNorm2d(256, 1e-3)

    self.deconv3_3 = nn.ConvTranspose2d(256, 128, 3, 1)
    self.relud_3_3 = nn.ReLU(True)
    self.batchnormd_3_3 = nn.BatchNorm2d(128, 1e-3)

    self.unpool2 = nn.MaxUnpool2d(2, 2)
    self.deconv2_1 = nn.ConvTranspose2d(256, 128, 3, 1)
    self.relud_2_1 = nn.ReLU(True)
    self.batchnormd_2_1 = nn.BatchNorm2d(128, 1e-3)

    self.deconv2_2 = nn.ConvTranspose2d(128, 64, 3, 1)
    self.relud_2_2 = nn.ReLU(True)
    self.batchnormd_2_2 = nn.BatchNorm2d(64, 1e-3)

    self.unpool1 = nn.MaxUnpool2d(2, 2)
    self.deconv1_1 = nn.ConvTranspose2d(128, 64, 3, 1)
    self.relud_1_1 = nn.ReLU(True)
    self.batchnormd_1_1 = nn.BatchNorm2d(64, 1e-3)

    self.deconv1_2 = nn.ConvTranspose2d(64, 3, 3, 1)
    self.relud_1_2 = nn.ReLU(True)
    self.batchnormd_1_2 = nn.BatchNorm2d(3, 1e-3)

    self.disp2_deconv = nn.ConvTranspose2d(64, 3, 3, 1)
    self.disp2_relu = nn.ReLU(True)
    self.disp2_batchnorm = nn.BatchNorm2d(3, 1e-3)

    self.disp1_conv = nn.Conv2d(3, 1, 3, 1, 1)
    self.disp2_conv = nn.Conv2d(3, 1, 3, 1, 1)
    self.disp1_sigmoid = nn.Sigmoid()
    self.disp2_sigmoid = nn.Sigmoid()


  def forward(self, x):

    #Encoder
    x = self.identity(x)
    conv1_1_output = self.relu1_1(self.batchnorm_1_1(self.conv1_1(x)))
    conv1_2_output = self.relu1_2(self.batchnorm_1_2(self.conv1_2(conv1_1_output)))
    pool1_output, indices1 = self.pool1(conv1_2_output)
    conv2_1_output = self.relu2_1(self.batchnorm_2_1(self.conv2_1(pool1_output)))
    conv2_2_output = self.relu2_2(self.batchnorm_2_2(self.conv2_2(conv2_1_output)))
    pool2_output, indices2 = self.pool2(conv2_2_output)

    conv3_1_output = self.relu3_1(self.batchnorm_3_1(self.conv3_1(pool2_output)))
    conv3_2_output = self.relu3_2(self.batchnorm_3_2(self.conv3_2(conv3_1_output)))
    conv3_3_output = self.relu3_3(self.batchnorm_3_3(self.conv3_3(conv3_2_output)))

    pool3_output, indices3 = self.pool3(conv3_3_output)

    conv4_1_output = self.relu4_1(self.batchnorm_4_1(self.conv4_1(pool3_output)))
    conv4_2_output = self.relu4_2(self.batchnorm_4_2(self.conv4_2(conv4_1_output)))
    conv4_3_output = self.relu4_3(self.batchnorm_4_3(self.conv4_3(conv4_2_output)))

    pool4_output, indices4 = self.pool4(conv4_3_output)

    #Decoder
    deconv_5_1_output = self.relud_5_1(self.batchnormd_5_1(self.deconv5_1(pool4_output)))

    unpool4_output = self.unpool4(deconv_5_1_output, indices4, conv4_3_output.size())
    join4_output = torch.cat([unpool4_output, conv4_3_output], 1)

    deconv4_1_output = self.batchnormd_4_1(self.relud_4_1(self.deconv4_1(join4_output)))
    deconv4_2_output = self.batchnormd_4_2(self.relud_4_2(self.deconv4_2(deconv4_1_output)))
    deconv4_3_output = self.batchnormd_4_3(self.relud_4_3(self.deconv4_3(deconv4_2_output)))

    unpool3_output = self.unpool3(deconv4_3_output, indices3, conv3_3_output.size())
    join3_output = torch.cat([unpool3_output, conv3_3_output], 1)

    deconv3_1_output = self.batchnormd_3_1(self.relud_3_1(self.deconv3_1(join3_output)))
    deconv3_2_output = self.batchnormd_3_2(self.relud_3_2(self.deconv3_2(deconv3_1_output)))
    deconv3_3_output = self.batchnormd_3_3(self.relud_3_3(self.deconv3_3(deconv3_2_output)))

    unpool2_output = self.unpool2(deconv3_3_output, indices2, conv2_2_output.size())
    join2_output = torch.cat([unpool2_output, conv2_2_output], 1)

    deconv2_1_output = self.batchnormd_2_1(self.relud_2_1(self.deconv2_1(join2_output)))
    deconv2_2_output = self.batchnormd_2_2(self.relud_2_2(self.deconv2_2(deconv2_1_output)))

    # deconv2_2_output_disp2 = self.disp2_batchnorm(self.disp2_relu(self.disp2_deconv(deconv2_2_output)))
    # disp2 = self.disp2_sigmoid(self.disp2_conv(deconv2_2_output_disp2))

    unpool1_output = self.unpool1(deconv2_2_output, indices1, conv1_2_output.size())
    join1_output = torch.cat([unpool1_output, conv1_2_output], 1)

    deconv1_1_output = self.batchnormd_1_1(self.relu1_1(self.deconv1_1(join1_output)))
    deconv1_2_output = self.batchnormd_1_2(self.relu1_2(self.deconv1_2(deconv1_1_output)))

    disp1 = self.disp1_sigmoid(self.disp1_conv(deconv1_2_output))

    return disp1


class SiameseDepthModel(nn.Module):
  def __init__(self, width, height, focal_length, baseline):
    super(SiameseDepthModel, self).__init__()
    self.depth_model = DepthModel()
    self.width = width
    self.height = height
    self.focal_length = focal_length
    self.baseline = baseline

  def forward(self, l_image, r_image):
    disp1_l = -1 * self.depth_model.forward(l_image)
    disp1_r = self.depth_model.forward(r_image)

    print(disp1_l.size())
    print(disp1_r.size())
    print(r_image.size())
    print(l_image.size())

    self.depth_l = self.focal_length * self.baseline / disp1_l
    self.depth_r = self.focal_length * self.baseline / disp1_r

    projected_img_l = apply_disparity(r_image, disp1_l)
    projected_img_r = apply_disparity(l_image, disp1_r)



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