
import torch.nn as nn
import torch


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
    conv2_2_output = self.batchnorm_2_2(self.reul2_2(self.conv2_2(conv2_1_output)))
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

    deconv2_2_output_disp2 = self.disp2_batchnorm(self.disp2_relu(self.disp2_deconv(deconv2_2_output)))
    disp2 = self.disp2_sigmoid(self.disp2_conv(deconv2_2_output_disp2))

    unpool1_output = self.unpool1(deconv2_2_output, indices1)
    join1_output = torch.cat([unpool1_output, conv1_2_output], 2)

    deconv1_1_output = self.batchnorm_1_1(self.relu1_1(self.deconv1_1(join1_output)))
    deconv1_2_output = self.batchnorm_1_2(self.relu1_2(self.deconv1_2(deconv1_1_output)))

    disp1 = self.disp1_sigmoid(self.disp1_conv(deconv1_2_output))

    return [disp1, disp2]


class SiameseDepthModel(nn.Module):
  def __init__(self):
    super(SiameseDepthModel, self).__init__()
    self.depth_model = DepthModel()

  def forward(self, l_image, r_image):
    [disp1_l, disp2_l] = self.depth_model.forward(l_image)
    [disp1_r, disp2_r] = self.depth_model.forward(r_image)



    return


