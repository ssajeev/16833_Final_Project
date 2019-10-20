
import torch.nn as nn
import torch


class DepthModel(nn.Module):
  def __init__(self):
    super(DepthModel, self).__init__()

    #Encoder
    self.conv1_1 = nn.Conv2d(3, 64, 3, 1)
    self.relu1 = nn.ReLU
    self.batchnorm_1 = nn.BatchNorm2d(64, 1e-3)

    self.conv1_2 = nn.Conv2d(64, 64, 3, 1)
    self.relu1 = nn.ReLU
    self.batchnorm_1 = nn.BatchNorm2d(64, 1e-3)

    self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
    self.conv2_1 = nn.Conv2d(64, 128, 3, 1)
    self.relu1 = nn.ReLU
    self.batchnorm_1 = nn.BatchNorm2d(64, 1e-3)

    self.conv2_2 = nn.Conv2d(128, 128, 3, 1)
    self.relu1 = nn.ReLU
    self.batchnorm_1 = nn.BatchNorm2d(64, 1e-3)

    self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)
    self.conv3_1 = nn.Conv2d(128, 256, 3, 1)
    self.relu1 = nn.ReLU
    self.batchnorm_1 = nn.BatchNorm2d(64, 1e-3)

    self.conv3_2 = nn.Conv2d(256, 256, 3, 1)
    self.relu1 = nn.ReLU
    self.batchnorm_1 = nn.BatchNorm2d(64, 1e-3)

    self.conv3_3 = nn.Conv2d(256, 256, 3, 1)
    self.relu1 = nn.ReLU
    self.batchnorm_1 = nn.BatchNorm2d(64, 1e-3)

    self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)
    self.conv4_1 = nn.Conv2d(256, 512, 3, 1)
    self.relu1 = nn.ReLU
    self.batchnorm_1 = nn.BatchNorm2d(64, 1e-3)

    self.conv4_2 = nn.Conv2d(512, 512, 3, 1)
    self.conv4_3 = nn.Conv2d(512, 512, 3, 1)
    self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)
    self.conv5_1 = nn.Conv2d(512, 512, 3, 1)
    self.conv5_2 = nn.Conv2d(512, 512, 3, 1)
    self.conv5_3 = nn.Conv2d(512, 512, 3, 1)
    self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)
    self.conv6 = nn.Conv2d(512, 512, 3, 1)

    #Decoder
    self.unpool5 = nn.MaxUnpool2d(2, 2)
    self.deconv5_1 = nn.ConvTranspose2d(512, 512, 3, 1)
    self.deconv5_2 = nn.ConvTranspose2d(512, 512, 3, 1)
    self.deconv5_3 = nn.ConvTranspose2d(512, 512, 3, 1)
    self.unpool4 = nn.MaxUnpool2d(2, 2)
    self.deconv4_3 = nn.ConvTranspose2d(512, 512, 3, 1)
    self.deconv4_2 = nn.ConvTranspose2d(512, 512, 3, 1)
    self.deconv4_1 = nn.ConvTranspose2d(512, 256, 3, 1)
    self.unpool3 = nn.MaxUnpool2d(2, 2)
    self.deconv3_3 = nn.ConvTranspose2d(256, 256, 3, 1)
    self.deconv3_2 = nn.ConvTranspose2d(256, 256, 3, 1)
    self.deconv3_1 = nn.ConvTranspose2d(256, 128, 3, 1)
    self.unpool2 = nn.MaxUnpool2d(2, 2)
    self.deconv2_2 = nn.Conv2d(128, 128, 3, 1)
    self.deconv2_1 = nn.Conv2d(128, 64, 3, 1)



    #Decoder


  def forward(self, x):

    return x

net = DepthModel()


class SiameseDepthModel(nn.Module):
  def __init__(self):
    super(SiameseDepthModel, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 4 * 4)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

net2 = SiameseDepthModel()

