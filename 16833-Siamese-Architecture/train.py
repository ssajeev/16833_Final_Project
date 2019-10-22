import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import libcudnn


from endoscopic_dataset import *
from siamese_model import *

ROOT_DIR = 'data/images/'
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 0.001



class SSI_loss():
  def forward(self, output, target):
    y = output
    x = target

    C1 = 6.5025
    C2 = 58.5225

    mu_x = torch.AvgPool2d(x, requires_grad=True)
    mu_y = torch.AvgPool2d(y, requires_grad=True)
    mu_x_sq = torch.dot(mu_x, mu_x)
    mu_y_sq = torch.dot(mu_y, mu_y)
    mu_xy = torch.dot(mu_x, mu_y)
    X_2 = torch.dot(x, x)
    Y_2 = torch.dot(y, y)
    XY = torch.dot(x, y)
    sigma_x_sq = torch.AvgPool2d(X_2, requires_grad=True) - mu_x_sq
    sigma_y_sq = torch.AvgPool2d(Y_2, requires_grad=True) - mu_y_sq
    sigma_xy = torch.AvgPool2d(XY, requires_grad=True) - mu_xy
    A1 = mu_xy * 2 + C1
    A2 = sigma_xy * 2 + C2
    B1 = mu_x_sq + mu_y_sq + C1
    B2 = sigma_x_sq + sigma_y_sq + C2
    A = torch.dot(A1, A2)
    B = torch.dot(B1, B2)
    app_loss = (1 - torch.mean(torch.div(A, B)))
    self.app_loss_shape = np.shape(app_loss)
    self.app_loss = app_loss
    return app_loss

  def backward(self):
    return self.app_loss.backward(self.app_loss_shape)

def main():
  # cudnn = libcudnn.cudnnCreate()
  #
  # # use_cuda = torch.cuda.is_available()
  # # device = torch.device("cuda:0" if use_cuda else "cpu")
  # cudnn.benchmark = True

  train_dataset = EndoscopicDataset(csv_file='train.csv',
                                           root_dir='/Users/Sandra/Downloads/daVinci/train/',
                                           transform=transforms.Compose([

                                               # transforms.RandomCrop(224),
                                               transforms.ToTensor()
                                           ]))

  # test_dataset = EndoscopicDataset(csv_file='test.csv',
  #                                   root_dir='/Users/Sandra/Downloads/daVinci/test/',
  #                                   transform=transforms.Compose([
  #                                     transforms.Rescale(256),
  #                                     transforms.RandomCrop(224),
  #                                     transforms.ToTensor()
  #                                   ]))

  dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=NUM_WORKERS)

  width = 384
  height = 192
  focal_length = ((373.47833252)**2 + (373.47833252)**2)**0.5
  baseline = -5.63117313

  model = SiameseDepthModel(width, height, focal_length, baseline)
  criterion = SSI_loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  for epoch in range(EPOCHS):
    for batch_input in dataset_loader:

      optimizer.zero_grad()
      outputs = model(batch_input["image_l"], batch_input["image_r"])
      err = criterion.forward(outputs, [batch_input["image_l"], batch_input["image_r"]])
      dparams = criterion.backward()
      optimizer.step()





  pass


if __name__ == '__main__':
  main()