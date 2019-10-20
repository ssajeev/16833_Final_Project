import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import libcudnn

from endoscopic_dataset import *
from siamese_model import *

ROOT_DIR = 'data/images/'
BATCH_SIZE = 64
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
    mu_x_sq = torch.dot(mu_x, mu_x, requires_grad=True)
    mu_y_sq = torch.dot(mu_y, mu_y, requires_grad=True)
    mu_xy = torch.dot(mu_x, mu_y, requires_grad=True)
    X_2 = torch.dot(x, x, requires_grad=True)
    Y_2 = torch.dot(y, y, requires_grad=True)
    XY = torch.dot(x, y, requires_grad=True)
    sigma_x_sq = torch.AvgPool2d(X_2, requires_grad=True) - mu_x_sq
    sigma_y_sq = torch.AvgPool2d(Y_2, requires_grad=True) - mu_y_sq
    sigma_xy = torch.AvgPool2d(XY, requires_grad=True) - mu_xy
    A1 = mu_xy * 2 + C1
    A2 = sigma_xy * 2 + C2
    B1 = mu_x_sq + mu_y_sq + C1
    B2 = sigma_x_sq + sigma_y_sq + C2
    A = torch.dot(A1, A2)
    B = torch.dot(B1, B2)
    app_loss = (1 - torch.mean(torch.div(A, B), requires_grad=True))
    self.app_loss_shape = np.shape(app_loss)
    self.app_loss = app_loss
    return app_loss

  def backward(self):
    return self.app_loss.backward()

def main():
  cudnn = libcudnn.cudnnCreate()

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  cudnn.benchmark = True

  train_dataset = EndoscopicDataset(csv_file='train.csv',
                                           root_dir='daVinci/train/',
                                           transform=transforms.Compose([
                                               transforms.Rescale(256),
                                               transforms.RandomCrop(224),
                                               transforms.ToTensor()
                                           ]))

  test_dataset = EndoscopicDataset(csv_file='test.csv',
                                    root_dir='daVinci/test/',
                                    transform=transforms.Compose([
                                      transforms.Rescale(256),
                                      transforms.RandomCrop(224),
                                      transforms.ToTensor()
                                    ]))

  dataset_loader = utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=NUM_WORKERS)

  model =  SiameseDepthModel()

  criterion = SSI_loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  for epoch in range(EPOCHS):
    for batch_input, zero in dataset_loader:


      optimizer.zero_grad()
      outputs = model.forward(batch_input)
      err = criterion.forward(outputs, targetTable)
      dparams = criterion.backward()
      model.backward(batch_input, dparams)
      optimizer.step()


      return err

  pass


if __name__ == '__main__':
  main()