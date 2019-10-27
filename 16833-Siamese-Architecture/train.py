import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#import libcudnn
import datetime
from tqdm import tqdm
from endoscopic_dataset import *
from siamese_model import *

ROOT_DIR = 'data/images/'
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "saved_model_" + datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S") + ".pt"

class SSI_loss(nn.Module):
  def __init__(self):
    super(SSI_loss, self).__init__()

  def forward(self, output, target):
    y = (output)
    x = (target)

    C1 = 6.5025
    C2 = 58.5225

    mu_x = nn.AvgPool2d(3, stride=1)(x)
    mu_y = nn.AvgPool2d(3, stride=1)(y)

    mu_x_sq = (mu_x * mu_x)
    mu_y_sq = (mu_y * mu_y)
    mu_xy = (mu_x * mu_y)
    X_2 = (x * x)
    Y_2 = (y * y)
    XY = (x * y)
    sigma_x_sq = nn.AvgPool2d(3, 1)(X_2) - mu_x_sq
    sigma_y_sq = nn.AvgPool2d(3, 1)(Y_2) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(XY) - mu_xy
    A1 = mu_xy * 2 + C1
    A2 = sigma_xy * 2 + C2
    B1 = mu_x_sq + mu_y_sq + C1
    B2 = sigma_x_sq + sigma_y_sq + C2
    A = (A1 * A2)
    B = (B1 * B2)
    app_loss = (1 - ((A / B)).mean())
    self.app_loss_shape = np.shape(app_loss)
    self.app_loss = app_loss
    return app_loss

  def backward(self):
    return None

def main():
  #cudnn = libcudnn.cudnnCreate()

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:0" if use_cuda else "cpu")
  #cudnn.benchmark = True

  train_dataset = EndoscopicDataset(csv_file='train.csv',
                                           root_dir='daVinci/train/',
                                           transform=transforms.Compose([
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
  model.cuda()
  model = model.train()
  for param in model.parameters():
    param.requires_grad = True

  criterion = SSI_loss()

  for param in criterion.parameters():
    param.requires_grad = True

  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    t = tqdm(iter(dataset_loader), leave=False, total=len(dataset_loader))
    for i, batch_input in enumerate(t):
      left_img, right_img = batch_input["image_l"].to(device), batch_input["image_r"].to(device)
      optimizer.zero_grad()
      outputs = model(left_img, right_img)

      loss_1 = criterion.forward(outputs[0], left_img)
      loss_2 = criterion.forward(outputs[1], right_img)
      loss = loss_1 + loss_2

      loss.backward()
      optimizer.step()
    print("Loss: ", loss)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

  pass


if __name__ == '__main__':
  main()
