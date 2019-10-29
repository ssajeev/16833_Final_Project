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

class cummalative_loss(nn.Module):
  def __init__(self, alpha=0.5, beta=0.5, c=1.0):
    super(cummalative_loss, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.c = c

  def SSI(self, x, y):
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
    return app_loss


  def forward(self, left_img, right_img, projected_img_l, projected_img_r,
              right_to_left_disp, left_to_right_disp, disp1_l, disp1_r):

    #SSI Loss
    ssim_loss_left = self.SSI(left_img, projected_img_l)
    ssim_loss_right = self.SSI(right_img, projected_img_r)

    #L1 Reconstruction Loss
    l1_loss_left = nn.L1Loss(left_img, projected_img_l)
    l1_loss_right = nn.L1Loss(right_img, projected_img_r)

    #Weighted Loss
    loss_left = self.alpha*(ssim_loss_left + l1_loss_left)
    loss_right = self.beta*(ssim_loss_right + l1_loss_right)

    #Consistency Loss
    lr_left_loss = nn.L1Loss(disp1_l, right_to_left_disp)
    lr_right_loss = nn.L1Loss(disp1_r, left_to_right_disp)
    lr_loss = lr_left_loss + lr_right_loss

    total_loss = self.c*lr_loss + loss_left + loss_right

    return total_loss

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

  criterion = cummalative_loss()

  for param in criterion.parameters():
    param.requires_grad = True

  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    t = tqdm(iter(dataset_loader), leave=False, total=len(dataset_loader))
    for i, batch_input in enumerate(t):
      left_img, right_img = batch_input["image_l"].to(device), batch_input["image_r"].to(device)
      optimizer.zero_grad()
      [projected_img_l, projected_img_r,
       right_to_left_disp, left_to_right_disp, disp1_l, disp1_r] = model(left_img, right_img)

      loss = criterion.forward(left_img, right_img, projected_img_l, projected_img_r,
              right_to_left_disp, left_to_right_disp, disp1_l, disp1_r)
      loss.backward()
      optimizer.step()
    print("Loss: ", loss)

    MODEL_SAVE_PATH = "saved_model_" + datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S") + ".pt"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

  pass


if __name__ == '__main__':
  main()
