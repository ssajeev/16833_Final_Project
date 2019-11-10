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
EPOCHS = 50
LEARNING_RATE = 0.001

class cummalative_loss(nn.Module):
  def __init__(self, alpha=0.5, beta=0.5, c=1.0):
    super(cummalative_loss, self).__init__()
    self.alpha = alpha
    self.beta = beta
    self.c = c

  def gradient_x(self, img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx
  
  def gradient_y(self, img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy

  def disparity_smoothness(self, disp, pyramid ):
    disp_gradients_x = [self.gradient_x(d) for d in disp]
    disp_gradients_y = [self.gradient_y(d) for d in disp]

    image_gradients_x = [self.gradient_x(img) for img in pyramid]
    image_gradients_y = [self.gradient_y(img) for img in pyramid]

    weights_x = [torch.exp(-1*torch.mean(torch.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
    weights_y = [torch.exp(-1*torch.mean(torch.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y
    return smoothness_x + smoothness_y
 
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
    l1_loss_left = torch.mean(torch.abs(left_img - projected_img_l))
    l1_loss_right = torch.mean(torch.abs(right_img - projected_img_r))

    #Weighted Loss
    loss_left = self.alpha*(ssim_loss_left + l1_loss_left)
    loss_right = self.beta*(ssim_loss_right + l1_loss_right)

    #Consistency Loss
    lr_left_loss = torch.mean(torch.abs(disp1_l - right_to_left_disp))
    lr_right_loss = torch.mean(torch.abs(disp1_r - left_to_right_disp))
    lr_loss = lr_left_loss + lr_right_loss
    
    #Disparity Smoothing 
    #disp_left_loss  = torch.mean(torch.abs(disp_left_smoothness))
    #disp_right_loss = torch.mean(torch.abs(disp_right_smoothness))
    #disp_gradient_loss = disp_left_loss + disp_right_loss
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
  model.load_state_dict(torch.load("models/hist_model_epoch_21.pt"))
  model = model.train()
  for param in model.parameters():
    param.requires_grad = True

  criterion = cummalative_loss()

  for param in criterion.parameters():
    param.requires_grad = True

  optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
  for epoch in range(EPOCHS):
    epoch_loss = 0
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

      epoch_loss += loss.item()

    print("Loss: ", epoch_loss/len(dataset_loader))
    MODEL_SAVE_PATH = "models/inpaint_hist_model_epoch_" + str(epoch) + ".pt"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

  pass


if __name__ == '__main__':
  main()
