from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from PIL import Image


def apply_image_preprocessing(img):

  #Removes the white parts of the image and applies content-aware fill
  lower = np.array([254, 254, 254])  #-- Lower range --
  upper = np.array([255, 255, 255])  #-- Upper range --
  mask = cv2.inRange(img, lower, upper)

  mask_result = np.average(cv2.bitwise_and(img, img, mask= mask), 2).astype('uint8')

  img = cv2.inpaint(img, mask_result , 3 ,cv2.INPAINT_TELEA)

  #Historgram equalization
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_norm = cv2.equalizeHist(img_gray)
  img_norm_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

  return img_norm_color

class EndoscopicDataset(Dataset):
  """Hamlyn Endoscopic Surgery Images Dataset."""

  def __init__(self, csv_file, root_dir, transform=None):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.meta_frame = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.meta_frame)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()



    img_name_l = os.path.join(self.root_dir, 'image_0',
                            self.meta_frame.iloc[idx, 1])
    image_l = cv2.imread(img_name_l)
    image_l = apply_image_preprocessing(image_l)
    image_l = Image.fromarray(image_l)

    img_name_r = os.path.join(self.root_dir, 'image_1',
                              self.meta_frame.iloc[idx, 1])
    image_r = cv2.imread(img_name_r)
    image_r = apply_image_preprocessing(image_r)
    image_r = Image.fromarray(image_r)
    sample = {'image_l': image_l, 'image_r': image_r}

    if self.transform:
      sample["image_l"] = self.transform(sample["image_l"])
      sample["image_r"] = self.transform(sample["image_r"])

    return sample
