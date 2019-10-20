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
                            self.meta_frame.iloc[idx, 0])
    image_l = cv2.imread(img_name_l)

    img_name_r = os.path.join(self.root_dir, 'image_1',
                              self.meta_frame.iloc[idx, 0])
    image_r = cv2.imread(img_name_r)

    sample = {'image_l': image_l, 'image_r': image_r}

    if self.transform:
      sample = self.transform(sample)

    return sample