
import argparse
import torch
import cv2
from PIL import Image
from torchvision import transforms 
from siamese_model_inf import *

def load_model(model_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  width = 384
  height = 192
  focal_length = ((373.47833252)**2 + (373.47833252)**2)**0.5
  baseline = -5.63117313

  model = SiameseDepthModel(width, height, focal_length, baseline)
  
  model.load_state_dict(torch.load(model_path))
  model.eval()
  return model 

def inference(model, img_l, img_r):
  transform = transforms.Compose([transforms.ToTensor()])
  output = model.forward(transform(img_l).unsqueeze(0), transform(img_r).unsqueeze(0))
  img_l_depth, img_r_depth = model.get_depth_imgs()
  cv2.imwrite("test1.png", img_l_depth)
  return [img_l_depth, img_r_depth]


def main():
  model = load_model("saved_model_10-25-2019-23:35:09.pt");
  img_1 = Image.open("daVinci/train/image_0/000300.png")
  img_2 = Image.open("daVinci/train/image_1/000300.png")
  inference(model, img_1, img_2)


main()
