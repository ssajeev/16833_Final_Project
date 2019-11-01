
import argparse
import torch
import cv2
from PIL import Image
from torchvision import transforms 
from siamese_model_inf import *
from datetime import datetime
from tqdm import tqdm


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

def inference(model, img_l, img_r, i):
  transform = transforms.Compose([transforms.ToTensor()])
  output = model.forward(transform(img_l).unsqueeze(0), transform(img_r).unsqueeze(0))
  img_l_depth, img_r_depth = model.get_depth_imgs()
  cv2.imwrite("output/saved_model_10-31-2019-03:05:09/"+ str(i)+ ".png", cv2.applyColorMap(img_l_depth, cv2.COLORMAP_JET))
  cv2.imwrite("output_r/saved_model_10-31-2019-03:05:09/"+ str(i) +".png", cv2.applyColorMap(img_r_depth, cv2.COLORMAP_JET))
  return [img_l_depth, img_r_depth]


def run_inference():
  model = load_model("saved_model_10-31-2019-03:05:09.pt");

  for i in tqdm(range(1, 7192)):
      img_1 = Image.open("daVinci/test/image_0/00" + str(i).zfill(4) + ".png")
      img_2 = Image.open("daVinci/test/image_1/00"+ str(i).zfill(4) + ".png")
      img_l_depth, img_r_depth = inference(model, img_1, img_2, i)


run_inference()


