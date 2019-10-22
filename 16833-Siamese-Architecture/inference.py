
import argparse
import torch

def load_model(model_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = torch.load(model_path)
  model.eval()

def inference(model, img_l, img_r):
  output = model({'image_l': img_l, 'image_r':img_r})
  img_l_depth, img_r_depth = model.get_depth_imgs()
  return [img_l_depth, img_r_depth]