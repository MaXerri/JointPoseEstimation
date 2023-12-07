import os
import pandas as pd
from torchvision.io import read_image
from utils.heatmap_funcs import generate_single_image_gaussian

class MPIIDataset():
  """
  MPII pose dataset 

  Parameters:
    - image_labels: list of joint annotations of shape (n,14,3)
    - image_name: list of image names corresponding to the joint annotations at the same index 
    - img_dir: image directory to the location storing images
  """
  def __init__(self, image_labels, image_name, img_dir):
    self.image_labels = image_labels
    self.img_dir = img_dir
    self.image_name = image_name
    # can add transforms if we need/want

  def __len__(self):
    return len(self.image_labels)
  
  def __getitem__(self, idx):
    # inputs
    # idx: int -> index of image sample
    #
    # image: torch.Tensor -> pytorch tensor representing image specified
    # label: list[dict] -> each dict contains x and y coordinate, joint id, and whether the joint is visible
    
    img_path = self.img_dir + self.image_name[idx]
    image = read_image(img_path)

    # generate heatmap label
    label = generate_single_image_gaussian(self.image_labels[idx], (56,56), 2)
    
    return image, label
  
class LSPDataset():
  """
  Leeds Sports pose dataset 

  Parameters:
    - image_labels: list of joint annotations of shape (n,14,3)
    - image_name: list of image names corresponding to the joint annotations at the same index 
    - img_dir: image directory to the location storing images
  """
  def __init__(self, image_labels, image_names, img_dir):
    self.image_labels = image_labels
    self.image_names = image_names
    self.img_dir = img_dir
    self.sigma = sigma
    # can add transforms if we need/want

  def __len__(self):
    return (self.image_labels.shape[0])

  def __getitem__(self, idx):
    # idx: int -> index of image sample
    # image: torch.Tensor -> pytorch tensor representing image specified
    # label: list[dict] -> each dict contains x and y coordinate, joint id, and whether the joint is visible

    
    # img_path = self.img_dir + "resized_im" + '0'*(5-len(str(idx+1))) + str(idx + 1) + ".jpg"
    img_path = self.img_dir + self.image_names[idx]
    image = read_image(img_path)

    # generate heatmap label 
    label = generate_single_image_gaussian(self.image_labels[idx], (56,56), self.sigma)

    return image, label, img_path