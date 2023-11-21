import os
import pandas as pd
from torchvision.io import read_image

class MPIIDataset():
  # image_labels: List[Dict] of the filtered mpii annolist
  def __init__(self, image_labels, img_dir):
    self.image_labels = image_labels
    self.img_dir = img_dir
    # can add transforms if we need/want

  def __len__(self):
    return len(self.image_labels)
  
  def __getitem__(self, idx):
    # inputs
    # idx: int -> index of image sample
    #
    # image: torch.Tensor -> pytorch tensor representing image specified
    # label: list[dict] -> each dict contains x and y coordinate, joint id, and whether the joint is visible

    img_path = self.img_dir + self.image_labels[idx]['image']['name']
    image = read_image(img_path)
    label = self.image_labels['annorect']['annopoints']['point']
    
    return image, label
  
class LSPDataset():
  # image_labels: List[Dict] of the filtered mpii annolist
  def __init__(self, image_labels, img_dir):
    self.image_labels = image_labels
    self.img_dir = img_dir
    # can add transforms if we need/want

  def __len__(self):
    return (self.image_labels.shape[0])

  def __getitem__(self, idx):
    # inputs
    # idx: int -> index of image sample
    #
    # image: torch.Tensor -> pytorch tensor representing image specified
    # label: list[dict] -> each dict contains x and y coordinate, joint id, and whether the joint is visible

    # img_path = self.img_dir + self.image_labels[idx]['image']['name']
    img_path = self.img_dir + "resized_im" + '0'*(5-len(str(idx+1))) + str(idx + 1) + ".jpg"
    image = read_image(img_path)

    label = self.image_labels[idx]

    return image, label