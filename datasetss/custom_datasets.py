import os
import pandas as pd
from torchvision.io import read_image
from utils.heatmap_funcs import generate_single_image_gaussian
  
class CustomDataset():
  """
  Custom Pose Dataset modelled after the Joint structur eof the LSP dataset.  
  Other datasets have to be preprocessed to be in this 14 joint format.

  Args: 
      train_data: list of tuples/lists in the following format: [image_labels, image_name, img_dir, sigma, inv] 

      - image_labels: list of joint annotations of shape (14,3)
      - image_name: list of image names corresponding to the joint annotations at the same index 
      - img_dir: image directory to the location storing images
      - sigma: gaussian blur standrad deviation
      - inv: True when 0 represents a visible joint (lsp_og)
      
  Returns: The image as a tensor, the joint heatmap labels, and the path to the image
  """
  #def __init__(self, image_labels, image_names, img_dir, sigma, inv=False):
  def __init__(self, train_data):

    self.train_data = train_data
    
    # can add transforms if we need/want

  def __len__(self):
    return (len(self.train_data))

  def __getitem__(self, idx):
    # idx: int -> index of image sample
    # image: torch.Tensor -> pytorch tensor representing image specified
    # label: list[dict] -> each dict contains x and y coordinate, joint id, and whether the joint is visible

    # img_path = self.img_dir + "resized_im" + '0'*(5-len(str(idx+1))) + str(idx + 1) + ".jpg"
    #img_path = self.img_dir[idx] + self.image_names[idx]
    img_path = self.train_data[idx][2] + self.train_data[idx][1]
    image = read_image(img_path)

    # generate heatmap label 
    # label = generate_single_image_gaussian(self.image_labels[idx], (56,56), self.sigma[idx], self.inv[idx])
    label = generate_single_image_gaussian(self.train_data[idx][0], (56,56), self.train_data[idx][3], self.train_data[idx][4])

    target_weight =self.train_data[idx][0][:,2]
    
    if self.train_data[idx][4]: #idx
      target_weight = 1- target_weight # inverts the tensor of 0s and 1s   

    return image, label, target_weight, img_path