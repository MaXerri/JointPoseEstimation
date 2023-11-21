from .embeddings import Embeddings, PatchEmbeddings
import torch.nn as nn

class ViT(nn.Module):
  def __init__(self):
    super().__init__()
    self.img_size = (224,224)
    self.patch_size = (16,16)
    
    self.embeddings = Embeddings()
   
  def forward(self,x):
    x = self.embeddings(x)
    return x