import torch
import torch.nn as nn
from .constants_embeddings import NUM_CHANNELS, HIDDEN_SIZE, HIDDEN_DROPOUT_PROB

class PatchEmbeddings(nn.Module):
  """
  image patches -> embeddings
  """
  def __init__(self):
    super().__init__()
    self.img_size = (224,224)
    self.patch_size = (16,16)

    self.patch_embed = nn.Conv2d(NUM_CHANNELS, HIDDEN_SIZE, self.patch_size, stride=self.patch_size) # (B,C,H,W)

  def forward(self,x):
    x = self.patch_embed(x.float()).flatten(2) # output of (B, new channel dimension, #patches)
    x= x.transpose(1,2) # (B, #patches, new channel dimension) ??
    return x


  #print(input.shape)

  #print(input.shape)
  #print(output)
  #embed = patch_embedding(i)
  #print(embed)

class Embeddings(nn.Module):
  """
  creates the embeddings vectors for the patches by combining patch embeddings with positional embeddings
  """

  def __init__(self):
    super().__init__()
    self.patch_embeddings = PatchEmbeddings()

    # TODO add CLS token

    num_patches = (self.patch_embeddings.img_size[0] // self.patch_embeddings.patch_size[0]) ** 2

    # creating positional embedding, this is a learned parameter
    # num_patches -> num_patches + 1 for adding CLS
    self.position_embed =  nn.Parameter(torch.randn(1, num_patches, HIDDEN_SIZE)) # this is without cls token for now
    self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)

  def forward(self, patches):
    # create embedding for this patch
    embeddings = self.patch_embeddings(patches)
    #print("pos embeddings")
    #print(embeddings.shape)
    batch_size = embeddings.size()[0]

    # TODO Add CLS tokens

    # add positional embedding to patch embedding
    embeddings = embeddings + self.position_embed
    embeddings = self.dropout(embeddings)
    return embeddings

