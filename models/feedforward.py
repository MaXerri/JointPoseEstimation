import torch
import torch.nn as nn
from torch.nn import functional as F
from models.constants_embeddings import NUM_CHANNELS, HIDDEN_SIZE, HIDDEN_DROPOUT_PROB, BATCH_SIZE, ATTENTION_HEADS, PATCH_DIM

class feedForward(nn.Module):
  """
  MLP for the transformer block
  """
  def __init__(self, activ):
    super().__init__()
    self.indim = HIDDEN_SIZE
    self.outdim = HIDDEN_SIZE
    self.lay1 = nn.Linear(self.indim,self.outdim)
    self.lay2 = nn.Linear(self.outdim,self.indim)
    self.activ = activ() # test out changing to differnt activations
    self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)
  def forward(self,x):
    return self.dropout(self.lay2(self.activ(self.lay1(x))))