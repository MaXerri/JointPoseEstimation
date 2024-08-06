import torch
import torch.nn as nn
from torch.nn import functional as F
from models.constants_embeddings import NUM_CHANNELS, HIDDEN_SIZE, HIDDEN_DROPOUT_PROB, BATCH_SIZE, ATTENTION_HEADS, PATCH_DIM

class feedForward(nn.Module):
  """
  MLP for the transformer block
  """
  def __init__(self, activ, mlp_interm_factor = 4):
    super().__init__()
    self.indim = HIDDEN_SIZE
    self.mlp_intermediate_dim = mlp_interm_factor * HIDDEN_SIZE
    self.outdim = HIDDEN_SIZE

    self.lay1 = nn.Linear(self.indim,self.mlp_intermediate_dim)
    self.lay2 = nn.Linear(self.mlp_intermediate_dim,self.outdim)
    self.activ = activ() # test out different activations
    self.dropout = nn.Dropout(HIDDEN_DROPOUT_PROB)

  def forward(self,x):
    return self.dropout(self.lay2(self.activ(self.lay1(x)))) # missing Ocaml pipelining rn :(