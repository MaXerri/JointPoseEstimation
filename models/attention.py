import torch
import torch.nn as nn
from torch.nn import functional as F
from models.constants_embeddings import NUM_CHANNELS, HIDDEN_SIZE, HIDDEN_DROPOUT_PROB, BATCH_SIZE, ATTENTION_HEADS, PATCH_DIM
import math

class attentionHead(nn.Module):
  """
  Multiheaded attention layer.  Heads are all processed at the same time
  """
  def __init__(self):
    super().__init__()
    self.feats = HIDDEN_SIZE #TODO abstract this
    self.firstlin = nn.Linear(self.feats,self.feats*3)

  def forward(self,x, mode):

    batch_size = x.shape[0]
    x = self.firstlin(x) # (B, #patches, hidden_size*3)
    # -> (B, #heads, #patches, 3 matrices, __ )
    x = torch.reshape(x,(batch_size,ATTENTION_HEADS,3,PATCH_DIM*PATCH_DIM,HIDDEN_SIZE//ATTENTION_HEADS)) 
    # split into the indivisual Q K V  each of -> (B, # heads, #patches, __ )
    Q,K,V = torch.unbind(x,2)
    # calculations for the trasnformer equation: QK^T/sqrt(d_k)
    QK = torch.matmul(Q,torch.transpose(K,3,2))
    QK = QK / math.sqrt(HIDDEN_SIZE//ATTENTION_HEADS)
    QK = F.softmax(QK,dim = -1) #May need to swap dimension of softmax
    out = torch.matmul(QK,V)
    # get back to shape (B, #patches, hidden_size)
    out = torch.transpose(out,1,2)
    out = torch.flatten(out, start_dim = 2)
    return out