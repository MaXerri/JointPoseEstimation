import torch
import torch.nn as nn
from torch.nn import functional as F
from models.constants_embeddings import NUM_CHANNELS, HIDDEN_SIZE, HIDDEN_DROPOUT_PROB, BATCH_SIZE, ATTENTION_HEADS, PATCH_DIM

class attentionHead(nn.Module):
  def __init__(self):
    super().__init__()
    self.feats = HIDDEN_SIZE #TODO abstract this
    self.firstlin = nn.Linear(self.feats,self.feats*3)

  def forward(self,x, mode):
    """
    x :
    """
    batch_size = x.shape[0]
    x = self.firstlin(x) # (B, #patches, hidden_size*3)
    x = torch.reshape(x,(batch_size,ATTENTION_HEADS,3,PATCH_DIM*PATCH_DIM,HIDDEN_SIZE//ATTENTION_HEADS)) 
    Q,K,V = torch.unbind(x,2)
    QK = torch.matmul(Q,torch.transpose(K,3,2))
    # QK = F.softmax(QK,dim = -1) #May need to swap dimension of softmax
    out = torch.matmul(QK,V)
    out = torch.transpose(out,1,2)
    out = torch.flatten(out, start_dim = 2)
    return out