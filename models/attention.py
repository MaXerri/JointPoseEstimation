import torch
import torch.nn as nn
from torch.nn import functional as F
from models.constants_embeddings import NUM_CHANNELS, HIDDEN_SIZE, HIDDEN_DROPOUT_PROB, ATTN_DROPOUT_PROB, BATCH_SIZE, ATTENTION_HEADS, PATCH_DIM
import math

class attentionHead(nn.Module):
  """
  Multiheaded attention. Heads are simulatnaeously processed
  Multiheaded attention mechanism adapted from VITPose 2022 model backbone  
  """
  def __init__(self):
    super().__init__()
    self.feats = HIDDEN_SIZE 
    self.num_heads = ATTENTION_HEADS
    head_dim = self.feats // self.num_heads
    all_heads_dim = head_dim * self.num_heads

    self.first_proj = nn.Linear(self.feats,all_heads_dim *3) # for original config these 2 nums are equal
    self.last_proj = nn.Linear(all_heads_dim,self.feats)
    self.dropout = nn.Dropout(ATTN_DROPOUT_PROB)
    self.last_proj_drop = nn.Dropout(HIDDEN_DROPOUT_PROB) # add config param

  def forward(self,x, mode):
    """
    x input shape: (batach, num_patches, hidden_size)
    """
    batch_size = x.shape[0] 
    num_patches = x.shape[1]
    x = self.first_proj(x) # (B, #patches, all_heads_dim*3)

    #x = torch.reshape(x,(batch_size,ATTENTION_HEADS,3,PATCH_DIM*PATCH_DIM,HIDDEN_SIZE//ATTENTION_HEADS)) 
    #Q,K,V = torch.unbind(x,2)
    
    x = x.reshape(batch_size, num_patches, 3, self.num_heads, -1).permute(2,0,3,1,4)
    Q,K,V = x[0],x[1],x[2] 
    
    QK = torch.matmul(Q,torch.transpose(K,3,2))

    # These 3 lines were skipped during original training
    # QK = QK / math.sqrt(HIDDEN_SIZE//ATTENTION_HEADS) # attention head size
    # QK = F.softmax(QK,dim = -1) #May need to swap dimension of softmax
    # out = self.dropout(QK)

    out = torch.matmul(QK,V)
    out = torch.transpose(out,1,2)
    out = torch.flatten(out, start_dim = 2)
    out = self.last_proj(out)
    out = self.last_proj_drop(out)
    
    return out

  