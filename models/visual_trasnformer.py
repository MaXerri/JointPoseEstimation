from embeddings import Embeddings, PatchEmbeddings
from attention import attentionHead
from feedforward import feedForward
import torch
import torch.nn as nn
from torch.nn import functional as F
from models.constants_embeddings import NUM_CHANNELS, HIDDEN_SIZE, HIDDEN_DROPOUT_PROB, BATCH_SIZE, ATTENTION_HEADS, PATCH_DIM


class Transformer(nn.Module):
  def __init__(self):
    super().__init__()
    self.img_size = (224,224)
    self.patch_size = (16,16)
    self.embedsize = HIDDEN_SIZE
    self.attention = attentionHead()
    self.ffn = feedForward()
    self.norm1 = nn.LayerNorm((HIDDEN_SIZE,PATCH_DIM*PATCH_DIM))
    self.norm2 = nn.LayerNorm((PATCH_DIM*PATCH_DIM,HIDDEN_SIZE))
    self.embeds = PatchEmbeddings()

  def forward(self,x):
    print(x.shape)
    embed = self.embeds(x)
    tensor1 = self.norm1(embed)
    tensor1 = self.attention(tensor1)
    tensor1 = tensor1 + torch.transpose(embed,1,2)
    tensor2 = self.norm2(tensor1)
    tensor2 = self.ffn(tensor2)
    tensor2 = tensor2 + tensor1

    return tensor2