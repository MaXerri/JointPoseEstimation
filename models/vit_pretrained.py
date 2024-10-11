import torch
import torch.nn as nn
from transformers import ViTForImageClassification


class PretrainedViTModel(nn.Module):
    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(PretrainedViTModel, self).__init__()
        
        # Load the pretrained ViT model
        self.vit = ViTForImageClassification.from_pretrained(pretrained_model_name)
        
        # Freeze the embedding weights (patch embeddings)
        for param in self.vit.vit.embeddings.parameters():
            param.requires_grad = False
        

    def forward(self, input):
        # input tensor should be of the shape: [batch_size, channels, height, width]
        return self.vit(input)

