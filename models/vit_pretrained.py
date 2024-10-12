import torch
import torch.nn as nn
from transformers import ViTModel


class PretrainedViTModel(nn.Module):
    """
    Initializing a bare Huggingface pretrained ViT model which returns the raw final hidden state tensor.
    Refer to https://huggingface.co/docs/transformers/en/model_doc/vit 
    """


    def __init__(self, pretrained_model_name='google/vit-base-patch16-224'):
        super(PretrainedViTModel, self).__init__()
        
        # Load the pretrained ViT model
        model = ViTModel.from_pretrained(pretrained_model_name)

        self.vit = model

        # Freeze embeddings
        #for param in self.vit.embeddings.parameters():
        #    param.requires_grad = False

        # Freeze encoder
        #for param in self.vit.encoder.parameters():
        #    param.requires_grad = False

        # pooler module needs to have trainable weights as they are initializde to 0

    def forward(self, input):
        # input tensor should be of the shape: [batch_size, channels, height, width]
        x = (self.vit(input).last_hidden_state) # extracts the return tensow with shape (batch_size, num_patches + 1, hidden_size)
        x = x[:,1:,:] # remove the CLS token 
        return x

