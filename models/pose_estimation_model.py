import torch.nn as nn
from models.transformer_block import TransformerBlock
from models.head import DecoderHeadSimple
from models.embeddings import Embeddings
from models.constants_embeddings import HIDDEN_SIZE # 768

class TransformerPoseModel(nn.Module):
    """
    Transformer based pose estimation model.
    """
    def __init__(self, num_blocks, num_keypoints=14, num_deconv_layers=2,
                 num_deconv_filters=(224, 224),
                 num_deconv_kernels=(2, 2)):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_deconv_layers = num_deconv_layers
        self.num_deconv_filters = num_deconv_filters
        self.num_deconv_kernels = num_deconv_kernels
        self.embeds = Embeddings().to('mps')
        self.blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            block = TransformerBlock().to('mps')
            self.blocks.append(block)

        self.head = DecoderHeadSimple(
            in_channels= HIDDEN_SIZE, # gotta check this info I think its the size of the patch 
            out_channels=num_keypoints,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels
        ).to('mps')
    
    def forward(self, x, training):

        x = self.embeds(x) # pass to the patch embedding layer

        #print("embedding shape")
        #print(x.shape)

        for blocks in self.blocks: # pass through the transformer blocks
            x = blocks(x, training)

        #print("postT_trans shape")
        #print(x.shape)
        x = self.head(x) # pass through the decoder head

        return x