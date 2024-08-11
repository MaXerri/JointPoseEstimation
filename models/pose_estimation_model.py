import torch.nn as nn
import torch
from models.transformer_block import TransformerBlock
from models.head import DecoderHeadSimple
from models.embeddings import Embeddings
from models.constants_embeddings import HIDDEN_SIZE, INITIALIZER_RANGE

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
        self.embeds = Embeddings()
        self.blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            block = TransformerBlock()
            self.blocks.append(block)

        self.head = DecoderHeadSimple(
            in_channels= HIDDEN_SIZE, # gotta check this info I think its the size of the patch 
            out_channels=num_keypoints,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels
        )

        self.apply(self._init_weights_backbone) # initialize weights
    
    def forward(self, x, mode):

        x = self.embeds(x) # pass to the patch embedding layer

        #print("embedding shape")
        #print(x.shape)

        for blocks in self.blocks: # pass through the transformer blocks
            x = blocks(x, mode)

        #print("postT_trans shape")
        #print(x.shape)
        x = self.head(x) # pass through the decoder head

        return x

    def _init_weights_backbone(self, module):
        """
        Initializing the weights on non-pretrained model.  If model is every pretrained, modify and use pretrained weight

        initializer range - for std of weights
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=INITIALIZER_RANGE)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embed.data = nn.init.trunc_normal_(
                module.position_embed.data.to(torch.float32),
                mean = 0.0,
                std = INITIALIZER_RANGE
            ).to(module.position_embed.data.dtype)

        print("backbone weights initialized")

