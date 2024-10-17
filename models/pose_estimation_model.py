import torch.nn as nn
import torch
from models.transformer_block import TransformerBlock
from models.head import DecoderHeadSimple
from models.embeddings import Embeddings
from models.constants_embeddings import HIDDEN_SIZE, INITIALIZER_RANGE
from models.vit_pretrained import PretrainedViTModel

class TransformerPoseModel(nn.Module):
    """
    Transformer based pose estimation model.

    Params:

    num_blocks: number of transformer blocks
    num_keypoints: number of joint keypoints to predict
    num_deconv_layers: number of deconvolution layers in deconv head
    num_deconv_filters: for attention head
    num_deconv_kernels: deconv kernel size for attention head
    """
    def __init__(self, num_blocks, num_keypoints=14, num_deconv_layers=2,
                 num_deconv_filters=(224, 224),
                 num_deconv_kernels=(4, 4),
                 pretrained_model=None):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_deconv_layers = num_deconv_layers
        self.num_deconv_filters = num_deconv_filters
        self.num_deconv_kernels = num_deconv_kernels
        self.pretrained_model = pretrained_model

        if pretrained_model is None:
            self.transformer_backbone = TransformerBackbone(num_blocks, num_keypoints)
        else:
            self.transformer_backbone = PretrainedViTModel() # initialize pretrained model
            self.transformer_backbone.classifier = torch.nn.Identity() # remove the classifier layer extracting last hidden layer

        self.head = DecoderHeadSimple(
            in_channels= HIDDEN_SIZE, # gotta check this info I think its the size of the patch 
            out_channels=num_keypoints,
            num_deconv_layers=num_deconv_layers,
            num_deconv_filters=num_deconv_filters,
            num_deconv_kernels=num_deconv_kernels
        )

        # initialize model weights - non-pretrained
        if pretrained_model is None:
            self.transformer_backbone.init_weights_backbone()
        
        self.head.init_weights()
        
    def forward(self, x, mode):
        if self.pretrained_model is None:
            x = self.transformer_backbone(x, mode) # pass through the backbone
        else:
            x = self.transformer_backbone(x)
        
        # print(x.shape)
        x = self.head(x) # pass through the decoder head

        return x


class TransformerBackbone(nn.Module):
    """
    Transformer backbone for the pose estimation Model
    """

    def __init__(self, num_blocks, num_keypoints=14):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.embeds = Embeddings()
        self.blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            block = TransformerBlock()
            self.blocks.append(block)


    def forward(self, x, mode):

        x = self.embeds(x) # pass to the patch embedding layer

        #print("embedding shape")
        #print(x.shape)

        for blocks in self.blocks: # pass through the transformer blocks
            x = blocks(x, mode)

        return x 

    def init_weights_backbone(self):
        """
        Initializing the weights on non-pretrained model.  If model is every pretrained, modify and use pretrained weight

        initializer range - for std of weights
        """

        def _initialize(module):
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

        self.apply(_initialize)

        
