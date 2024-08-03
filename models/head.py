import torch
import torch.nn as nn
from models.constants_embeddings import PATCH_DIM, HIDDEN_SIZE

class DecoderHeadSimple(nn.Module):
    """
    Heatmap based decoder head for the human pose estimation task.
    """
    def __init__(self, in_channels, out_channels = 14, num_deconv_layers=2,
                 num_deconv_filters=(224, 224),
                 num_deconv_kernels=(4, 4)): 
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels
        )

        self.last_conv = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )


    def forward(self, x):
        x =torch.swapaxes(x,1, 2) # -> (batches, channels, #patches)
        x = x.view((-1, HIDDEN_SIZE, PATCH_DIM, PATCH_DIM)) # ->( batches, channels, height, width)
        

        x = self.deconv_layers(x) 
        x = self.last_conv(x)
        return x

    def  _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """
        Create the deconvolution layers.
        """

        assert num_layers == len(num_filters) == len(num_kernels)

        layers = []
        
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i]) # selects for kernel sizes of 4,3,2 for the deconvolution layers
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2, 
                    padding=padding,
                    output_padding=output_padding,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes # update the number of input channels for the next deconvolution layer
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """
        setting kernel, padding and output_padding for deconvolution
        """
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError("Incompatible deconvolution kernel size.")

        return deconv_kernel, padding, output_padding