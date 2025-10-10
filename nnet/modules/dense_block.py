import torch
import torch.nn as nn

from .conv import Conv2dBlock

class DenseBlock(nn.Module):
    def __init__(self, in_dims, out_dims, mode = "enc", **kargs):
        super(DenseBlock, self).__init__()
        if mode not in ["enc", "dec"]:
            raise RuntimeError("The mode option must be 'enc' or 'dec'!")
            
        n = 1 if mode == "enc" else 2
        self.n = n
        self.conv1 = Conv2dBlock(in_dims=in_dims*n, out_dims=in_dims, **kargs)
        self.conv2 = Conv2dBlock(in_dims=in_dims*(n+1), out_dims=in_dims, **kargs)
        self.conv3 = Conv2dBlock(in_dims=in_dims*(n+2), out_dims=in_dims, **kargs)
        self.conv4 = Conv2dBlock(in_dims=in_dims*(n+3), out_dims=in_dims, **kargs)
        self.conv5 = Conv2dBlock(in_dims=in_dims*(n+4), out_dims=out_dims, **kargs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.conv1(x)
        y2 = self.conv2(torch.cat([x, y1], 1))
        y3 = self.conv3(torch.cat([x, y1, y2], 1))
        y4 = self.conv4(torch.cat([x, y1, y2, y3], 1))
        y5 = self.conv5(torch.cat([x, y1, y2, y3, y4], 1))
        return y5