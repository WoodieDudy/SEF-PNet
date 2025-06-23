import torch
import torch.nn as nn

from .streaming_norm import StreamingInstanceNorm, EMAStreamingInstanceNorm, StreamingInstanceNormF

class TCNBlock(nn.Module):
    """
    TCN block:
        IN - ELU - Conv1D - IN - ELU - Conv1D
    """

    def __init__(
            self,
            in_dims: int = 384,
            out_dims: int = 384,
            kernel_size: int = 3,
            stride: int = 1,
            paddings: int = 1,
            dilation: int = 1,
            causal: bool = False,
            norm="streaming_ema"
        ) -> None:
        super(TCNBlock, self).__init__()
        self.elu1 = nn.ELU()
        if paddings == 1:
            dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
                dilation * (kernel_size - 1))
        else:
            dconv_pad = 0
        # dilated conv
        self.dconv1 = nn.Conv1d(
            in_dims,
            out_dims,
            kernel_size,
            padding=dconv_pad,
            dilation=dilation,
            groups=in_dims,
            bias=True)
        
        self.elu2 = nn.ELU()    
        self.dconv2 = nn.Conv1d(in_dims, out_dims, 1, bias=True)
        
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad
        
        self.buf = None

        if norm == "streaming_instance":
            self.norm1 = StreamingInstanceNorm()
            self.norm2 = StreamingInstanceNorm()
        elif norm == "streaming_instance_f":
            self.norm1 = StreamingInstanceNormF()
            self.norm2 = StreamingInstanceNormF()
        elif norm == "instance":
            self.norm1 = nn.InstanceNorm1d(in_dims)
            self.norm2 = nn.InstanceNorm1d(in_dims)
        elif norm == "streaming_ema":
            self.norm1 = EMAStreamingInstanceNorm()
            self.norm2 = EMAStreamingInstanceNorm()
        else:
            self.norm1 = nn.Identity(in_dims)
            self.norm2 = nn.Identity(in_dims)
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.elu1(self.norm1(x))
        y = self.dconv1(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.elu2(self.norm2(y))
        y = self.dconv2(y)
        x = x + y
        return x
