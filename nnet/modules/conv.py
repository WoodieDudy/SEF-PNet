from numpy import isin
import torch
import torch.nn as nn

from .streaming_norm import StreamingInstanceNorm, EMAStreamingInstanceNorm, StreamingInstanceNormF


class Conv2dBlock(nn.Module):
    def __init__(
            self, 
            in_dims: int = 16,
            out_dims: int = 32,
            kernel_size: tuple[int] = (3, 3),
            stride: tuple[int] = (1, 1),
            padding: tuple[int] = (1, 1),
            norm="streaming_ema",
            cut_right_padding=False
        ) -> None:
        super(Conv2dBlock, self).__init__() 
        self.conv2d = nn.Conv2d(in_dims, out_dims, kernel_size, stride, padding)     
        self.elu = nn.ELU()
        if norm == "streaming_instance":
            self.norm = StreamingInstanceNorm()
        elif norm == "streaming_instance_f":
            self.norm = StreamingInstanceNormF()
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_dims)
        elif norm == "streaming_ema":
            self.norm = EMAStreamingInstanceNorm()
        else:
            self.norm = nn.Identity()
        self.cut_right_padding = cut_right_padding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(x)
        if x is None:
            return None
        x = self.elu(x)
        x = self.norm(x)
        if self.cut_right_padding:
            return x[..., :-2, :]
        return x

    def reset(self):
        if isinstance(self.norm, StreamingInstanceNorm):
            self.norm.reset()


class ConvTrans2dBlock(nn.Module):
    def __init__(
            self, 
            in_dims: int = 32,
            out_dims: int = 16,
            kernel_size: tuple[int] = (3, 3),
            stride: tuple[int] = (1, 2),
            padding: tuple[int] = (1, 0),
            output_padding: tuple[int] = (0, 0),
            bias=True,
            norm="streaming_ema",
            cut_right_padding=True
        ) -> None:
        super(ConvTrans2dBlock, self).__init__() 
        self.convtrans2d = nn.ConvTranspose2d(in_dims, out_dims, kernel_size, stride, padding, output_padding, bias=bias)     
        self.elu = nn.ELU()

        if norm == "streaming_instance":
            self.norm = StreamingInstanceNorm()
        elif norm == "streaming_instance_f":
            self.norm = StreamingInstanceNormF()
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_dims)
        elif norm == "streaming_ema":
            self.norm = EMAStreamingInstanceNorm()
        else:
            self.norm = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtrans2d(x)
        if x is None:
            return None
        x = self.elu(x)
        return self.norm(x)
