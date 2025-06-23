import torch
import torch.nn as nn

class StreamingMixin:
    def __init__(self, conv: nn.Module, axis: int = -1):
        super().__init__()
        self.conv = conv
        self.axis = axis

        kernel = conv.kernel_size
        len_k = len(kernel)
        total_dims = len_k + 2

        ax = axis if axis >= 0 else axis + total_dims
        spatial_start = total_dims - len_k
        idx = ax - spatial_start
        self.idx = idx
        if idx < 0 or idx >= len_k:
            raise ValueError(f"axis={axis} not in spatial dims {total_dims-len_k}..{total_dims-1}")

        k = kernel[idx]
        s = conv.stride[idx] if hasattr(conv, "stride") else 1
        d = conv.dilation[idx] if hasattr(conv, "dilation") else 1
        p = conv.padding[idx]   if hasattr(conv, "padding")   else 0

        self.padding = d * (k - s)

        self.k_eff  = (k - 1) * d + 1
        assert self.k_eff >= 1
        self.stride = s

        self.register_buffer("_buf", torch.zeros(0))

    def _split(self, x: torch.Tensor):
        L_in = x.size(self.axis)
        if L_in < self.k_eff:
            return None, x

        extra = (L_in - self.k_eff) % self.stride
        head_len = L_in - extra
        tail_len = L_in - (head_len - (self.k_eff - self.stride))

        head = x.narrow(self.axis, 0, head_len)
        tail = x.narrow(self.axis, head_len - (self.k_eff - self.stride), tail_len)
        return head, tail

    def reset(self):
        self._buf = torch.zeros(0, device=self._buf.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None
        if self._buf.numel():
            x = torch.cat((self._buf, x), dim=self.axis)
        else:
            if x.dim() == 4:
                B, C, T, F = x.shape
                self._buf = torch.zeros(B, C, self.padding, F, device=x.device)
                x = torch.cat((self._buf, x), dim=self.axis)
            else:
                B, C, L = x.shape
                self._buf = torch.zeros(B, C, self.padding, device=x.device)
                x = torch.cat((self._buf, x), dim=self.axis)

        head, tail = self._split(x)
        if head is None:
            self._buf = x.detach()
            return None

        self._buf = tail.detach()

        y = self.conv(head)
        return y
    

class StreamingConvTransposeMixin:
    def __init__(self, conv: nn.Module, idx: int = 0):
        super().__init__()
        self.conv = conv
        kernel = conv.kernel_size
        k = kernel[idx]
        d = conv.dilation[idx]
        self.k_eff  = (k - 1) * d + 1

        self.stride = conv.stride[idx]

        self.buffer = None
        self.step = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x is None:
            return None

        y = self.conv(x)
        if y.dim() == 3:
            y = y.unsqueeze(-1)
        B, C, _, F = y.shape

        if self.buffer is None:
            self.buffer = torch.zeros(B, C, self.k_eff, F,
                                   device=y.device, dtype=y.dtype)

        self.buffer += y

        n_convs = min(self.step + 1, self.k_eff)
        bias_sub = n_convs - 1

        out = self.buffer[..., :self.stride, :].clone()

        if self.conv.bias is not None and bias_sub > 0:
            b = self.conv.bias.view(1, -1, 1, 1)
            out = out - bias_sub * b

        zeros = torch.zeros(B, C, self.stride, F,
                         device=y.device, dtype=y.dtype)
        self.buffer = torch.cat([self.buffer[..., self.stride:, :], zeros], dim=2)

        self.step += 1
        return out


class StreamingConv1d(StreamingMixin, nn.Module):
    def __init__(self, conv1d: nn.Conv1d):
        nn.Module.__init__(self)
        StreamingMixin.__init__(self, conv1d, axis=-1)

class StreamingConv2d(StreamingMixin, nn.Module):
    def __init__(self, conv2d: nn.Conv2d):
        nn.Module.__init__(self)
        StreamingMixin.__init__(self, conv2d, axis=2)

class StreamingConvTrans2d(StreamingConvTransposeMixin, nn.Module):
    def __init__(self, deconv2d: nn.ConvTranspose2d):
        nn.Module.__init__(self)
        StreamingConvTransposeMixin.__init__(self, deconv2d, idx=0)



def make_streaming(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv1d):
            setattr(module, name, StreamingConv1d(child))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, StreamingConv2d(child))
        elif isinstance(child, nn.ConvTranspose2d):
            setattr(module, name, StreamingConvTrans2d(child))
        else:
            make_streaming(child)


def reset_streaming_modules(model: torch.nn.Module) -> None:
    for name, child in model.named_children():
        reset_fn = getattr(child, 'reset', None)
        if callable(reset_fn):
            child.reset()
        else:
            reset_streaming_modules(child)