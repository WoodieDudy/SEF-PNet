import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from scipy.signal import get_window


def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    """
    Return window coefficient
    """
    def sqrthann(win_len):
        return get_window("hann", win_len, fftbins=True)**0.5
    
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    elif win_type == "sqrthann":
        window = sqrthann(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)#**0.5
    
    N = fft_len
    fourier_basis = np.fft.rfft(np.eye(N))[:win_len]
    real_kernel = np.real(fourier_basis)
    imag_kernel = np.imag(fourier_basis)
    kernel = np.concatenate([real_kernel, imag_kernel], 1).T
    
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    kernel = kernel*window
    kernel = kernel[:, None, :]
    return torch.from_numpy(kernel.astype(np.float32)), torch.from_numpy(window[None,:,None].astype(np.float32))


class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
        super().__init__() 
        
        if fft_len is None:
            self.fft_len = int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        kernel, _ = init_kernels(win_len, win_inc, self.fft_len, win_type)
        out_ch, in_ch, k_size = kernel.shape
        
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k_size,
            stride=win_inc,
            bias=False
        )
        with torch.no_grad():
            self.conv.weight.copy_(kernel)
        self.conv.weight.requires_grad = False
        
        self.feature_type = feature_type
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        sig_len = inputs.size(-1)
        pad = 0 if sig_len % self.stride == 0 else self.stride - sig_len % self.stride
        inputs = F.pad(inputs, [0, pad])

        # inputs = F.pad(inputs, [self.win_len-self.stride, self.win_len-self.stride])

        outputs = self.conv(inputs)
        if outputs is None:
            return None
        
        if self.feature_type == 'complex':
            return outputs
        else:
            dim = self.dim // 2 + 1
            real = outputs[:, :dim, :]
            imag = outputs[:, dim:, :]
            mags = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            return mags, phase

class ConviSTFT(nn.Module):

    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming', feature_type='real'):
        super(ConviSTFT, self).__init__() 
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        kernel, window = init_kernels(win_len, win_inc, self.fft_len, win_type, invers=True)
        self.register_buffer('weight', kernel)
        self.feature_type = feature_type
        self.win_type = win_type
        self.win_len = win_len
        self.stride = win_inc
        self.stride = win_inc
        self.dim = self.fft_len
        self.register_buffer('window', window)
        self.register_buffer('enframe', torch.eye(win_len)[:,None,:])

    def forward(self, inputs, phase=None):
        """
        inputs : [B, N+2, T] (complex spec) or [B, N//2+1, T] (mags)
        phase: [B, N//2+1, T] (if not none)
        """ 
      
        if phase is not None:
            real = inputs*torch.cos(phase)
            imag = inputs*torch.sin(phase)
            inputs = torch.cat([real, imag], 1)
        outputs = F.conv_transpose1d(inputs, self.weight, stride=self.stride) 
    


        # this is from torch-stft: https://github.com/pseeth/torch-stft
        t = self.window.repeat(1,1,inputs.size(-1))**2
        coff = F.conv_transpose1d(t, self.enframe, stride=self.stride)
        outputs = outputs/(coff+1e-8)
        #outputs = torch.where(coff == 0, outputs, outputs/coff)
        outputs = outputs[...,self.win_len-self.stride:-(self.win_len-self.stride)]
        
        return outputs
    

class StreamingISTFT(nn.Module):
    def __init__(
            self,
            win_len: int,
            win_inc: int,
            fft_len: int | None = None,
            win_type: str = 'hamming',
            feature_type: str = 'complex'
        ):
        super().__init__()

        self.win_len = int(win_len)
        self.win_inc = int(win_inc)
        self.fft_len = int(2 ** np.ceil(np.log2(win_len))) if fft_len is None else int(fft_len)
        self.feature_type = feature_type

        if win_type == 'hamming':
            window = torch.hamming_window(self.win_len, periodic=False)
        elif win_type == 'hann':
            window = torch.hann_window(self.win_len, periodic=False)
        elif win_type == 'rect':
            window = torch.ones(self.win_len)
        else:
            raise ValueError(f'Inknown window type: {win_type}')

        self.register_buffer('window', window)
        self.register_buffer('win_sq', window.pow(2))

        self._tail_len = self.win_len - self.win_inc
        self.reset()

    def forward(self, frame: torch.Tensor | tuple):
        if self.feature_type == 'complex':
            if frame.dim() == 1:
                frame = frame.unsqueeze(0)
            dim = self.fft_len // 2 + 1
            real, imag = frame[:, :dim], frame[:, dim:]
            spec_pos = torch.complex(real, imag)
        elif self.feature_type == 'real':
            mag, phase = frame
            spec_pos = torch.polar(mag, phase)
        else:
            raise ValueError('feature_type: "complex" or "real"')

        spec_full = torch.cat(
            [spec_pos, torch.flip(spec_pos[..., 1:-1].conj(), dims=[-1])],
            dim=-1
        )

        time_frame = torch.fft.irfft(spec_full, n=self.fft_len)[..., :self.win_len]
        time_frame = time_frame * self.window

        B = time_frame.size(0)
        norm_frame = self.win_sq.repeat(B, 1)

        L, H = self.win_len, self.win_inc
        prev_L = self._tail_len

        time_frame[:, :prev_L] += self._overlap
        norm_frame[:, :prev_L] += self._overlap_norm

        eps = 1e-12
        out_chunk = time_frame[:, :H] / torch.clamp(norm_frame[:, :H], min=eps)

        self._overlap = time_frame[:, H:].clone()
        self._overlap_norm = norm_frame[:, H:].clone()

        return out_chunk

    def flush(self):
        eps = 1e-12
        tail = self._overlap / torch.clamp(self._overlap_norm, min=eps)
        self.reset(batch_size=tail.size(0), device=tail.device)
        return tail

    def reset(self, batch_size: int = 1, device: torch.device | str | None = None):
        if device is None:
            device = self.window.device
        self._overlap = torch.zeros(batch_size, self._tail_len, device=device)
        self._overlap_norm = torch.zeros(batch_size, self._tail_len, device=device)