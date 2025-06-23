# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:42:21 2021

@author: Jyhan
"""

import torch as th
import torch
import torch.nn as nn
import torchaudio
from torch.amp import autocast

from typing import Tuple, List
from memonger import SublinearSequential
# from torch.nn import Sequential as SublinearSequential
from modules.transform_enroll import PreEmphasis, SpeakerFuseLayer, SpeakerTransform
from memonger.resnet import ResNet34 as ResNet

from modules import *
from streaming_stuff import reset_streaming_modules


class DenseUNet(nn.Module):
    def __init__(self, 
                 win_len: int = 512,    # 32 ms
                 win_inc: int = 128,    # 8 ms
                 fft_len: int = 512,
                 win_type: str = "sqrthann",                 
                 kernel_size: Tuple[int] = (3, 3),
                 stride1: Tuple[int] = (1, 1),
                 stride2: Tuple[int] = (1, 2),
                 paddings: Tuple[int] = (1, 0),
                 output_padding: Tuple[int] = (0, 0),
                 tcn_dims: int = 384,
                 tcn_blocks: int = 10,
                 tcn_layers: int = 2,
                 causal: bool = False,     
                 pool_size: Tuple[int] = (4, 8, 16, 32),
                 num_spks: int = 1) -> None:
        super(DenseUNet, self).__init__()
        
        self.fft_len = fft_len
        self.num_spks = num_spks
        self.win_len = win_len
        self.win_inc = win_inc
        
        paddings = (2, 0)
        self.stft = ConvSTFT(win_len, win_inc, fft_len, win_type, 'complex')
        self.conv2d = nn.Conv2d(2, 16, kernel_size, stride1, paddings)           
        self.encoder = self._build_encoder(
                    kernel_size=kernel_size,
                    stride=stride2,
                    padding=paddings,
                    cut_right_padding=True,
                )
        self.tcn_layers = self._build_tcn_layers(
                    tcn_layers,
                    tcn_blocks,
                    in_dims=tcn_dims,
                    out_dims=tcn_dims,
                    causal=True,
                )
        self.decoder = self._build_decoder(
                    kernel_size=kernel_size,
                    stride=stride2,
                    padding=(0, 0),
                    output_padding=output_padding,
                    cut_right_padding=True,
                )
        self.avg_pool = StreamingTemporalAvgPool(pool_size)
        # self.avg_pool = self._build_avg_pool(pool_size)
        # self.avg_proj = nn.Conv2d(64, 32, 1, 1)
        self.avg_proj = nn.Conv2d(128, 32, 1, 1)
        
        self.deconv2d = nn.ConvTranspose2d(32, 2*num_spks, kernel_size, stride1, (0, 0))

        self.spk_model = ResNet(feat_dim=80, embed_dim=256)
        pretrained_model = torch.load("voxceleb_resnet34_avg_model.pt")
        state = self.spk_model.state_dict()
        for key in state.keys():
            if key in pretrained_model.keys():
                state[key] = pretrained_model[key]
                # print(key)
            else:
                print("not %s loaded" % key)
        self.spk_model.load_state_dict(state)
        for param in self.spk_model.parameters():
            param.requires_grad = False

        self.preEmphasis = PreEmphasis()
        self.spk_transform = SpeakerTransform()
        self.spk_encoder = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, # энкодер обучен на 16кгц
            n_fft=fft_len,
            win_length=win_len,
            hop_length=win_inc,
            f_min=20,
            window_fn=th.hamming_window,
            n_mels=80,
        )

        self.spk_fuse = SpeakerFuseLayer(
            embed_dim=256,
            # feat_dim=511,
            feat_dim=255,
            fuse_type="multiply",
        )
        self.register_buffer('window', torch.hamming_window(self.win_len, periodic=False))
        
    def _build_encoder(self, **enc_kargs):
        """
        Build encoder layers 
        """
        encoder = nn.ModuleList()
        encoder.append(DenseBlock(16, 16, "enc", padding=(2, 1), cut_right_padding=enc_kargs['cut_right_padding']))
        for i in range(4):
            encoder.append(
                    SublinearSequential(
                            Conv2dBlock(in_dims=16 if i==0 else 32, out_dims=32, **enc_kargs),
                            DenseBlock(32, 32, "enc", padding=(2, 1), cut_right_padding=enc_kargs['cut_right_padding'])
                        )
                    )
        encoder.append(Conv2dBlock(in_dims=32, out_dims=64, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=64, out_dims=128, **enc_kargs))
        encoder.append(Conv2dBlock(in_dims=128, out_dims=384, **enc_kargs))

        # encoder.append(Conv2dBlock(in_dims=128, out_dims=256, **enc_kargs))
        # encoder.append(Conv2dBlock(in_dims=256, out_dims=384, **enc_kargs))

        return encoder
    
    def _build_decoder(self, **dec_kargs):
        """
        Build decoder layers 
        """
        decoder = nn.ModuleList()
        decoder.append(ConvTrans2dBlock(in_dims=384*2, out_dims=128, **dec_kargs))
        # decoder.append(ConvTrans2dBlock(in_dims=384*2, out_dims=256, **dec_kargs))
        # decoder.append(ConvTrans2dBlock(in_dims=256*2, out_dims=128, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=128*2, out_dims=64, **dec_kargs))
        decoder.append(ConvTrans2dBlock(in_dims=64*2, out_dims=32, **dec_kargs))        
        for i in range(4):
            decoder.append(
                    SublinearSequential(
                            DenseBlock(32, 64, "dec", padding=(2, 1), cut_right_padding=dec_kargs['cut_right_padding']),
                            ConvTrans2dBlock(in_dims=64, 
                                             out_dims=32  if i!=3 else 16,
                                             **dec_kargs)
                            )
                    )
        decoder.append(DenseBlock(16, 32, "dec", padding=(2, 1), cut_right_padding=dec_kargs['cut_right_padding']))                            
        
        return decoder    
    
    def _build_tcn_blocks(self, tcn_blocks, **tcn_kargs):
        """
        Build TCN blocks in each repeat (layer)
        """
        blocks = [
            TCNBlock(**tcn_kargs, dilation=(2**b))
            for b in range(tcn_blocks)
        ]
        
        return SublinearSequential(*blocks)
    
    def _build_tcn_layers(self, tcn_layers, tcn_blocks, **tcn_kargs):
        """
        Build TCN layers
        """
        layers = [
            self._build_tcn_blocks(tcn_blocks, **tcn_kargs)
            for _ in range(tcn_layers)
        ]
        
        return SublinearSequential(*layers)
    
    def _build_avg_pool(self, pool_size):
        """
        Build avg pooling layers
        """
        avg_pool = nn.ModuleList()
        for sz in pool_size:
            avg_pool.append(
                    SublinearSequential(
                            nn.AvgPool2d(sz),
                            nn.Conv2d(32, 8, 1, 1)                            
                            )
                )
        
        return avg_pool
    
    def wav2spec(self, x: th.Tensor, mags: bool = False) -> th.Tensor:
        """
        convert waveform to spectrogram
        """
        assert x.dim() == 2 
        # x = x / th.std(x, -1, keepdims=True)        # variance normalization
        specs = self.stft(x)
        real = specs[:,:self.fft_len//2+1]
        imag = specs[:,self.fft_len//2+1:]
        spec = th.stack([real,imag], 1)
        spec = th.einsum("hijk->hikj", spec)    # batchsize, 2, T, F        
        if mags:
            return th.sqrt(real**2+imag**2+1e-8)
        else:
            return spec    
    
    def sep(self, spec: th.Tensor) -> List[th.Tensor]:
        """
        spec: (batchsize, 2*num_spks, T, F)
        return [real, imag] or waveform for each speaker
        """
        spec = th.einsum("hijk->hikj", spec)        # (batchsize, 2*num_spks, F, T)
        spec = th.chunk(spec, self.num_spks, 1)
        B, N, F, T = spec[0].shape
        est1 = th.chunk(spec[0], 2, 1)      # [(B, 1, F, T), (B, 1, F, T)]
        est2 = th.chunk(spec[1], 2, 1)  
        est1 = th.cat(est1, 2).reshape(B, -1, T)      # B, 1, 2F, T
        est2 = th.cat(est2, 2).reshape(B, -1, T)  
        return [th.squeeze(self.istft(est1)), th.squeeze(self.istft(est2))]
        

    def process_enroll(self, enroll):
        with th.no_grad(), autocast(device_type="cuda", enabled=False):
            spk_emb_input = self.preEmphasis(enroll)
            spk_emb_input = self.spk_encoder(spk_emb_input) + 1e-8
            spk_emb_input = spk_emb_input.log()
            spk_emb_input = spk_emb_input - torch.mean(
                spk_emb_input, dim=-1, keepdim=True)
            
            spk_emb_input = spk_emb_input.permute(0, 2, 1)

            tmp_spk_emb_input = self.spk_model(spk_emb_input)
            if isinstance(tmp_spk_emb_input, tuple):
                spk_emb_input = tmp_spk_emb_input[-1]
            else:
                spk_emb_input = tmp_spk_emb_input

        spk_embedding = self.spk_transform(spk_emb_input)
        spk_embedding = spk_embedding.unsqueeze(1).unsqueeze(3)
        return spk_embedding
    
    def reset(self):
        reset_streaming_modules(self)

    # def forward(self, x: th.Tensor, enroll: th.Tensor) -> th.Tensor:
    def forward(self, x, speaker_features) -> th.Tensor:

        # speaker_features = self.process_enroll(enroll)

        if x.dim() == 1:
            x = th.unsqueeze(x, 0)
            
        spec = self.wav2spec(x)
        out = self.conv2d(spec)
        out_list = []
        out = self.encoder[0](out)

        out = self.spk_fuse(out.transpose(2, 3), speaker_features).transpose(2, 3)
        out_list.append(out)

        for _, enc in enumerate(self.encoder[1:]):
            out = enc(out)
            out_list.append(out)
        B, N, T, F = out.shape
        out = self.tcn_layers(out.reshape(B, N, T*F))
        out = th.unsqueeze(out, -1)
        out_list = out_list[::-1]
        for idx, dec in enumerate(self.decoder):
            # return out_list[idx]
            out = dec(th.cat([out_list[idx], out], 1))
            # return out
            out = out[..., :-2, :]

        # -------------------
        # Pyramidal pooling
        out = self.avg_pool(out)
        # B, N, T, F = out.shape
        # upsample = nn.Upsample(size=(T, F), mode='bilinear')
        # pool_list = []
        # for avg in self.avg_pool:
        #     pool_list.append(upsample(avg(out)))
        # out = th.cat([out, *pool_list], 1)
        # -------------------- 

        out = self.avg_proj(out)
        out = self.deconv2d(out)[..., :-2, :]
        
        est_spec = torch.transpose(out, 2, 3)  # (batchsize, 2, F, T)
        B, N, F, T = est_spec.shape
        est_spec = torch.chunk(est_spec, 2, 1)  # [(B, 1, F, T), (B, 1, F, T)])
        est_spec = torch.complex(est_spec[0], est_spec[1]).squeeze()

        out = torch.istft(
            est_spec, n_fft=self.fft_len,
            hop_length=self.win_inc,
            win_length=self.win_len,
            window=self.window,
            center=False
        ).squeeze()

        return out


def test_covn2d_block():
    x = th.randn(2, 16, 257, 200)
    conv = Conv2dBlock()
    y = conv(x)
    print(y.shape)
    convtrans = ConvTrans2dBlock()
    z = convtrans(y)
    print(z.shape)
    
def test_dense_block():
    x = th.randn(2, 16, 257, 200)
    dense = DenseBlock(16, 32, "enc")
    y = dense(x)
    print(y.shape)
    
def test_tcn_block():
    x = th.randn(2, 384, 1000)
    tcn = TCNBlock(dilation=128)
    print(tcn(x).shape)


if __name__ == "__main__":
    # nnet = DenseUNet(
    #     win_len=1024,
    #     win_inc=512,
    #     fft_len=1024
    # )

    nnet = DenseUNet(
        win_len=512,
        win_inc=128,
        fft_len=512
    )
    x = th.randn(1, 32000)
    est1 = nnet(x, x)
    if est1.dim() < 2:
        est1 = est1.unsqueeze(0)

    print(est1)
    print(est1.shape)