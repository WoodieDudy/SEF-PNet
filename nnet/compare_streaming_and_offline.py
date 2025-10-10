import torch as th
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

from DPCCN import DenseUNet as OFFLINEDenseUNet
from DPCCN_streaming import DenseUNet
from streaming_stuff import make_streaming

th.manual_seed(0)
BATCH = 2
SIG_LEN = 32768
CHUNK  = 128

offline = OFFLINEDenseUNet()
offline.eval()

streaming = DenseUNet()
streaming.load_state_dict(offline.state_dict(), strict=False)
make_streaming(streaming)
streaming.eval()
# streaming.reset()

audio = th.randn(BATCH, SIG_LEN)
spk_features = th.rand(BATCH, 1, 256, 1)

zeros = th.zeros(BATCH, 384, device=audio.device, dtype=audio.dtype)
audio2 = th.cat([zeros, audio], dim=1) # TODO
with th.no_grad():
    off_out = offline(audio2, spk_features)
print(f"{off_out.shape=}")

parts = []
with th.no_grad():
    for p in range(0, SIG_LEN, CHUNK):
        chunk = audio[..., p:p+CHUNK]
        st_out = streaming(chunk, spk_features)
        if st_out is None:
            continue
        parts.append(st_out)


print([x.shape for x in parts[:3]])
str_out = th.cat(parts, dim=-1)
print(f"{str_out.shape=}")
print(f"{off_out.shape=}")

off_out = off_out[..., :str_out.size(1)]
diff = (off_out - str_out).abs()

print(f"{diff.max()=}")
print(f"{diff.mean()=}")


