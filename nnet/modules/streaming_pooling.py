import torch
import torch.nn as nn


class StreamingTemporalAvgPool(nn.Module):
    def __init__(self, pool_sizes=(4, 8, 16, 32)):
        super().__init__()
        self.pool_sizes = list(pool_sizes)
        self.max_w = max(self.pool_sizes)
        self.register_buffer('ps_buffer', None)
        self.buf_idx = 0
        self.buf_len = 0

    def reset(self):
        self.ps_buffer = None
        self.buf_idx = 0
        self.buf_len = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, F = x.shape

        if T > 1:
            x_f = x.to(torch.float32)
            zero = torch.zeros((B, C, 1, F), dtype=x_f.dtype, device=x.device)
            ps = torch.cat([zero, torch.cumsum(x_f, dim=2)], dim=2)
            end_sum = ps[:, :, 1:T + 1, :]

            outs = []
            lens_base = torch.arange(1, T + 1, device=x.device)
            for w in self.pool_sizes:
                if w >= T:
                    start_sum = torch.zeros_like(end_sum)
                else:
                    pad = torch.zeros((B, C, w, F), dtype=x_f.dtype, device=x.device)
                    part = ps[:, :, 1:T + 1 - w, :]
                    start_sum = torch.cat([pad, part], dim=2)

                sum_w = end_sum - start_sum
                denom = torch.minimum(lens_base, torch.tensor(w, device=x.device))\
                             .view(1,1,T,1).to(sum_w.dtype)
                outs.append((sum_w / denom).to(x.dtype))

            return torch.cat(outs, dim=1)

        xf = x[:, :, 0, :].to(torch.float32)

        if self.ps_buffer is None:
            buf_size = self.max_w + 1
            buf = torch.zeros((B, C, buf_size, F), dtype=xf.dtype, device=x.device)
            buf[:, :, 0, :] = xf
            self.ps_buffer = buf
            self.buf_idx = 1
            self.buf_len = 1
        else:
            with torch.no_grad():
                prev_idx = (self.buf_idx - 1) % self.ps_buffer.size(2)
                last_ps = self.ps_buffer[:, :, prev_idx, :]
                new_ps = last_ps + xf
            self.ps_buffer[:, :, self.buf_idx, :] = new_ps
            self.buf_idx = (self.buf_idx + 1) % self.ps_buffer.size(2)
            self.buf_len = min(self.buf_len + 1, self.ps_buffer.size(2))

        outs = []
        buf_size = self.ps_buffer.size(2)
        last = (self.buf_idx - 1) % buf_size
        for w in self.pool_sizes:
            if self.buf_len < w:
                sum_w = self.ps_buffer[:, :, last, :]
                denom = float(self.buf_len)
            else:
                prev = (last - w) % buf_size
                sum_w = self.ps_buffer[:, :, last, :] - self.ps_buffer[:, :, prev, :]
                denom = float(w)
            outs.append((sum_w / denom).unsqueeze(2).to(x.dtype))

        return torch.cat(outs, dim=1)


def test_stream_vs_offline_avg_pool(
    B=2, C=3, T=20, F=5, pool_sizes=(4,8,16,32), atol=1e-5
):
    torch.manual_seed(42)
    x = torch.randn(B, C, T, F)

    pool = StreamingTemporalAvgPool(pool_sizes=pool_sizes)
    pool.eval()

    with torch.no_grad():
        offline_out = pool(x)

    pool.reset()
    streaming_steps = []
    for t in range(T):
        frame = x[:, :, t : t+1, :]
        out_t = pool(frame)
        streaming_steps.append(out_t)
    streaming_out = torch.cat(streaming_steps, dim=2)

    if not torch.allclose(streaming_out, offline_out, atol=atol):
        diff = (streaming_out - offline_out).abs().max()
        print(f"Mismatch! max abs diff = {diff.item():.3e}")
    else:
        print(f"OK: streaming ≡ offline (max abs diff = "
              f"{(streaming_out - offline_out).abs().max().item():.3e})")


if __name__ == "__main__":
    test_stream_vs_offline_avg_pool(4, 3, 310, 5)
