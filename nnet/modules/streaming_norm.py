import torch
import torch.nn as nn


class StreamingInstanceNorm(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps   = float(eps)

        self.register_buffer('mean',    None)
        self.register_buffer('mean_sq', None)
        self.register_buffer('step',    torch.tensor(0, dtype=torch.long))

    def reset(self):
        self.step = torch.tensor(0, dtype=torch.long, device=self.step.device)
        self.sum     = None
        self.sum_sq  = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.size(2) > 1:
            B, C, T, F = x.shape
            psum    = x.cumsum(dim=2)
            psq_sum = (x * x).cumsum(dim=2)

            sum_feat    = psum.sum(dim=3)
            sum_sq_feat = psq_sum.sum(dim=3)

            counts = torch.arange(1, T+1, device=x.device, dtype=x.dtype).view(1,1,T) * F

            mean_t = sum_feat    / counts
            var_t  = sum_sq_feat / counts - mean_t.pow(2)

            mean_t = mean_t.unsqueeze(-1)
            std_t  = torch.sqrt(var_t.clamp(min=0.0) + self.eps).unsqueeze(-1)
            return (x - mean_t) / std_t

        if x.ndim == 3 and x.size(2) > 1:
            B, C, T = x.shape
            x4 = x.unsqueeze(-1)
            psum    = x4.cumsum(dim=2)
            psq_sum = (x4 * x4).cumsum(dim=2)

            sum_feat    = psum.squeeze(-1)
            sum_sq_feat = psq_sum.squeeze(-1)

            counts = torch.arange(1, T+1, device=x.device, dtype=x.dtype).view(1,1,T)

            mean_t = sum_feat    / counts
            var_t  = sum_sq_feat / counts - mean_t.pow(2)

            mean_t = mean_t
            std_t  = torch.sqrt(var_t.clamp(min=0.0) + self.eps)
            return (x - mean_t) / std_t

        return self._forward_frame(x)


    def _forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            xf      = x[:, :, 0, :]
            sum2    = xf.sum(dim=2)
            sum2_sq = (xf**2).sum(dim=2)
            F       = x.size(-1)
        else:
            xf      = x[:, :, 0]
            sum2    = xf
            sum2_sq = xf**2
            F       = 1

        B, C = sum2.shape
        device = x.device

        if not hasattr(self, 'sum') or self.sum is None or self.sum.shape[0] != B:
            self.sum    = sum2.detach().clone()
            self.sum_sq = sum2_sq.detach().clone()
            self.step   = torch.ones(B, dtype=torch.long, device=device)
        else:
            self.sum    = self.sum    + sum2
            self.sum_sq = self.sum_sq + sum2_sq
            self.step   = self.step   + 1

        counts   = self.step.float().unsqueeze(1) * F
        mean_hat = self.sum    / counts
        var_hat  = self.sum_sq / counts - mean_hat.pow(2)
        std_hat  = torch.sqrt(var_hat.clamp(min=0.0) + self.eps)

        if x.ndim == 4:
            m = mean_hat.view(B, C, 1, 1)
            s = std_hat.view(B, C, 1, 1)
        else:
            m = mean_hat.view(B, C, 1)
            s = std_hat.view(B, C, 1)

        return (x - m) / s


class EMAStreamingInstanceNorm(nn.Module):
    def __init__(self, alpha: float = 0.05, eps: float = 1e-4):
        super().__init__()
        self.alpha = float(alpha)
        self.eps   = float(eps)
        self.register_buffer('mean', None)
        self.register_buffer('var',  None)

    def reset(self):
        self.mean = None
        self.var  = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.size(2) > 1:
            return self._offline_fast_4d(x)
        if x.ndim == 3 and x.size(2) > 1:
            return self._offline_fast_3d(x)
        return self._forward_frame(x)

    def _offline_fast_4d(self, x: torch.Tensor) -> torch.Tensor:
        self.reset()
        B, C, T, Fd = x.shape
        device, dtype = x.device, x.dtype
        lam   = 1.0 - self.alpha

        frame_mean = x.mean(dim=-1)
        frame_var  = x.var(dim=-1, unbiased=False)

        idx = torch.arange(T, device=device, dtype=dtype)
        lam_pow        = lam ** idx
        lam_pow_shift1 = lam_pow * lam
        inv_lam_pow    = lam_pow.reciprocal()

        lam_pow        = lam_pow.view(1, 1, T)
        lam_pow_shift1 = lam_pow_shift1.view(1, 1, T)
        inv_lam_pow    = inv_lam_pow.view(1, 1, T)

        prefix_mean = torch.cumsum(frame_mean * inv_lam_pow, dim=2)
        prefix_var  = torch.cumsum(frame_var  * inv_lam_pow, dim=2)

        mu0  = frame_mean[:, :, 0:1]
        var0 = frame_var [:, :, 0:1]

        mean_hat = lam_pow_shift1 * mu0 + self.alpha * lam_pow * prefix_mean
        var_hat  = lam_pow_shift1 * var0 + self.alpha * lam_pow * prefix_var

        self.mean = mean_hat[:, :, -1].detach()
        self.var  = var_hat [:, :, -1].detach()

        std_hat = torch.sqrt(var_hat.clamp(min=0.0) + self.eps)
        out = (x - mean_hat.unsqueeze(-1)) / std_hat.unsqueeze(-1)
        return out

    def _offline_fast_3d(self, x: torch.Tensor) -> torch.Tensor:
        self.reset()
        B, C, T = x.shape
        device, dtype = x.device, x.dtype
        lam   = 1.0 - self.alpha

        idx = torch.arange(T, device=device, dtype=dtype)
        lam_pow     = lam ** idx
        inv_lam_pow = lam_pow.reciprocal()
        lam_pow     = lam_pow.view(1, 1, T)
        inv_lam_pow = inv_lam_pow.view(1, 1, T)

        x_seq = x
        prefix_x = torch.cumsum(x_seq * inv_lam_pow, dim=2)
        x0 = x_seq[:, :, 0:1]
        lam_pow_shift1 = lam_pow * lam
        mean_hat = lam_pow_shift1 * x0 + self.alpha * lam_pow * prefix_x

        m_prev = torch.cat([x0, mean_hat[:, :, :-1]], dim=2)
        e2 = (x_seq - m_prev).pow(2)
        e2[:, :, 0] = 0.0

        prefix_e = torch.cumsum(e2 * inv_lam_pow, dim=2)
        var_hat = self.alpha * lam_pow * prefix_e

        self.mean = mean_hat[:, :, -1].detach()
        self.var  = var_hat[:,  :, -1].detach()

        std_hat = torch.sqrt(var_hat.clamp(min=0.0) + self.eps)
        out = (x - mean_hat) / std_hat
        return out

    def _forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            xf         = x[:, :, 0, :]
            frame_mean = xf.mean(dim=2)
            frame_var  = xf.var(dim=2, unbiased=False)
        else:  # F==1
            xf         = x[:, :, 0]
            frame_mean = xf
            if self.mean is None:
                frame_var = torch.zeros_like(xf)
            else:
                frame_var = (frame_mean - self.mean) ** 2

        if self.mean is None:
            mean_hat = frame_mean
            var_hat  = frame_var
        else:
            lam      = 1.0 - self.alpha
            mean_hat = lam * self.mean + self.alpha * frame_mean
            var_hat  = lam * self.var  + self.alpha * frame_var

        self.mean = mean_hat.detach()
        self.var  = var_hat.detach()

        std_hat = torch.sqrt(var_hat.clamp(min=0.0) + self.eps)
        if x.ndim == 4:
            m = mean_hat.view(*mean_hat.shape, 1, 1)
            s = std_hat.view(*std_hat.shape, 1, 1)
        else:
            m = mean_hat.view(*mean_hat.shape, 1)
            s = std_hat.view(*std_hat.shape, 1)
        return (x - m) / s


class StreamingInstanceNormF(nn.Module):
    def __init__(self, alpha: float = 0.05, eps: float = 1e-4):
        super().__init__()
        self.alpha = float(alpha)
        self.eps   = float(eps)

        self.register_buffer('sum',     None)
        self.register_buffer('sum_sq',  None)
        self.register_buffer('step',    torch.tensor(0, dtype=torch.long))

    def reset(self):
        self.sum     = None
        self.sum_sq  = None
        self.step    = torch.tensor(0, dtype=torch.long, device=self.step.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4 and x.size(2) > 1:
            B, C, T, F = x.shape

            psum    = x.cumsum(dim=2)
            psq_sum = (x * x).cumsum(dim=2)

            counts  = torch.arange(1, T + 1, device=x.device,
                                   dtype=x.dtype).view(1, 1, T, 1)

            mean_t  = psum    / counts
            var_t   = psq_sum / counts - mean_t.pow(2)

            std_t   = torch.sqrt(var_t.clamp(min=0.0) + self.eps)
            return (x - mean_t) / std_t

        if x.ndim == 3 and x.size(2) > 1:
            B, C, T = x.shape
            x4 = x.unsqueeze(-1)

            psum    = x4.cumsum(dim=2)
            psq_sum = (x4 * x4).cumsum(dim=2)

            counts  = torch.arange(1, T + 1, device=x.device,
                                   dtype=x.dtype).view(1, 1, T, 1)

            mean_t  = psum    / counts
            var_t   = psq_sum / counts - mean_t.pow(2)

            std_t   = torch.sqrt(var_t.clamp(min=0.0) + self.eps)
            return (x4 - mean_t)[:, :, :, 0] / std_t[:, :, :, 0]

        return self._forward_frame(x)

    def _forward_frame(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            xf        = x[:, :, 0, :]
            sum_inc   = xf
            sum_sq_inc= xf ** 2
            B, C, F   = xf.shape
            reshape_broadcast = lambda t: t.view(B, C, 1, F)
            counts_shape = (B, 1, 1)
        else:
            xf        = x[:, :, 0]
            sum_inc   = xf
            sum_sq_inc= xf ** 2
            B, C      = xf.shape
            F         = 1
            reshape_broadcast = lambda t: t.view(B, C, 1)
            counts_shape = (B, 1)

        device = x.device

        if (self.sum is None or
            self.sum.shape != sum_inc.shape):
            self.sum     = sum_inc.detach().clone()
            self.sum_sq  = sum_sq_inc.detach().clone()
            self.step    = torch.ones(B, dtype=torch.long, device=device)
        else:
            self.sum    += sum_inc
            self.sum_sq += sum_sq_inc
            self.step   += 1

        counts   = self.step.float().view(*counts_shape)
        mean_hat = self.sum    / counts
        var_hat  = self.sum_sq / counts - mean_hat.pow(2)
        std_hat  = torch.sqrt(var_hat.clamp(min=0.0) + self.eps)

        m = reshape_broadcast(mean_hat)
        s = reshape_broadcast(std_hat)
        return (x - m) / s
    

def test_norm_streaming_vs_offline_and_batch_independence(norm, shape, alpha=0.01, eps=1e-4, tol=1e-4):

    torch.manual_seed(0)
    x = torch.randn(shape)

    norm.reset()
    y_off = norm(x)
    norm.reset()
    if x.ndim == 4:
        frames = [norm(x[:, :, t:t+1, :]) for t in range(x.size(2))]
    else:
        frames = [norm(x[:, :, t:t+1]) for t in range(x.size(2))]
    y_stream = torch.cat(frames, dim=2)

    max_diff = (y_off - y_stream).abs().max().item()
    print(f"Shape {shape}: offline vs streaming max|diff| = {max_diff:.6f}")
    assert max_diff < tol, f"Offline vs streaming mismatch: {max_diff:.6f} > {tol}"

    B = shape[0]
    for i in range(B):
        single = x[i:i+1]
        norm.reset()
        y_single_off = norm(single)
        norm.reset()
        if single.ndim == 4:
            frames = [norm(single[:, :, t:t+1, :]) for t in range(single.size(2))]
        else:
            frames = [norm(single[:, :, t:t+1]) for t in range(single.size(2))]
        y_single_str = torch.cat(frames, dim=2)

        y_batch = y_stream[i:i+1]

        d_off = (y_batch - y_single_off).abs().max().item()
        d_str = (y_batch - y_single_str).abs().max().item()
        print(f" Sample {i}: batch vs single offline {d_off:.6f}, streaming {d_str:.6f}")
        assert d_off < tol, f"Batch-independence offline fail for sample {i}: {d_off:.6f} > {tol}"
        assert d_str < tol, f"Batch-independence streaming fail for sample {i}: {d_str:.6f} > {tol}"


if __name__ == "__main__":
    norm = StreamingInstanceNorm()
    test_norm_streaming_vs_offline_and_batch_independence(norm, (4, 2, 200, 3))
    test_norm_streaming_vs_offline_and_batch_independence(norm, (3, 3, 200))

    norm = EMAStreamingInstanceNorm()
    norm.eval()
    test_norm_streaming_vs_offline_and_batch_independence(norm, (4, 2, 400, 3))
    test_norm_streaming_vs_offline_and_batch_independence(norm, (3, 3, 200))

    norm = StreamingInstanceNormF()
    test_norm_streaming_vs_offline_and_batch_independence(norm, (4, 2, 200, 3))
    test_norm_streaming_vs_offline_and_batch_independence(norm, (3, 3, 200))
