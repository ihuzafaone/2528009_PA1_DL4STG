"""Autoformer (encoder-decoder) for Task 2, roll no 25280009 - v3 model file.

Written from scratch after Wu et al. (2021) and my own Q1 notebook
(SeriesDecomposition, delay scoring by FFT, delay aggregation).
Two mechanisms required by the handout:
  * progressive series decomposition (moving average) at the input and after
    every sub-layer, with decoder trend accumulation;
  * Auto-Correlation: delay scores by FFT, top-k delays, softmax with a learned
    temperature, aggregation z_t = sum_j a_j v_(t - tau_j).
New in v3 (my own choices, tested in q2_v3.py):
  * covariates enter through a small MLP followed by a depthwise temporal
    convolution (kernel 5, centred), so each step also sees nearby weather;
  * optional covariate-driven trend start: trend_init = context mean + linear(cov).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decomp(nn.Module):
    """x = seasonal + trend, trend = centred moving average with edge replication."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):  # [B,L,C]
        p = (self.k - 1) // 2
        xp = torch.cat([x[:, :1].expand(-1, p, -1), x, x[:, -1:].expand(-1, p, -1)], 1)
        tr = F.avg_pool1d(xp.transpose(1, 2), self.k, 1).transpose(1, 2)
        return x - tr, tr


class AutoCorrelation(nn.Module):
    def __init__(self, d, factor=2):
        super().__init__()
        self.factor = factor
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))
        self.raw_temp = nn.Parameter(torch.tensor(0.5413))  # softplus -> 1.0

    def forward(self, x, mem):
        B, L, d = x.shape
        q, k, v = self.wq(x), self.wk(mem), self.wv(mem)
        if mem.shape[1] < L:  # paper: zero-pad the shorter memory
            pad = (0, 0, 0, L - mem.shape[1])
            k, v = F.pad(k, pad), F.pad(v, pad)
        else:
            k, v = k[:, :L], v[:, :L]
        # R(tau) = sum_t q_t k_(t - tau), all delays at once with the FFT
        R = torch.fft.irfft(torch.fft.rfft(q, dim=1) * torch.fft.rfft(k, dim=1).conj(), n=L, dim=1)
        score = R.mean(-1) / d  # [B,L]
        K = max(1, int(self.factor * math.log(L)))
        top, tau = torch.topk(score, K, dim=1)
        w = torch.softmax(top / (F.softplus(self.raw_temp) + 1e-6), dim=1)  # [B,K]
        t = torch.arange(L, device=x.device)
        src = (t.view(1, 1, L) - tau.unsqueeze(-1)) % L  # [B,K,L]: read v at t - tau
        g = torch.gather(v.unsqueeze(1).expand(B, K, L, d), 2, src.unsqueeze(-1).expand(B, K, L, d))
        z = (w.view(B, K, 1, 1) * g).sum(1)
        return self.wo(z)


class SeasonNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        x = self.ln(x)
        return x - x.mean(1, keepdim=True)


def ffn(d, dr):
    return nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dr), nn.Linear(2 * d, d))


class Embed(nn.Module):
    """value token (conv k=3) + covariate MLP + depthwise temporal conv over covariates."""

    def __init__(self, n_cov, d, dr):
        super().__init__()
        self.val = nn.Conv1d(1, d, 3, padding=1, padding_mode="replicate", bias=False)
        self.cov = None
        if n_cov:
            self.cov = nn.Sequential(nn.Linear(n_cov, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            self.tconv = nn.Conv1d(d, d, 5, padding=2, padding_mode="replicate", groups=d)
        self.drop = nn.Dropout(dr)

    def forward(self, x, c):
        e = self.val(x.transpose(1, 2)).transpose(1, 2)
        if self.cov is not None:
            h = self.cov(c)
            e = e + h + self.tconv(h.transpose(1, 2)).transpose(1, 2)
        return self.drop(e)


class EncLayer(nn.Module):
    def __init__(self, d, k, factor, dr):
        super().__init__()
        self.ac, self.ff = AutoCorrelation(d, factor), ffn(d, dr)
        self.d1, self.d2, self.dr = Decomp(k), Decomp(k), nn.Dropout(dr)

    def forward(self, x):
        x, _ = self.d1(x + self.dr(self.ac(x, x)))
        x, _ = self.d2(x + self.dr(self.ff(x)))
        return x


class DecLayer(nn.Module):
    def __init__(self, d, k, factor, dr):
        super().__init__()
        self.sa, self.ca, self.ff = AutoCorrelation(d, factor), AutoCorrelation(d, factor), ffn(d, dr)
        self.d1, self.d2, self.d3 = Decomp(k), Decomp(k), Decomp(k)
        self.dr = nn.Dropout(dr)
        self.tproj = nn.Conv1d(d, 1, 3, padding=1, padding_mode="replicate", bias=False)

    def forward(self, x, enc):
        x, t1 = self.d1(x + self.dr(self.sa(x, x)))
        x, t2 = self.d2(x + self.dr(self.ca(x, enc)))
        x, t3 = self.d3(x + self.dr(self.ff(x)))
        return x, self.tproj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)


class Autoformer(nn.Module):
    def __init__(self, n_cov, seq_len=96, label_len=48, pred_len=168, d=16, k=25,
                 factor=2, dr=0.1, e_layers=1, d_layers=1, trend="cov"):
        super().__init__()
        self.lab, self.H, self.trend = label_len, pred_len, trend
        self.decomp = Decomp(k)
        self.enc_emb, self.dec_emb = Embed(n_cov, d, dr), Embed(n_cov, d, dr)
        self.enc = nn.ModuleList(EncLayer(d, k, factor, dr) for _ in range(e_layers))
        self.dec = nn.ModuleList(DecLayer(d, k, factor, dr) for _ in range(d_layers))
        self.en, self.dn = SeasonNorm(d), SeasonNorm(d)
        self.proj = nn.Linear(d, 1)
        self.cov_trend = nn.Linear(n_cov, 1) if (trend == "cov" and n_cov) else None

    def forward(self, x, ce, cd):  # x [B,L,1], ce [B,L,M], cd [B,lab+H,M] -> [B,H]
        seas, tr = self.decomp(x)
        mean = x.mean(1, keepdim=True).expand(-1, self.H, -1)
        s_init = torch.cat([seas[:, -self.lab:], torch.zeros_like(mean)], 1)
        t_init = torch.cat([tr[:, -self.lab:], mean], 1)
        if self.cov_trend is not None:
            t_init = t_init + self.cov_trend(cd)
        e = self.enc_emb(x, ce)
        for layer in self.enc:
            e = layer(e)
        e = self.en(e)
        z, trend = self.dec_emb(s_init, cd), t_init
        for layer in self.dec:
            z, t = layer(z, e)
            trend = trend + t
        return (trend + self.proj(self.dn(z)))[:, -self.H:, 0]


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
