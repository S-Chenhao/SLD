"""SLD inference architecture, preserving the supplied checkpoint parameter names.

Extracted from the SLD implementation used in the experiment notebooks.
This module performs no data loading, training, or device selection on import.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

class ShiftMean(nn.Module):
    """Normalize or denormalize two-channel fields with saved mean/std buffers."""

    def __init__(self, mean, std):
        super().__init__()
        mean = torch.as_tensor(mean).float().view(1, 1, -1, 1, 1)
        std = torch.as_tensor(std).float().view(1, 1, -1, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x, mode: str):
        if x.dim() == 4:
            x_in = x.unsqueeze(0)
            squeeze_back = True
        else:
            x_in = x
            squeeze_back = False
        if mode == 'sub':
            y = (x_in - self.mean) / (self.std + 1e-12)
        elif mode == 'add':
            y = x_in * (self.std + 1e-12) + self.mean
        else:
            raise NotImplementedError
        return y.squeeze(0) if squeeze_back else y

def haar_iwt2x(ll, lh, hl, hh):
    """
    Inverse Haar wavelet synthesis (2x upsampling).
    ll/lh/hl/hh: [B, C, H, W] -> out: [B, C, 2H, 2W]
    """
    x00 = (ll + lh + hl + hh) * 0.5
    x01 = (ll - lh + hl - hh) * 0.5
    x10 = (ll + lh - hl - hh) * 0.5
    x11 = (ll - lh - hl + hh) * 0.5
    (B, C, H, W) = ll.shape
    out = torch.empty(B, C, 2 * H, 2 * W, device=ll.device, dtype=ll.dtype)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out

class HaarPyramidSynthesis(nn.Module):

    def __init__(self, out_channels: int, scale: int):
        super().__init__()
        assert scale >= 2 and scale & scale - 1 == 0, 'scale must be power of 2'
        self.C = out_channels
        self.scale = scale
        self.L = int(math.log2(scale))

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        (B, ch, H, W) = packed.shape
        C = self.C
        assert ch == C * self.scale ** 2, f'expected {C * self.scale ** 2} channels, got {ch}'
        idx = 0
        ll = packed[:, idx:idx + C]
        idx += C
        lh = packed[:, idx:idx + C]
        idx += C
        hl = packed[:, idx:idx + C]
        idx += C
        hh = packed[:, idx:idx + C]
        idx += C
        x = haar_iwt2x(ll, lh, hl, hh)
        for m in range(1, self.L):
            r = 2 ** m
            per = C * (r * r)
            lh_p = packed[:, idx:idx + per]
            idx += per
            hl_p = packed[:, idx:idx + per]
            idx += per
            hh_p = packed[:, idx:idx + per]
            idx += per
            lh = F.pixel_shuffle(lh_p, r)
            hl = F.pixel_shuffle(hl_p, r)
            hh = F.pixel_shuffle(hh_p, r)
            x = haar_iwt2x(x, lh, hl, hh)
        assert idx == ch
        return x

class WaveletSRHead(nn.Module):

    def __init__(self, in_ch, out_ch, scale, conv3x3_factory):
        super().__init__()
        self.proj = conv3x3_factory(in_ch, out_ch * scale ** 2)
        self.synth = HaarPyramidSynthesis(out_channels=out_ch, scale=scale)

    def forward(self, feat):
        packed = self.proj(feat)
        return self.synth(packed)

class ConvNeXt_GRU_Cell(nn.Module):
    """
    ConvNeXt-style GRU cell used as a continuous-time GRU derivative:
        dh/dt = (1 - z) * (h_tilde - h) + alpha * Lap(h) + beta_x * Dx(h) + beta_y * Dy(h)

    Notes:
        - Dx, Dy, Lap are fixed depthwise conv operators
        - alpha, beta_x, beta_y are learnable per-channel parameters
    """

    def __init__(self, dim, custom_act=None):
        super().__init__()
        self.dim = dim
        self.gate_conv = nn.Sequential(nn.Conv2d(dim * 2, dim * 2, kernel_size=7, padding=3, groups=dim * 2, padding_mode='circular'), nn.GroupNorm(1, dim * 2), nn.Conv2d(dim * 2, dim * 2, kernel_size=1), nn.Sigmoid())
        self.candidate_dw = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, padding=3, groups=dim * 2, padding_mode='circular')
        self.candidate_norm = nn.GroupNorm(1, dim * 2)
        self.candidate_pw1 = nn.Conv2d(dim * 2, dim * 4, kernel_size=1)
        self.act = custom_act if custom_act is not None else nn.GELU()
        self.candidate_pw2 = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.dx = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False, padding_mode='circular')
        self.dy = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False, padding_mode='circular')
        self.lap = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False, padding_mode='circular')
        self._init_fixed_kernels()
        for p in self.dx.parameters():
            p.requires_grad = False
        for p in self.dy.parameters():
            p.requires_grad = False
        for p in self.lap.parameters():
            p.requires_grad = False
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta_x = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta_y = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def _init_fixed_kernels(self):
        """
        Central-difference-like first derivatives and 5-point Laplacian.
        """
        device = self.dx.weight.device
        dtype = self.dx.weight.dtype
        dx_kernel = torch.tensor([[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=dtype, device=device)
        dy_kernel = torch.tensor([[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=dtype, device=device)
        lap_kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)
        dx_weight = torch.zeros_like(self.dx.weight.data)
        dy_weight = torch.zeros_like(self.dy.weight.data)
        lap_weight = torch.zeros_like(self.lap.weight.data)
        dx_weight[:, 0, :, :] = dx_kernel
        dy_weight[:, 0, :, :] = dy_kernel
        lap_weight[:, 0, :, :] = lap_kernel
        with torch.no_grad():
            self.dx.weight.copy_(dx_weight)
            self.dy.weight.copy_(dy_weight)
            self.lap.weight.copy_(lap_weight)

    def forward(self, h, x_input):
        """
        h: [B, dim, H, W]
        x_input: [B, dim, H, W]
        """
        combined = torch.cat([h, x_input], dim=1)
        gates = self.gate_conv(combined)
        (r, z) = torch.split(gates, self.dim, dim=1)
        h_reset = r * h
        combined_reset = torch.cat([h_reset, x_input], dim=1)
        out = self.candidate_dw(combined_reset)
        out = self.candidate_norm(out)
        out = self.candidate_pw1(out)
        out = self.act(out)
        out = self.candidate_pw2(out)
        dh_dt_main = (1 - z) * (out - h)
        dh_dt_basis = self.alpha * self.lap(h) + self.beta_x * self.dx(h) + self.beta_y * self.dy(h)
        dh_dt = dh_dt_main + dh_dt_basis
        return dh_dt

class TemporalResidualRefiner(nn.Module):
    """
    对 [T,B,C,H,W] 在时间维做轻量残差修正
    """

    def __init__(self, channels, hidden=None, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, 'kernel_size should be odd'
        hidden = hidden or max(16, channels * 4)
        pad = kernel_size // 2
        self.in_proj = nn.Conv1d(channels, hidden, kernel_size=1)
        self.tconv1 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=pad)
        self.act = nn.GELU()
        self.tconv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=pad)
        self.out_proj = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x):
        """
        x: [T,B,C,H,W]
        """
        (T, B, C, H, W) = x.shape
        z = x.permute(1, 3, 4, 2, 0).contiguous()
        z = z.view(B * H * W, C, T)
        z0 = z
        z = self.in_proj(z)
        z = self.act(self.tconv1(z))
        z = self.act(self.tconv2(z))
        z = self.out_proj(z)
        z = z + z0
        z = z.view(B, H, W, C, T)
        z = z.permute(4, 0, 3, 1, 2).contiguous()
        return z

class temporal_sr(nn.Module):
    """Interpolate time, then refine with temporal residual convolutions."""

    def __init__(self, t_upscale_factor, channels=2, hidden=32, kernel_size=5):
        super().__init__()
        self.t_upscale_factor = t_upscale_factor
        self.refiner = TemporalResidualRefiner(channels=channels, hidden=hidden, kernel_size=kernel_size)

    def forward(self, x):
        (t, b, c, h, w) = x.shape
        z = x.permute(1, 3, 4, 2, 0)
        z = z.contiguous().view(b * h * w, c, t)
        z = F.interpolate(z, scale_factor=self.t_upscale_factor, mode='linear', align_corners=True)
        z = z.contiguous().view(b, h, w, c, t * self.t_upscale_factor)
        z = z.permute(4, 0, 3, 1, 2).contiguous()
        z = self.refiner(z)
        return z

class SLD(nn.Module):
    """SLD reconstruction model. Input [T,B,2,H,W]; output [4T,B,2,8H,8W] with the default refinement factors."""

    def __init__(self, n_feats, n_layers, upscale_factor, shift_mean_paras_lr, shift_mean_paras_hr, custom_act=None):
        super().__init__()
        (self.n_next_blocks, self.n_resblock) = n_layers
        (self.t_up_factor, self.s_up_factor) = upscale_factor
        (mean_lres, std_lres) = shift_mean_paras_lr
        (mean_hres, std_hres) = shift_mean_paras_hr
        self.tsr = temporal_sr(self.t_up_factor)
        self.feat_extract = nn.Conv2d(2, n_feats, kernel_size=3, padding=1, padding_mode='circular')
        self.h0_proj = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.gru_cell = ConvNeXt_GRU_Cell(n_feats, custom_act=custom_act)
        # Keep the legacy checkpoint namespace for the shared recurrent cell.
        self.ode_wrapper = nn.Module()
        self.ode_wrapper.add_module('cell', self.gru_cell)

        def wn_circ_conv3x3(in_ch, out_ch):
            return weight_norm(nn.Conv2d(in_ch, out_ch, 3, padding=1, padding_mode='circular'))
        self.tail = WaveletSRHead(n_feats, out_ch=2, scale=self.s_up_factor, conv3x3_factory=wn_circ_conv3x3)
        self.skip = WaveletSRHead(2, out_ch=2, scale=self.s_up_factor, conv3x3_factory=wn_circ_conv3x3)
        self.shift_in = ShiftMean(mean_lres, std_lres)
        self.shift_out = ShiftMean(mean_hres, std_hres)

    def forward(self, x, t_steps=None, odeint=None, method='rk4', use_control=True):
        """
        x: [T_in, B, 2, H, W]
        Returns: [T_out, B, 2, H*s_up, W*s_up]
        """
        (T_in, B, C, H, W) = x.shape
        x = self.shift_in(x, mode='sub')
        x_lr_up = self.tsr(x)
        T_out = x_lr_up.shape[0]
        if t_steps is None:
            t_steps = torch.linspace(0, float(T_out - 1), T_out, device=x.device, dtype=x.dtype)
        else:
            t_steps = t_steps.to(device=x.device, dtype=x.dtype)
            if t_steps.numel() != T_out:
                raise ValueError(f't_steps length {t_steps.numel()} must equal T_out {T_out}')
        x_flat = x_lr_up.reshape(T_out * B, C, H, W)
        U_flat = self.feat_extract(x_flat)
        U_seq = U_flat.view(T_out, B, -1, H, W)
        h = self.h0_proj(U_seq[0])
        hs = []
        dt = t_steps[1] - t_steps[0]
        for k in range(T_out):
            k1 = self.gru_cell(h, U_seq[k])
            h_mid = h + 0.5 * dt * k1
            k2 = self.gru_cell(h_mid, U_seq[k])
            h = h + dt * k2
            hs.append(h)
        h_stack = torch.stack(hs, dim=0)
        h_flat = h_stack.reshape(T_out * B, -1, H, W)
        xlr_flat = x_lr_up.reshape(T_out * B, 2, H, W)
        y_flat = self.tail(h_flat) + self.skip(xlr_flat)
        y = y_flat.view(T_out, B, 2, H * self.s_up_factor, W * self.s_up_factor)
        y = self.shift_out(y, mode='add')
        return y

# Historical class name for code that imports the notebook implementation.
PhySR_ODE_ConvGRU = SLD


def build_model(
    n_feats=64,
    n_layers=(6, 3),
    upscale_factor=(4, 8),
    shift_mean_paras_lr=None,
    shift_mean_paras_hr=None,
    custom_act=None,
    device=None,
):
    """Build SLD; strict checkpoint loading restores saved normalization buffers.

    Defaults reproduce the supplied architecture. The optional device argument
    only moves the constructed model; it never changes global device settings.
    """
    if shift_mean_paras_lr is None:
        shift_mean_paras_lr = (torch.zeros(2), torch.ones(2))
    if shift_mean_paras_hr is None:
        shift_mean_paras_hr = (torch.zeros(2), torch.ones(2))
    model = SLD(
        n_feats=n_feats,
        n_layers=n_layers,
        upscale_factor=upscale_factor,
        shift_mean_paras_lr=shift_mean_paras_lr,
        shift_mean_paras_hr=shift_mean_paras_hr,
        custom_act=custom_act,
    )
    return model.to(device) if device is not None else model
