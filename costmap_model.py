from __future__ import annotations
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class UNetCostMap(nn.Module):
    def __init__(self, in_ch: int = 4, out_ch: int = 1, base: int = 16):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base * 2, base * 4))
        self.up1t = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.up1c = DoubleConv(base * 4, base * 2)
        self.up2t = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.up2c = DoubleConv(base * 2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u1 = self.up1t(x3)
        # Pad if needed to handle odd sizes
        if u1.shape[-2:] != x2.shape[-2:]:
            diffY = x2.shape[-2] - u1.shape[-2]
            diffX = x2.shape[-1] - u1.shape[-1]
            u1 = F.pad(u1, (0, diffX, 0, diffY))
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up1c(u1)
        u2 = self.up2t(u1)
        if u2.shape[-2:] != x1.shape[-2:]:
            diffY = x1.shape[-2] - u2.shape[-2]
            diffX = x1.shape[-1] - u2.shape[-1]
            u2 = F.pad(u2, (0, diffX, 0, diffY))
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up2c(u2)
        logits = self.outc(u2)
        return logits


def postprocess_cost(logits: torch.Tensor) -> np.ndarray:
    """Apply softplus and clamp to [0, 3], return numpy HxW (float32)."""
    with torch.no_grad():
        if logits.dim() == 4:  # N,C,H,W → use first sample/channel
            logits = logits[0, 0]
        elif logits.dim() == 3:  # C,H,W
            logits = logits[0]
        x = F.softplus(logits)
        x = torch.clamp(x, 0.0, 3.0)
        arr = x.detach().cpu().numpy().astype(np.float32)
        return arr


def load_costmap_model(path: str = "costmap_unet.pt") -> Optional[nn.Module]:
    try:
        if not os.path.exists(path):
            return None
        model = UNetCostMap(in_ch=4, out_ch=1)
        sd = torch.load(path, map_location="cpu")
        # Allow loading from either plain state_dict or a wrapped dict
        if isinstance(sd, dict) and all(k.startswith("inc.") or k.startswith("down") or k.startswith("up") or k.startswith("outc") for k in sd.keys()):
            model.load_state_dict(sd)
        elif isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
            model.load_state_dict(sd["state_dict"])  # type: ignore[arg-type]
        else:
            model.load_state_dict(sd)  # best effort
        model.eval()
        return model
    except Exception:
        return None

