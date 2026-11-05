from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from crack_reconstruction.image.model_v2 import OmniTCRV2

class DUTRContextV3(nn.Module):
    """DUTR v3 2-D context adapter on a frozen/warm-started v2 frontend.

    Adds a gated multi-dilation RGB+probability context correction. The adapter
    output is zero-initialized, so initialization is exactly the v2 predictor.
    """
    def __init__(self, base_channels=32, ctx_channels=24, max_context=1.25):
        super().__init__()
        self.base = OmniTCRV2(base_channels=base_channels)
        self.max_context=float(max_context)
        c=ctx_channels
        self.ctx_stem=nn.Sequential(
            nn.Conv2d(4,c,3,padding=1,bias=False), nn.GroupNorm(6,c), nn.GELU())
        self.d1=nn.Conv2d(c,c,3,padding=1,groups=c,bias=False)
        self.d3=nn.Conv2d(c,c,3,padding=3,dilation=3,groups=c,bias=False)
        self.d7=nn.Conv2d(c,c,3,padding=7,dilation=7,groups=c,bias=False)
        self.mix=nn.Sequential(nn.Conv2d(3*c,c,1,bias=False),nn.GroupNorm(6,c),nn.GELU(),nn.Conv2d(c,c,3,padding=1,bias=False),nn.GroupNorm(6,c),nn.GELU())
        self.context_head=nn.Conv2d(c,1,1)
        self.context_gate=nn.Conv2d(c,1,1)
        nn.init.zeros_(self.context_head.weight); nn.init.zeros_(self.context_head.bias)

    def forward(self,rgb,omni_prob):
        out=self.base(rgb,omni_prob)
        p=omni_prob.clamp(1e-4,1-1e-4)
        z=self.ctx_stem(torch.cat([rgb,p],1))
        z=self.mix(torch.cat([self.d1(z),self.d3(z),self.d7(z)],1))
        uncertainty=1.0-torch.abs(2.0*p-1.0)
        gate=torch.sigmoid(self.context_gate(z))*(0.35+0.65*uncertainty)
        corr=self.max_context*torch.tanh(self.context_head(z))*gate
        refined_logit=out['refined_logit']+corr
        out['refined_logit']=refined_logit
        out['refined_prob']=torch.sigmoid(refined_logit)
        out['context_correction']=corr
        out['context_gate']=gate
        return out
