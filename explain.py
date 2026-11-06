"""Explainability: Grad-CAM (CNN branch) and Attention Rollout (Transformer branch)."""
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm


def _norm01(x):
    x = x - x.min()
    return x / (x.max() + 1e-8)


def grad_cam(model, x, class_idx=None):
    """x: [1,3,H,W]. Returns (cam[H,W] in 0..1, probs[C], class_idx)."""
    model.eval()
    fmap = model.backbone(x)
    fmap.retain_grad()
    logits, _ = model.encode(fmap)
    probs = logits.softmax(-1)[0].detach().cpu().numpy()
    if class_idx is None:
        class_idx = int(logits.argmax(-1))
    model.zero_grad()
    logits[0, class_idx].backward()
    w = fmap.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return _norm01(cam.detach().cpu().numpy()), probs, class_idx


@torch.no_grad()
def attention_rollout(model, x):
    """Abu-Hazem / Abnar & Zuidema rollout over all blocks, CLS -> patch tokens."""
    model.eval()
    logits, attns = model(x, return_attn=True)
    rollout = None
    for a in attns:
        a = a.mean(1)                                    # avg heads: B,N,N
        a = a + torch.eye(a.size(-1), device=a.device)   # residual connection
        a = a / a.sum(-1, keepdim=True)
        rollout = a if rollout is None else a @ rollout
    mask = rollout[0, 0, 1:].reshape(1, 1, model.grid, model.grid)
    mask = F.interpolate(mask, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    return _norm01(mask.cpu().numpy()), logits.softmax(-1)[0].cpu().numpy()


def overlay(img_01, heat, alpha=0.45):
    """img_01: HxWx3 float in 0..1; heat: HxW in 0..1."""
    colored = cm.jet(heat)[..., :3]
    return np.clip((1 - alpha) * img_01 + alpha * colored, 0, 1)
