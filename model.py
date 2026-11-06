"""Hybrid CNN-Transformer for skin lesion classification (HAM10000, 7 classes).

Pipeline:  image -> EfficientNet-B0 (local texture features, 7x7 map)
           -> 1x1 projection -> tokens (+CLS, +positional embedding)
           -> N Transformer encoder blocks (global context, attention exposed)
           -> CLS token -> classifier
"""
import torch
import torch.nn as nn
from torchvision import models, transforms

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def eval_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)              # each: B,H,N,d
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)  # B,H,N,N
        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out)), attn       # attn returned for XAI


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        a, attn = self.attn(self.norm1(x))
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x, attn


class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes=7, img_size=224, dim=256, depth=4, heads=8,
                 drop=0.1, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights).features  # stride 32, 1280 ch
        self.grid = img_size // 32
        self.proj = nn.Conv2d(1280, dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid ** 2 + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, 4.0, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, num_classes))

    def encode(self, fmap):
        """CNN feature map -> (logits, list of attention maps [B,H,N,N])."""
        x = self.proj(fmap).flatten(2).transpose(1, 2)            # B,49,dim
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = x + self.pos_embed
        attns = []
        for blk in self.blocks:
            x, a = blk(x)
            attns.append(a)
        return self.head(self.norm(x)[:, 0]), attns

    def forward(self, x, return_attn=False):
        logits, attns = self.encode(self.backbone(x))
        return (logits, attns) if return_attn else logits


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = ck["config"]
    model = HybridCNNTransformer(num_classes=len(ck["classes"]), img_size=cfg["img_size"],
                                 dim=cfg["dim"], depth=cfg["depth"], heads=cfg["heads"],
                                 pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["classes"], cfg
