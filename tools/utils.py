#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, os, glob
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import amp
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T          # >=0.19

# --------------------------- Low I/O priority ------------------------
def low_prio_worker_init(_):
    """Lower CPU & I/O priority for data loader workers to reduce system impact."""
    try:
        import psutil
        p = psutil.Process(os.getpid())
        try:
            os.nice(10)
        except Exception:
            pass
        try:
            if hasattr(psutil, "IOPRIO_CLASS_IDLE"):
                p.ionice(psutil.IOPRIO_CLASS_IDLE)
            else:
                p.ionice(psutil.IOPRIO_CLASS_BE, 7)
        except Exception:
            pass
    except Exception:
        pass

# --------------------------- DropPath ------------------------
def drop_path(x, p: float = 0., training: bool = False):
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
    return x.div(keep) * mask.floor()

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        return drop_path(x, self.p, self.training)

# --------------------------- Model --------------------------
class LayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps
    def forward(self, x):
        m, v = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
        return self.g * (x - m) / torch.sqrt(v + self.eps) + self.b

class PatchEmbed(nn.Module):
    def __init__(self, img, patch, in_c=3, d=256):
        super().__init__()
        self.proj = nn.Conv2d(in_c, d, patch, patch)  # kernel=stride=patch
        self.np = (img // patch) ** 2
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)    # (B, N, d)
        return x

class MHSA(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d)
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))

class FFN(nn.Module):
    def __init__(self, d, dff, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(d, dff)
        self.fc2 = nn.Linear(dff, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)
    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, d, h, dff, dp=0., do=0.):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.ffn = LayerNorm(d), MHSA(d, h), LayerNorm(d), FFN(d, dff, do)
        self.dp = DropPath(dp) if dp > 0 else nn.Identity()
    def forward(self, x):
        x = x + self.dp(self.attn(self.ln1(x)))
        return x + self.dp(self.ffn(self.ln2(x)))

class TinyViT(nn.Module):
    def __init__(self, img, patch, d, depth, heads, dff, classes, dp=0., do=0.):
        super().__init__()
        self.embed = PatchEmbed(img, patch, 3, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        # APE: learnable absolute positional embedding per [CLS]+patch tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, self.embed.np + 1, d))
        dpr = torch.linspace(0, dp, depth).tolist()
        self.layers = nn.ModuleList([Block(d, heads, dff, dpr[i], do) for i in range(depth)])
        self.head = nn.Linear(d, classes)
        # init
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
    def forward(self, x):
        x = self.embed(x)  # (B, N, D)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1)  # (B, N+1, D)
        x = x + self.pos_embed
        for blk in self.layers:
            x = blk(x)
        return self.head(x[:, 0])

# ---------------------- MixUp / CutMix ----------------------
def mixup(x, y, alpha=0.2, p=0.5):
    if torch.rand(1, device=x.device) > p:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix(imgs, y, alpha=1.0):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    B, C, H, W = imgs.size()
    idx = torch.randperm(B, device=imgs.device)
    rx, ry = np.random.randint(W), np.random.randint(H)
    bw, bh = int(W * math.sqrt(1 - lam)), int(H * math.sqrt(1 - lam))
    x0, y0 = max(rx - bw // 2, 0), max(ry - bh // 2, 0)
    x1, y1 = min(x0 + bw, W), min(y0 + bh, H)
    imgs[:, :, y0:y1, x0:x1] = imgs[idx, :, y0:y1, x0:x1]
    lam = 1 - (x1 - x0) * (y1 - y0) / (W * H)
    return imgs, y, y[idx], lam

def do_mix(imgs, y, alpha=0.2, p=0.5, use_cutmix=False):
    if torch.rand(()) > p:
        return imgs, y, y, 1.0
    if use_cutmix and torch.rand(()) < 0.5:
        return cutmix(imgs, y, alpha=max(alpha, 1e-6))
    else:
        return mixup(imgs, y, alpha=max(alpha, 1e-6), p=1.0)

def lin_decay(a0, a1, t, T):
    t = min(max(t, 0), T)
    return a0 + (a1 - a0) * (t / max(1, T))

# --------------- Cosine scheduler w/ warm-up ---------------
def cosine_warmup(opt, warm, total, min_lr=1e-5):
    def fn(step):
        if step < warm:
            return step / max(1, warm)
        prog = (step - warm) / max(1, (total - warm))
        return max(min_lr / opt.defaults['lr'], 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)

# -------------------- DataLoaders helpers ------------------
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CIFAR_MEAN, CIFAR_STD = (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)

def imagenet_class_mapping_from_root(root):
    ds = datasets.ImageFolder(os.path.join(root, "train"))
    return ds.class_to_idx  # dict[str,int]

def imagenet_loaders(root, img, bs, workers):
    train_t = T.Compose([
        T.RandomResizedCrop(img, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandAugment(),
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    val_t = T.Compose([
        T.Resize(int(img * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img),
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    tr = datasets.ImageFolder(os.path.join(root, "train"), transform=train_t)
    te = datasets.ImageFolder(os.path.join(root, "val"), transform=val_t)
    loader_kwargs = dict(
        num_workers=min(4, workers),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=low_prio_worker_init,
    )
    tr_ld = DataLoader(tr, bs, shuffle=True, **loader_kwargs, drop_last=True)
    te_ld = DataLoader(te, bs * 2, shuffle=False, **loader_kwargs)
    return tr_ld, te_ld

def cifar_loaders(root, bs, workers, img=32):
    """CIFAR-100 loaders; se img!=32 applica una pipeline tipo-ImageNet a risoluzioni maggiori."""
    if img == 32:
        train_t = T.Compose([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.RandAugment(),
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
            T.RandomErasing(p=0.25),
        ])
        val_t = T.Compose([
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_t = T.Compose([
            T.Resize(int(img * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(img, padding=int(0.125 * img), padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.RandAugment(),
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
            T.RandomErasing(p=0.25),
        ])
        val_t = T.Compose([
            T.Resize(int(img * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img),
            T.PILToTensor(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    tr = datasets.CIFAR100(root, True, download=True, transform=train_t)
    te = datasets.CIFAR100(root, False, download=True, transform=val_t)
    loader_kwargs = dict(
        num_workers=min(4, workers),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=low_prio_worker_init,
    )
    tr_ld = DataLoader(tr, bs, shuffle=True, **loader_kwargs)
    te_ld = DataLoader(te, bs * 2, shuffle=False, **loader_kwargs)
    return tr_ld, te_ld

# --------------- WebDataset (ImageNet shards) --------------
def glob_shards(dirpath, prefix):
    paths = sorted(glob.glob(os.path.join(dirpath, f"{prefix}-*.tar")))
    if not paths:
        raise FileNotFoundError(f"No shards found at: {dirpath} with pattern {prefix}-*.tar")
    return paths

def imagenet_webdataset_loaders(imagenet_root, shards_root, img, bs, workers):
    import webdataset as wds
    import torchvision.io as tvio
    from torchvision.io.image import ImageReadMode

    cls2idx = imagenet_class_mapping_from_root(imagenet_root)
    train_shards = glob_shards(os.path.join(shards_root, "train"), "train")
    val_shards   = glob_shards(os.path.join(shards_root, "val"),   "val")
    shardshuffle_train = min(256, len(train_shards))

    def decode_jpeg_bytes(img_bytes: bytes) -> torch.Tensor:
        buf = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8)  # writable
        try:
            img = tvio.decode_jpeg(buf, mode=ImageReadMode.RGB)
        except RuntimeError:
            img = tvio.decode_image(buf, mode=ImageReadMode.RGB)
        if img.ndim == 3 and img.shape[0] != 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1).contiguous()
        return img

    train_t = T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(img, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandAugment(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    val_t = T.Compose([
        T.ToImage(),
        T.Resize(int(img * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])

    def map_train(sample):
        img_bytes, cls_bytes = sample
        img = decode_jpeg_bytes(img_bytes)
        y = cls2idx[cls_bytes.decode().strip()]
        return train_t(img), y

    def map_val(sample):
        img_bytes, cls_bytes = sample
        img = decode_jpeg_bytes(img_bytes)
        y = cls2idx[cls_bytes.decode().strip()]
        return val_t(img), y

    handler = wds.handlers.warn_and_continue
    tr = (wds.WebDataset(train_shards, handler=handler, shardshuffle=shardshuffle_train)
            .shuffle(2000).to_tuple("jpg", "cls").map(map_train))
    te = (wds.WebDataset(val_shards,   handler=handler, shardshuffle=False)
            .to_tuple("jpg", "cls").map(map_val))

    loader_kwargs = dict(
        batch_size=bs,
        num_workers=min(4, workers),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=low_prio_worker_init,
    )
    tr_ld = wds.WebLoader(tr, shuffle=False, **loader_kwargs)
    loader_kwargs['batch_size'] = bs * 2
    te_ld = wds.WebLoader(te, shuffle=False, **loader_kwargs)
    return tr_ld, te_ld

# --------------- Load pretrained (ignore head) --------------
def load_pretrained(model, path, resize_patch=True):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # drop head if num_classes changed
    head_w = model.head.weight.shape[0]
    sd = {k: v for k, v in sd.items() if not (k.startswith("head.") and v.shape[0] != head_w)}
    # handle patch embed mismatch
    k_w = "embed.proj.weight"
    if resize_patch and k_w in sd:
        w_src = sd[k_w]                      # [D,3,Kh,Kw]
        w_tgt = model.embed.proj.weight      # [D,3,kh,kw]
        if w_src.shape[:2] == w_tgt.shape[:2] and w_src.shape[2:] != w_tgt.shape[2:]:
            kh, kw = w_tgt.shape[2], w_tgt.shape[3]
            with torch.no_grad():
                w_new = F.interpolate(w_src, size=(kh, kw), mode="bicubic", align_corners=False)
            sd[k_w] = w_new
            print(f"[load_pretrained] Resized patch embed {tuple(w_src.shape[2:])} -> {(kh, kw)}")
    if k_w in sd and sd[k_w].shape != model.embed.proj.weight.shape:
        print(f"[load_pretrained] Dropping {k_w} due to shape mismatch {tuple(sd[k_w].shape)} -> {tuple(model.embed.proj.weight.shape)}")
        sd.pop(k_w, None); sd.pop("embed.proj.bias", None)
    # handle pos_embed mismatch
    k_pe = "pos_embed"
    if k_pe in sd:
        pe_src = sd[k_pe]                    # [1, 1+Nsrc, D]
        pe_tgt = model.pos_embed             # [1, 1+Ntgt, D]
        if pe_src.shape != pe_tgt.shape:
            try:
                cls_src, grid_src = pe_src[:, :1], pe_src[:, 1:]      # [1,1,D], [1,Nsrc,D]
                Nt_src = grid_src.shape[1]
                Nt_tgt = model.embed.np
                gs_src = int(math.sqrt(Nt_src))
                gs_tgt = int(math.sqrt(Nt_tgt))
                if gs_src * gs_src == Nt_src and gs_tgt * gs_tgt == Nt_tgt:
                    grid_src = grid_src.reshape(1, gs_src, gs_src, -1).permute(0, 3, 1, 2)     # [1,D,H,W]
                    grid_new = F.interpolate(grid_src, size=(gs_tgt, gs_tgt),
                                             mode="bicubic", align_corners=False)              # [1,D,H',W']
                    grid_new = grid_new.permute(0, 2, 3, 1).reshape(1, Nt_tgt, -1)             # [1,Ntgt,D]
                    sd[k_pe] = torch.cat([cls_src, grid_new], dim=1)
                    print(f"[load_pretrained] Resized pos_embed {gs_src}x{gs_src} -> {gs_tgt}x{gs_tgt}")
                else:
                    print("[load_pretrained] Dropping pos_embed (non-square grid)")
                    sd.pop(k_pe, None)
            except Exception as e:
                print(f"[load_pretrained] Could not resize pos_embed: {e}")
                sd.pop(k_pe, None)
    msg = model.load_state_dict(sd, strict=False)
    print(f"Loaded {path}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")

# ------------------------- Eval helper ----------------------
@torch.no_grad()
def eval_on_loader(model_eval: nn.Module, loader, device, max_batches=None, tta=False):
    """Top-1 accuracy su loader (opzionalmente con TTA flip orizzontale)."""
    model_eval.eval()
    total, correct, n_batches = 0, 0, 0
    with amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        for imgs, lbl in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbl  = torch.as_tensor(lbl, device=device, dtype=torch.long)
            if tta:
                out1 = model_eval(imgs)
                out2 = model_eval(torch.flip(imgs, dims=[3]))
                out  = (out1 + out2) * 0.5
            else:
                out = model_eval(imgs)
            correct += (out.argmax(1) == lbl).sum().item()
            total   += imgs.size(0)
            n_batches += 1
            if (max_batches is not None) and (n_batches >= max_batches):
                break
    return 100.0 * correct / max(1, total)
