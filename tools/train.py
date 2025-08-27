#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, time, math
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import amp
import yaml
from torch.optim.swa_utils import (AveragedModel, get_ema_multi_avg_fn, update_bn)

from train_utils import (
    TinyViT, cosine_warmup,
    imagenet_loaders, imagenet_webdataset_loaders, cifar_loaders,
    load_pretrained, eval_on_loader, do_mix, lin_decay
)

torch.backends.cudnn.benchmark = True

# ------------------------- Train ---------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Seeds
    import random
    random.seed(0); np.random.seed(0)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    # --------------------------- Data ---------------------------
    if args.stage == "pretrain":
        if args.webdataset:
            ld_train, ld_val = imagenet_webdataset_loaders(
                args.imagenet, args.webdataset, args.img_size, args.bs, args.workers)
        else:
            ld_train, ld_val = imagenet_loaders(
                args.imagenet, args.img_size, args.bs, args.workers)
        num_classes = 1000
    else:
        # finetune
        if not args.ft_fullres:
            args.img_size, args.patch = 32, 4
        # CIFAR-100 con img opzionale (32 o 224, etc.)
        ld_train, ld_val = cifar_loaders(args.data, args.bs, args.workers, img=args.img_size)
        num_classes = 100

    # --------------------------- Model --------------------------
    net = TinyViT(args.img_size, args.patch, args.dim, args.layers, args.heads,
                  args.dff, num_classes, args.drop_path, args.dropout).to(device)

    if args.resume:
        load_pretrained(net, args.resume)

    # EMA sui pesi del backbone (net)
    ema = AveragedModel(net, device=device, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    model = torch.compile(net, mode="max-autotune") if args.compile else net

    # --------------------------- Optimizer ----------------------
    if args.stage == "finetune" and (args.ft_llrd != 1.0 or args.ft_head_lr_mul != 1.0):
        groups = []
        depth = len(net.layers)
        # embed
        groups.append({"params": list(net.embed.parameters()),
                       "lr": args.lr * (args.ft_llrd ** depth)})
        # blocchi (dal basso lr basso → alto)
        for i, blk in enumerate(net.layers):
            lr_i = args.lr * (args.ft_llrd ** (depth - 1 - i))
            groups.append({"params": list(blk.parameters()), "lr": lr_i})
        # pos_embed e cls
        groups.append({"params": [net.pos_embed], "lr": args.lr})
        groups.append({"params": [net.cls],       "lr": args.lr})
        # head con moltiplicatore
        groups.append({"params": list(net.head.parameters()),
                       "lr": args.lr * args.ft_head_lr_mul})
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scaler = amp.GradScaler()
    total  = len(ld_train) * args.epochs if hasattr(ld_train, "__len__") else args.epochs * 1000
    warm   = args.warmup * (len(ld_train) if hasattr(ld_train, "__len__") else 1000)
    sched  = cosine_warmup(opt, warm, total, args.min_lr)
    crit   = nn.CrossEntropyLoss(label_smoothing=args.ls)

    # --------------------------- SWA (solo FT) ------------------
    swa_model, swa_start = None, None
    if args.stage == "finetune" and args.ft_swa:
        swa_model = AveragedModel(net, device=device, multi_avg_fn=get_ema_multi_avg_fn(1.0))
        swa_start = max(1, args.epochs - args.ft_swa_epochs + 1)

    best = 0.0

    # --------------------- Print model summary ------------------
    PARAMS = [p for p in model.parameters()]
    tot_elems = sum(p.numel() for p in PARAMS)
    print(f"Parameters: {tot_elems:,}")
    def to_mib(bytes_): return bytes_ / 1024**2
    fp32_bytes = sum(p.numel() * 4 for p in PARAMS)
    print(f"Memory fp32 (ass.): {to_mib(fp32_bytes):.2f} MiB")
    dtype_bytes = sum(p.numel() * p.element_size() for p in PARAMS)
    print(f"Memory actual dtype: {to_mib(dtype_bytes):.2f} MiB")
    int8_bytes = sum(p.numel() * 1 for p in PARAMS)
    print(f"Memory int8 (quant): {to_mib(int8_bytes):.2f} MiB")
    print("Model compiled:", args.compile)
    print("Using mixed precision:", torch.cuda.is_available())

    Path("checkpoints").mkdir(exist_ok=True)

    # ------------------------- Freeze backbone ------------------
    def set_backbone_trainable(flag: bool):
        for n, p in net.named_parameters():
            if n.startswith("head."):
                continue
            p.requires_grad = flag

    try:
        step = 0
        for epoch in range(1, args.epochs + 1):
            if args.stage == "finetune" and args.ft_freeze_backbone_epochs > 0:
                if epoch == 1:
                    set_backbone_trainable(False)
                if epoch == args.ft_freeze_backbone_epochs + 1:
                    set_backbone_trainable(True)

            model.train()
            t0 = time.time()
            loss_sum = 0.0
            tot_samples = 0.0
            train_correct_soft = 0.0

            # scheduling MixUp/CutMix
            if args.stage == "finetune" and args.ft_mix_decay:
                alpha_ep = lin_decay(args.mixup_alpha, 0.0, epoch - 1, args.epochs - 1)
                prob_ep  = lin_decay(args.mix_prob,     0.0, epoch - 1, args.epochs - 1)
            else:
                alpha_ep, prob_ep = args.mixup_alpha, args.mix_prob

            for imgs, lbls in ld_train:
                imgs = imgs.to(device, non_blocking=True)
                lbls = torch.as_tensor(lbls, device=device, dtype=torch.long)
                imgs, y1, y2, lam = do_mix(
                    imgs, lbls, alpha=alpha_ep, p=prob_ep,
                    use_cutmix=(args.stage == "finetune" and args.ft_use_cutmix)
                )

                opt.zero_grad(set_to_none=True)
                with amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    logits = model(imgs)
                    ce = lam * crit(logits, y1) + (1 - lam) * crit(logits, y2)
                    loss = ce

                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()

                ema.update_parameters(net)

                if swa_model is not None and epoch >= swa_start:
                    swa_model.update_parameters(net)

                sched.step()
                step += 1

                B = imgs.size(0)
                loss_sum += loss.item() * B
                tot_samples += B
                pred = logits.argmax(1)
                correct_soft = (lam * (pred == y1).float() + (1 - lam) * (pred == y2).float()).sum().item()
                train_correct_soft += correct_soft

            train_loss = loss_sum / max(1.0, tot_samples)
            train_acc_soft = 100.0 * train_correct_soft / max(1.0, tot_samples)

            train_acc_nomix = None
            if args.train_eval_batches > 0:
                train_acc_nomix = eval_on_loader(ema, ld_train, device, max_batches=args.train_eval_batches, tta=False)

            val_tta = (args.stage == "finetune" and args.ft_tta)
            val_acc = eval_on_loader(ema, ld_val, device, max_batches=None, tta=val_tta)

            log = (f"[E{epoch:03d}] loss {train_loss:.4f} "
                   f"tr_soft {train_acc_soft:5.2f}% "
                   f"val {val_acc:5.2f}% lr {sched.get_last_lr()[0]:.2e} "
                   f"time {time.time()-t0:.1f}s")
            if train_acc_nomix is not None:
                log = log.replace(" val ", f" tr_nomix(EMA) {train_acc_nomix:5.2f}% val ")
            if val_tta:
                log += " (TTA)"
            print(log)

            torch.save({
                "model": net.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch, "step": step, "args": vars(args),
            }, "checkpoints/last.pth")
            print("Saved checkpoint ➜ checkpoints/last.pth")

            if val_acc > best:
                best = val_acc
                torch.save(ema.module.state_dict(), f"checkpoints/best.pth")
                print("Best model saved ➜ checkpoints/best.pth")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print(f"Best val acc (EMA): {best:.2f}%")

    # --------------------------- SWA eval/save ------------------
    if swa_model is not None:
        update_bn(ld_train, swa_model, device=device)
        swa_val = eval_on_loader(swa_model, ld_val, device, max_batches=None, tta=val_tta)
        print(f"SWA val acc: {swa_val:.2f}%")
        torch.save(swa_model.module.state_dict(), "checkpoints/best_swa.pth")
        print("Saved SWA model ➜ checkpoints/best_swa.pth")

    # ----------- Export ONNX + YAML (robust, no ONNX parsing) -----------
    def to_pyint(x):
        if isinstance(x, torch.Tensor):
            return int(x.detach().cpu().item())
        if isinstance(x, np.generic):
            return int(x.item())
        return int(x)

    export_dir = Path("models")
    export_dir.mkdir(parents=True, exist_ok=True)

    base = ema.module if hasattr(ema, "module") else ema
    base.eval()

    onnx_path = str(export_dir / ("vit_model_pretrain.onnx" if args.stage == "pretrain" else "vit_model_finetune.onnx"))
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    torch.onnx.export(
        base, dummy, onnx_path,
        input_names=["images"], output_names=["logits"],
        opset_version=17, do_constant_folding=False,
    )
    print(f"ONNX exported ➜ {onnx_path}")

    d_model   = to_pyint(base.head.in_features)
    d_ff      = to_pyint(base.layers[0].ffn.fc1.weight.shape[0])
    layers    = to_pyint(len(base.layers))
    out_dim   = to_pyint(base.head.weight.shape[0])
    tokens    = to_pyint(base.embed.np + 1)
    patches   = to_pyint(base.embed.np)
    patch_dim = to_pyint(base.embed.proj.weight.shape[0])

    cfg = {
        "DMODEL":    d_model,
        "DFF":       d_ff,
        "HEADS":     int(args.heads),
        "TOKENS":    tokens,
        "LAYERS":    layers,
        "OUT_DIM":   out_dim,
        "EPS_SHIFT": 12,
        "shifts":    {},
        "PATCHES":   patches,
        "PATCH_DIM": patch_dim,
        "POSENC":    "APE",
    }
    print("[YAML sanity]", cfg)

    yaml_path = export_dir / ("vit_config_pretrain.yaml" if args.stage == "pretrain" else "vit_config_finetune.yaml")
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"YAML exported ➜ {yaml_path}")

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("Tiny ViT train (ImageNet➜CIFAR) with finetune-only upgrades")
    p.add_argument("--stage", choices=["pretrain", "finetune"], default="finetune")
    p.add_argument("--data", default="./data", help="root CIFAR-100")
    p.add_argument("--imagenet", default="./imagenet", help="root ImageNet train/ val/ (used for class mapping)")
    p.add_argument("--webdataset", default=None, help="root of WebDataset shards (contains train/*.tar and val/*.tar)")
    p.add_argument("--img_size", type=int, default=224, help="input res (pretrain or finetune if --ft_fullres)")
    p.add_argument("--patch", type=int, default=16, help="patch size (pretrain or finetune if --ft_fullres)")

    # train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=5e-4)

    # model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dff", type=int, default=512)

    # reg
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ls", type=float, default=0.1)

    # mixup
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--mix_prob", type=float, default=0.5)

    # ema
    p.add_argument("--ema", type=float, default=0.9999)

    # misc
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", help="checkpoint .pth to resume / fine-tune")
    p.add_argument("--train_eval_batches", type=int, default=0, help="If >0, evaluates the EMA on N train batches without MixUp.")
    p.add_argument("--grad_clip", type=float, default=1.0)

    # ---------------- finetune-only upgrades ----------------
    p.add_argument("--ft_fullres", action="store_true", help="In finetune, use the same resolution/patch size as pretrain (do not force 32/4)")
    p.add_argument("--ft_llrd", type=float, default=0.8, help="Layer-wise learning rate decay (1.0=off)")
    p.add_argument("--ft_head_lr_mul", type=float, default=5.0, help="Learning rate multiplier for the head during finetune")
    p.add_argument("--ft_freeze_backbone_epochs", type=int, default=0, help="Freeze the backbone for N epochs at the beginning of finetune")
    p.add_argument("--ft_use_cutmix", action="store_true", help="Also alternate CutMix (in addition to MixUp) during finetune")
    p.add_argument("--ft_mix_decay", action="store_true", help="Linear decay of MixUp/CutMix prob/alpha towards 0 at the end of finetune")
    p.add_argument("--ft_swa", action="store_true", help="Use SWA in the last epochs of finetune")
    p.add_argument("--ft_swa_epochs", type=int, default=20, help="Number of final epochs to average in SWA")
    p.add_argument("--ft_tta", action="store_true", help="Use HFlip TTA during validation in finetune")

    args = p.parse_args()

    if args.stage == "finetune" and not args.ft_fullres:
        args.img_size, args.patch = 32, 4

    Path(args.data).mkdir(parents=True, exist_ok=True)
    train(args)

    # PRETRAIN:
    # python3 tools/train.py --stage pretrain --imagenet /data/dataset/pytorch_only/imagenet/ --webdataset /data/dataset/pytorch_only/imagenet-shards/ --bs 512 --epochs 200 --train_eval_batches 20 --workers 4 --compile
    #
    # FINETUNE (fullres + LLRD + mix-decay + SWA + TTA):
    # python3 tools/train.py --stage finetune --data ./cifar-100-python --resume checkpoints/pretrain250827.pth --bs 512 --epochs 100 --ema 0.99 --compile --ft_fullres --ft_llrd 0.8 --ft_head_lr_mul 5 --ft_use_cutmix --ft_mix_decay --ft_swa --ft_swa_epochs 20 --ft_tta
