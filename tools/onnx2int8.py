#!/usr/bin/env python3
# =====================================================================
#  onnx2int8.py  ―  Tiny-ViT INT8 packer (weights.bin + model.h only)
#  Generates:
#      • weights.bin   – INT8/INT32 tensors (weights, biases, LN γ/β)
#      • model.h       – dimensions, fixed-point shifts, arena layout,
#                        weight offsets and pre-computed Q0.15 scalers
#  Runtime becomes 100 % integer; no floating-point tables are loaded.
# ---------------------------------------------------------------------
#  Compatibility notes (engine.c):
#    • All GEMM weights are saved as K×N (row-major): input_dim × out_dim
#    • Per-channel quantization along columns (axis=1) → scales length N
#    • CLS and APE quantized in the PatchEmbed activation domain:
#         act_scale_pe = max(s_pe) / 2**PE_SHIFT
# ---------------------------------------------------------------------
#  Copyright (c) 2025  Alessandro Varaldi
#  MIT License
# =====================================================================

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional

import numpy as np
import onnx
import yaml

# ---------------------------------------------------------------------
#  Quantisation helpers
# ---------------------------------------------------------------------
def sym_int8_quant(arr: np.ndarray, *, per_channel: bool, axis: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetric INT8 quantisation.
    Returns (q_int8, scale) where `scale` is either a scalar or a
    float32 vector (per-channel).
    """
    if per_channel:
        arr_ch_first = np.moveaxis(arr, axis, 0)            # (C, …)
        flat         = arr_ch_first.reshape(arr_ch_first.shape[0], -1)
        scales       = np.abs(flat).max(axis=1)
        scales       = np.where(scales == 0, 1.0, scales) / 127.0
        q_flat       = np.clip(np.round(flat / scales[:, None]),
                               -127, 127).astype(np.int8)
        q            = np.moveaxis(q_flat.reshape(arr_ch_first.shape), 0, axis)
        return q, scales.astype(np.float32)

    scale = float(np.abs(arr).max() / 127.0 or 1.0)
    q     = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
    return q, np.float32(scale)


def quantize_bias(bias: np.ndarray, weight_scales, in_scale: float = 1.0
                  ) -> np.ndarray:
    """
    bias_int32 = round(bias_float / (in_scale * weight_scale))
    `weight_scales` may be scalar or per-channel (length N).
    """
    q = np.round(bias / (in_scale * weight_scales)).astype(np.int64)
    q = np.clip(q, -2**31, 2**31 - 1).astype(np.int32)
    return q


def write_array(fh, arr: np.ndarray) -> int:
    """
    Write `arr` raw bytes, return the starting offset.
    """
    off = fh.tell()
    fh.write(arr.tobytes())
    return off


def quantize_act_global(arr: np.ndarray, scale: float) -> np.ndarray:
    """
    Quantize ACTIVATIONS in INT8 with a single global `scale`:
      q = clamp(round(arr / scale), -127, 127).astype(int8)
    """
    q = np.round(arr.astype(np.float32) / float(scale)).astype(np.int32)
    q = np.clip(q, -127, 127).astype(np.int8)
    return q


# ---------------------------------------------------------------------
#  Fixed-point helpers
# ---------------------------------------------------------------------
def to_q15(in_scale: float, w_scale: float, _out_shift: int) -> int:
    """
    Return Q0.15 multiplier as signed int16.
    Note: OUT_SHIFT is *not* folded in here; the kernel shifts by
    (15 + OUT_SHIFT) after the multiply.
    """
    alpha = in_scale * w_scale
    m = int(round(alpha * 32768))
    return max(-32768, min(32767, m))


def theor_shift(k: int) -> int:
    """ceil(log2(k*127)) – shift that avoids INT8 saturation."""
    return int(math.ceil(math.log2(max(1, k) * 127.0)))


def auto_fill_shifts(cfg: dict) -> None:
    """
    Fill cfg['shifts'] with theoretical values if missing or
    --auto-shift is requested.
    """
    if cfg.get("shifts") and not cfg.get("_force_auto", False):
        return
    dm, dff = cfg["DMODEL"], cfg["DFF"]
    hd      = dm // cfg["HEADS"]
    pd      = cfg.get("PATCH_DIM", dm)
    cfg["shifts"] = {
        "PE_SHIFT":    theor_shift(pd),
        "QKV_SHIFT":   theor_shift(dm),
        "ATT_SHIFT":   max(0, theor_shift(hd) - 4),   # softmax shrinks range
        "MHA_O_SHIFT": theor_shift(dm),
        "FFN_SHIFT1":  theor_shift(dm),
        "FFN_SHIFT2":  theor_shift(dff),
        "HEAD_SHIFT":  theor_shift(dm),
    }


# ---------------------------------------------------------------------
#  ONNX helpers (robust name resolution)
# ---------------------------------------------------------------------
def get_init(inits: Dict[str, onnx.TensorProto],
             keys: Iterable[str],
             *,
             required: bool = True,
             debug_all: Optional[Iterable[str]] = None) -> np.ndarray:
    """
    Try multiple candidate names; return numpy array if found.
    If not found and required, raise a KeyError with helpful hints.
    """
    for k in keys:
        if k in inits:
            return onnx.numpy_helper.to_array(inits[k])
    if required:
        msg = f"Missing initializer. Tried keys: {list(keys)}"
        if debug_all is not None:
            # print a compact list of available keys
            avail = list(debug_all)
            msg += f"\nAvailable initializers ({len(avail)}): {avail}"
        raise KeyError(msg)
    return None


# ---------------------------------------------------------------------
#  Header generator
# ---------------------------------------------------------------------
def gen_header(cfg: dict,
               w_ofs: Dict[str, int],
               sc_q15: Dict[str, int],
               arena: dict) -> str:
    lines = [
        "/* model.h — AUTOGENERATED; do not edit */",
        "#ifndef MODEL_H_",
        "#define MODEL_H_",
        "",
        "#include <stdint.h>",
        "",
    ]

    # Model dimensions
    for key in (
        "DMODEL", "DFF", "HEADS", "TOKENS",
        "PATCHES", "PATCH_DIM", "LAYERS", "OUT_DIM"
    ):
        lines.append(f"#define MODEL_{key:<11} {cfg[key]}")
    lines.append(f"#define MODEL_EPS_SHIFT {cfg['EPS_SHIFT']}\n")

    # Fixed-point shifts
    for name, val in cfg["shifts"].items():
        lines.append(f"#define {name:<14} {val}")
    lines.append("")

    # Arena layout
    lines.append(f"#define ARENA_BYTES      {arena['bytes']}")
    for name, off in arena["offsets"].items():
        lines.append(f"#define ARENA_OFF_{name:<6} {off}")
    lines.append("")

    # Weight/bias offsets (sorted for stability)
    for name in sorted(w_ofs):
        lines.append(f"#define OFF_{name:<13} {w_ofs[name]}")
    lines.append("")

    # Pre-computed Q0.15 scalers
    for name in sorted(sc_q15):
        lines.append(f"#define SC_{name:<11} {sc_q15[name]}")
    lines.append("")

    lines += [
        "extern const uint8_t  weights_bin[];",
        "",
        "#endif /* MODEL_H_ */",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Pack Tiny-ViT ONNX → int8 blobs + header")
    ap.add_argument("--model", required=True, help="input .onnx file")
    ap.add_argument("--cfg",   required=True, help="YAML config")
    ap.add_argument("--out",   required=True, help="output directory")
    ap.add_argument("--auto-shift", action="store_true",
                    help="re-compute shifts ignoring YAML")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- YAML ----------------------------------------------------------------
    cfg = yaml.safe_load(Path(args.cfg).read_text())
    if args.auto_shift or not cfg.get("shifts"):
        cfg["_force_auto"] = args.auto_shift
        auto_fill_shifts(cfg)

    # --- ONNX ----------------------------------------------------------------
    model = onnx.load(args.model)
    inits = {t.name: t for t in model.graph.initializer}
    init_keys = list(inits.keys())  # for debug messages

    w_ofs: Dict[str, int] = {}
    sc_q15: Dict[str, int] = {}

    weights_path = out_dir / "weights.bin"
    with open(weights_path, "wb") as wfh:
        # ---------------------------------------------------------------------
        # Patch-embedding (expect TinyViT: embed.proj.{weight,bias})
        # Save as K×N = [PATCH_DIM, DMODEL]; per-col scales (axis=1).
        # ---------------------------------------------------------------------
        w_pe = get_init(
            inits,
            keys=("embed.proj.weight", "patch_embed.weight", "patch_embed.proj.weight", "embed.weight"),
            debug_all=init_keys
        )
        if w_pe.ndim == 4:                            # conv kernel O×I×kH×kW
            w_pe = w_pe.reshape(w_pe.shape[0], -1)    # [O, Ikk] = [DM, PD]
        w_pe = w_pe.T                                 # ONNX Linear is [out,in] → [in,out]
        q_pe, s_pe = sym_int8_quant(w_pe, per_channel=True, axis=1)
        w_ofs["W_PE"] = write_array(wfh, q_pe)
        sc_q15["PE"]  = to_q15(1.0, float(s_pe.max()), cfg["shifts"]["PE_SHIFT"])

        b_pe = get_init(
            inits,
            keys=("embed.proj.bias", "patch_embed.bias", "patch_embed.proj.bias", "embed.bias"),
            debug_all=init_keys
        )
        w_ofs["B_PE"] = write_array(wfh, quantize_bias(b_pe, s_pe))

        # --- Activation scale after PatchEmbed (for CLS & APE) ---------------
        pe_shift     = int(cfg["shifts"]["PE_SHIFT"])
        s_pe_max     = float(np.max(s_pe)) if np.ndim(s_pe) else float(s_pe)
        act_scale_pe = s_pe_max / float(2 ** pe_shift)

        # ---------------------------------------------------------------------
        # Transformer blocks
        #   All saved as K×N; per-col (axis=1) scales.
        # ---------------------------------------------------------------------
        L   = cfg["LAYERS"]
        DM  = cfg["DMODEL"]

        for l in range(L):
            # QKV fused: usually "layers.{l}.attn.qkv.weight"
            wqkv = get_init(
                inits,
                keys=(f"layers.{l}.attn.qkv.weight",
                      f"layers.{l}.attn.Wqkv.weight",
                      f"layers.{l}.attn.W_qkv.weight"),
                debug_all=init_keys
            ).T                                                          # [DM, 3*DM] = K×N
            q, s = sym_int8_quant(wqkv, per_channel=True, axis=1)        # N=3*DM
            w_ofs[f"W_QKV_L{l}"] = write_array(wfh, q)
            sc_q15[f"QKV_L{l}"]  = to_q15(1.0, float(s.max()), cfg["shifts"]["QKV_SHIFT"])

            b = get_init(
                inits,
                keys=(f"layers.{l}.attn.qkv.bias",
                      f"layers.{l}.attn.Wqkv.bias",
                      f"layers.{l}.attn.W_qkv.bias"),
                debug_all=init_keys
            )
            w_ofs[f"B_QKV_L{l}"] = write_array(wfh, quantize_bias(b, s))

            # Output projection: "layers.{l}.attn.proj.weight" or "out_proj.weight"
            wo = get_init(
                inits,
                keys=(f"layers.{l}.attn.proj.weight",
                      f"layers.{l}.attn.out_proj.weight"),
                debug_all=init_keys
            ).T                                                          # [DM, DM] = K×N
            q, s = sym_int8_quant(wo, per_channel=True, axis=1)
            w_ofs[f"W_O_L{l}"] = write_array(wfh, q)
            sc_q15[f"MHAO_L{l}"] = to_q15(1.0, float(s.max()), cfg["shifts"]["MHA_O_SHIFT"])

            b = get_init(
                inits,
                keys=(f"layers.{l}.attn.proj.bias",
                      f"layers.{l}.attn.out_proj.bias"),
                debug_all=init_keys
            )
            w_ofs[f"B_O_L{l}"] = write_array(wfh, quantize_bias(b, s))

            # FFN fc1: "layers.{l}.ffn.fc1.weight" or "mlp.fc1.weight"
            w1 = get_init(
                inits,
                keys=(f"layers.{l}.ffn.fc1.weight",
                      f"layers.{l}.mlp.fc1.weight"),
                debug_all=init_keys
            ).T                                                          # [DM, DFF] = K×N
            q, s = sym_int8_quant(w1, per_channel=True, axis=1)          # N=DFF
            w_ofs[f"W_FFN1_L{l}"] = write_array(wfh, q)
            sc_q15[f"FFN1_L{l}"]  = to_q15(1.0, float(s.max()), cfg["shifts"]["FFN_SHIFT1"])

            b = get_init(
                inits,
                keys=(f"layers.{l}.ffn.fc1.bias",
                      f"layers.{l}.mlp.fc1.bias"),
                debug_all=init_keys
            )
            w_ofs[f"B_FFN1_L{l}"] = write_array(wfh, quantize_bias(b, s))

            # FFN fc2: "layers.{l}.ffn.fc2.weight" or "mlp.fc2.weight"
            w2 = get_init(
                inits,
                keys=(f"layers.{l}.ffn.fc2.weight",
                      f"layers.{l}.mlp.fc2.weight"),
                debug_all=init_keys
            ).T                                                          # [DFF, DM] = K×N
            q, s = sym_int8_quant(w2, per_channel=True, axis=1)          # N=DM
            w_ofs[f"W_FFN2_L{l}"] = write_array(wfh, q)
            sc_q15[f"FFN2_L{l}"]  = to_q15(1.0, float(s.max()), cfg["shifts"]["FFN_SHIFT2"])

            b = get_init(
                inits,
                keys=(f"layers.{l}.ffn.fc2.bias",
                      f"layers.{l}.mlp.fc2.bias"),
                debug_all=init_keys
            )
            w_ofs[f"B_FFN2_L{l}"] = write_array(wfh, quantize_bias(b, s))

            # LayerNorm params: prefer weight/bias; fallback gamma/beta
            for ln_alias, tag in (("ln1", "LN1"), ("ln2", "LN2")):
                g = get_init(
                    inits,
                    keys=(f"layers.{l}.{ln_alias}.weight",
                          f"layers.{l}.{ln_alias}.gamma",
                          f"layers.{l}.norm{1 if ln_alias=='ln1' else 2}.weight"),
                    debug_all=init_keys
                ).astype(np.float32)
                b = get_init(
                    inits,
                    keys=(f"layers.{l}.{ln_alias}.bias",
                          f"layers.{l}.{ln_alias}.beta",
                          f"layers.{l}.norm{1 if ln_alias=='ln1' else 2}.bias"),
                    debug_all=init_keys
                ).astype(np.float32)
                to_i32 = lambda x: np.round(x * 32768).clip(-32768, 32767).astype(np.int32)
                w_ofs[f"G_{tag}_L{l}"] = write_array(wfh, to_i32(g))
                w_ofs[f"B_{tag}_L{l}"] = write_array(wfh, to_i32(b))

        # ---------------------------------------------------------------------
        # CLS token embedding — quantized in PatchEmbed activation domain
        #   Expected names: "cls" (TinyViT) or "cls_token"
        # ---------------------------------------------------------------------
        cls_f = get_init(
            inits,
            keys=("cls", "cls_token"),
            debug_all=init_keys
        ).astype(np.float32).reshape(-1)  # [DMODEL]
        q_cls = quantize_act_global(cls_f, act_scale_pe)
        w_ofs["CLS_EMB"] = write_array(wfh, q_cls)

        # ---------------------------------------------------------------------
        # Absolute Positional Embedding (APE) — optional
        #   Names tried: "pos_embed", "pos_embedding", "positional_embedding",
        #                "pos_embed.weight"
        #   Shape accepted: [1, TOKENS, DMODEL] or [TOKENS, DMODEL]
        # ---------------------------------------------------------------------
        pos = get_init(
            inits,
            keys=("pos_embed", "pos_embedding", "positional_embedding", "pos_embed.weight"),
            required=False,
            debug_all=init_keys
        )
        if pos is not None:
            pos = pos.astype(np.float32)
            if pos.ndim == 3 and pos.shape[0] == 1:
                pos = pos[0]
            pos = pos.reshape(cfg["TOKENS"], cfg["DMODEL"])
            q_pos = quantize_act_global(pos, act_scale_pe).astype(np.int8)
            w_ofs["POS_EMB"] = write_array(wfh, q_pos)

        # ---------------------------------------------------------------------
        # Classifier head  [OUT_DIM, DMODEL] (ONNX out,in) → transpose to K×N
        # ---------------------------------------------------------------------
        w_head = get_init(inits, keys=("head.weight",), debug_all=init_keys).T  # [DM, OUT]
        q, s   = sym_int8_quant(w_head, per_channel=True, axis=1)               # N=OUT
        w_ofs["HEAD_W"] = write_array(wfh, q)
        sc_q15["HEAD"]  = to_q15(1.0, float(s.max()), cfg["shifts"]["HEAD_SHIFT"])

        b_head = get_init(inits, keys=("head.bias",), debug_all=init_keys)
        w_ofs["HEAD_B"] = write_array(wfh, quantize_bias(b_head, s))

    # -------------------------------------------------------------------------
    # Runtime arena layout (token-major ping-pong)
    # -------------------------------------------------------------------------
    tok, dm, dff = cfg["TOKENS"], cfg["DMODEL"], cfg["DFF"]
    arena = {
        "bytes": tok * dm * 5 + tok * dff,   # BUF0 BUF1 Q K V + TMP
        "offsets": {
            "BUF0": 0,
            "BUF1": tok * dm,
            "Q":    tok * dm * 2,
            "K":    tok * dm * 3,
            "V":    tok * dm * 4,
            "TMP":  tok * dm * 5,
        },
    }

    # -------------------------------------------------------------------------
    # Write header
    # -------------------------------------------------------------------------
    (out_dir / "model.h").write_text(gen_header(cfg, w_ofs, sc_q15, arena))

    kb_w = weights_path.stat().st_size / 1024
    print(f"✅  Generated weights.bin ({kb_w:.1f} KB) and model.h")


if __name__ == "__main__":
    main()
