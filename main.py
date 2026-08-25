#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from fault_dataset import EarthquakeFaultDataset, collate_fn
from model import FaultTrans, FaultLossClean          # ← FaultCleanRule → FaultTrans


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_iou(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    pred_bin = (pred_prob > threshold).float()
    inter = (pred_bin * target).sum(dim=(-2, -1))
    union = ((pred_bin + target) > 0).float().sum(dim=(-2, -1))
    iou = inter / (union + 1e-6)

    has_label = target.sum(dim=(-2, -1)) > 0
    keep = valid_mask.bool() & has_label
    if keep.sum() == 0:
        return float("nan")
    return iou[keep].mean().item()


def compute_metrics(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    计算 IoU / F1 / Precision / Recall，返回字典。
    仅统计 valid_mask=True 且 target 非空的帧。
    """
    pred_bin = (pred_prob > threshold).float()
    tp = (pred_bin * target).sum(dim=(-2, -1))
    fp = (pred_bin * (1.0 - target)).sum(dim=(-2, -1))
    fn = ((1.0 - pred_bin) * target).sum(dim=(-2, -1))
    union = ((pred_bin + target) > 0).float().sum(dim=(-2, -1))

    iou       = tp / (union + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2.0 * precision * recall / (precision + recall + 1e-6)

    has_label = target.sum(dim=(-2, -1)) > 0
    keep = valid_mask.bool() & has_label
    if keep.sum() == 0:
        nan = float("nan")
        return {"iou": nan, "f1": nan, "precision": nan, "recall": nan}

    return {
        "iou":       iou[keep].mean().item(),
        "f1":        f1[keep].mean().item(),
        "precision": precision[keep].mean().item(),
        "recall":    recall[keep].mean().item(),
    }


def train_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_iou  = 0.0
    n_loss = 0
    n_iou  = 0

    for batch in loader:
        inputs      = batch["inputs"].to(device)
        target      = batch["rup_grid"].to(device)
        times       = batch["times"].to(device)
        valid_frame = batch["valid_frame"].to(device)
        seq_mask    = batch["seq_mask"].to(device)
        epicenter   = batch["epicenter"].to(device)
        strike      = batch["strike"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                belief_logits, committeds = model(inputs, times, seq_mask, epicenter)
                loss = criterion(
                    belief_logits=belief_logits,
                    committeds=committeds,
                    target=target,
                    valid_frame=valid_frame,
                    seq_mask=seq_mask,
                    strike=strike,
                )
            if not torch.isfinite(loss):
                log.warning("NaN/Inf loss detected, skipping batch")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                log.warning("NaN/Inf grad norm detected, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            belief_logits, committeds = model(inputs, times, seq_mask, epicenter)
            loss = criterion(
                belief_logits=belief_logits,
                committeds=committeds,
                target=target,
                valid_frame=valid_frame,
                seq_mask=seq_mask,
                strike=strike,
            )
            if not torch.isfinite(loss):
                log.warning("NaN/Inf loss detected, skipping batch")
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                log.warning("NaN/Inf grad norm detected, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()

        valid_mask       = (valid_frame > 0.5) & seq_mask
        committed_target = torch.cummax(target, dim=1)[0]
        iou = compute_iou(committeds.detach(), committed_target, valid_mask, threshold=0.5)

        total_loss += loss.item()
        n_loss += 1
        if np.isfinite(iou):
            total_iou += iou
            n_iou += 1

    return total_loss / max(n_loss, 1), total_iou / max(n_iou, 1)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    n_loss = 0
    n_iou  = 0

    for batch in loader:
        inputs      = batch["inputs"].to(device)
        target      = batch["rup_grid"].to(device)
        times       = batch["times"].to(device)
        valid_frame = batch["valid_frame"].to(device)
        seq_mask    = batch["seq_mask"].to(device)
        epicenter   = batch["epicenter"].to(device)
        strike      = batch["strike"].to(device)

        belief_logits, committeds = model(inputs, times, seq_mask, epicenter)
        loss = criterion(
            belief_logits=belief_logits,
            committeds=committeds,
            target=target,
            valid_frame=valid_frame,
            seq_mask=seq_mask,
            strike=strike,
        )

        valid_mask       = (valid_frame > 0.5) & seq_mask
        committed_target = torch.cummax(target, dim=1)[0]
        iou = compute_iou(committeds.detach(), committed_target, valid_mask, threshold=0.5)

        total_loss += loss.item()
        n_loss += 1
        if np.isfinite(iou):
            total_iou += iou
            n_iou += 1

    return total_loss / max(n_loss, 1), total_iou / max(n_iou, 1)


@torch.no_grad()
def test_epoch(model, loader, device, threshold: float = 0.5):
    """
    在测试集上计算完整指标（IoU / F1 / Precision / Recall）。
    使用累积GT（committed target）与 committed 输出进行比较，
    与训练/验证阶段保持一致。
    结果以均值 ± 标准差形式返回，便于直接写入论文。
    """
    model.eval()

    all_iou       = []
    all_f1        = []
    all_precision = []
    all_recall    = []

    for batch in loader:
        inputs      = batch["inputs"].to(device)
        target      = batch["rup_grid"].to(device)
        times       = batch["times"].to(device)
        valid_frame = batch["valid_frame"].to(device)
        seq_mask    = batch["seq_mask"].to(device)
        epicenter   = batch["epicenter"].to(device)

        _, committeds = model(inputs, times, seq_mask, epicenter)

        valid_mask       = (valid_frame > 0.5) & seq_mask
        committed_target = torch.cummax(target, dim=1)[0]

        m = compute_metrics(committeds, committed_target, valid_mask, threshold)

        for key, lst in [("iou", all_iou), ("f1", all_f1),
                         ("precision", all_precision), ("recall", all_recall)]:
            if np.isfinite(m[key]):
                lst.append(m[key])

    def summary(lst):
        if len(lst) == 0:
            return float("nan"), float("nan")
        arr = np.array(lst)
        return float(arr.mean()), float(arr.std())

    return {
        "iou":       summary(all_iou),
        "f1":        summary(all_f1),
        "precision": summary(all_precision),
        "recall":    summary(all_recall),
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | Channel: {args.channel}")

    train_ds, val_ds, test_ds = EarthquakeFaultDataset.train_val_test_split(
        args.h5,
        ratios=(0.7, 0.15, 0.15),
        seed=args.seed,
        normalize=True,
        only_valid=args.only_valid,
        channel=args.channel,
        sparse_sim=args.sparse_sim,
        n_stations=args.n_stations,
    )
    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_fn, pin_memory=pin_memory,
    )

    model = FaultTrans(                               # ← FaultCleanRule → FaultTrans
        in_channels=1,
        patch_size=args.patch_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        grid_size=args.grid_size,
        decoder_dim=args.decoder_dim,
        decoder_mid_dim=args.decoder_mid_dim,
        decoder_refine_blocks=args.decoder_refine_blocks,
        causal=args.causal,
        commit_tau=args.commit_tau,
        commit_temp=args.commit_temp,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params / 1e6:.2f} M")
    log.info(
        f"Rule settings: commit_tau={args.commit_tau}, commit_temp={args.commit_temp}, "
        f"decoder_mid_dim={args.decoder_mid_dim}, refine_blocks={args.decoder_refine_blocks}"
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    criterion = FaultLossClean(
        belief_w=args.belief_w,
        committed_w=args.committed_w,
        tv_w=args.tv_w,
        dir_w=args.dir_w,
        dice_w=args.dice_w,
        bce_w=args.bce_w,
        pos_weight=args.pos_weight,     # ← 新增
        min_ecc=args.min_ecc,           # ← 新增
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            init_scale=1024.0,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )
        if device.type == "cuda"
        else None
    )

    out_dir = Path(args.out) / args.channel
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_iou     = -1.0
    best_val_loss    = float("inf")
    patience_counter = 0
    best_ckpt_path   = out_dir / "best_iou_rule_hybrid.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_iou = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl_loss, vl_iou = val_epoch(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        log.info(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"tr_loss={tr_loss:.4f} tr_iou={tr_iou:.4f} "
            f"vl_loss={vl_loss:.4f} vl_iou={vl_iou:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} t={elapsed:.1f}s"
        )

        if vl_iou > best_val_iou:
            best_val_iou     = vl_iou
            patience_counter = 0
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_iou":   vl_iou,
                    "val_loss":  vl_loss,
                    "args":      vars(args),
                },
                best_ckpt_path,
            )
            log.info(f"  -> best IoU saved ({best_val_iou:.4f})")
        else:
            patience_counter += 1

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), out_dir / "best_loss_rule_hybrid.pth")

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                out_dir / f"ckpt_rule_hybrid_ep{epoch:03d}.pth",
            )

        if patience_counter >= args.patience:
            log.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {args.patience} epochs)"
            )
            break

    log.info(f"Training complete. Best val IoU: {best_val_iou:.4f}")

    # ── 测试集评估 ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Loading best checkpoint for test set evaluation ...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    log.info(f"  Loaded from epoch {ckpt['epoch']} (val IoU = {ckpt['val_iou']:.4f})")

    test_metrics = test_epoch(model, test_loader, device, threshold=0.5)

    log.info("=" * 60)
    log.info("Test set results (mean ± std, threshold=0.5):")
    for metric, (mean, std) in test_metrics.items():
        log.info(f"  {metric:10s}: {mean:.4f} ± {std:.4f}")
    log.info("=" * 60)

    result_path = out_dir / "test_metrics.txt"
    with open(result_path, "w") as f:
        f.write(f"Best checkpoint: epoch {ckpt['epoch']}, val_iou={ckpt['val_iou']:.4f}\n")
        f.write(f"Test set size: {len(test_ds)}\n")
        f.write(f"Threshold: 0.5\n\n")
        for metric, (mean, std) in test_metrics.items():
            f.write(f"{metric}: {mean:.4f} +/- {std:.4f}\n")
    log.info(f"Test metrics saved to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",         default="./data/data/events.h5")
    parser.add_argument("--channel",    default="pgv", choices=["pga", "pgv", "int"])
    parser.add_argument("--workers",    type=int, default=0)
    parser.add_argument("--sparse_sim", action="store_true")
    parser.add_argument("--n_stations", type=int, default=80)

    parser.add_argument("--patch_size",            type=int, default=10)
    parser.add_argument("--d_model",               type=int, default=256)
    parser.add_argument("--n_layers",              type=int, default=6)
    parser.add_argument("--grid_size",             type=int, default=150)
    parser.add_argument("--decoder_dim",           type=int, default=32)
    parser.add_argument("--decoder_mid_dim",       type=int, default=64)
    parser.add_argument("--decoder_refine_blocks", type=int, default=3)

    parser.add_argument("--commit_tau",  type=float, default=0.60)
    parser.add_argument("--commit_temp", type=float, default=0.08)

    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--wd",         type=float, default=1e-4)
    parser.add_argument("--patience",   type=int,   default=20)
    parser.add_argument("--seed",       type=int,   default=42)

    parser.add_argument("--belief_w",    type=float, default=0.6)
    parser.add_argument("--committed_w", type=float, default=0.5)
    parser.add_argument("--dice_w",      type=float, default=0.5)
    parser.add_argument("--bce_w",       type=float, default=0.5)
    parser.add_argument("--tv_w",        type=float, default=0.15)
    parser.add_argument("--dir_w",       type=float, default=0.10)
    parser.add_argument("--pos_weight", type=float, default=3.0,
                        help="BCE positive-class weight for sparse fault pixels")
    parser.add_argument("--min_ecc",    type=float, default=0.01,
                        help="Eccentricity gate in direction loss (NaN guard)")

    parser.add_argument("--only_valid", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--causal",     action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out",        default="checkpoints_rule_hybrid")

    main(parser.parse_args())