"""Single-entry WSSED pipeline with joint weak training and class-specific strong retraining."""

from __future__ import annotations

import colorsys
import csv
import importlib
import importlib.util
import random
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.optim import Adam

import config
from data_fnjv import DatasetBundle, load_datasets, summarize_bundle
from birdnet_direct import BirdNetDirectExtractor, import_birdnet_analyzer_strict
from embedding_io import embedding_cache_path, load_precomputed_txt_embeddings, load_segment_embeddings
from preprocessing import InstanceWindow, load_audio_mono, split_into_windows


VERBOSE_DEBUG = False
CLASS_COLOR_MAP: Dict[str, tuple] = {}


def _debug(message: str) -> None:
    if VERBOSE_DEBUG:
        print(message)


def print_config() -> None:
    if not VERBOSE_DEBUG:
        return
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for key in sorted(name for name in dir(config) if name.isupper() or name in {"threshold"}):
        print(f"{key} = {getattr(config, key)}")
    print("=" * 80)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_birdnet_analyzer_import() -> None:
    import_birdnet_analyzer_strict()


def _auto_extract_single_audio_direct(audio_path: str) -> None:
    from extract_embeddings import extract_for_audio

    extractor = BirdNetDirectExtractor()
    extract_for_audio(Path(audio_path), Path(config.BIRDNET_CACHE_DIR), extractor)
    _debug("[birdnet] direct extractor cache generation: OK")


def _fallback_features_for_debug(windows: List[InstanceWindow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not config.USE_FALLBACK_FEATURES:
        raise RuntimeError("BirdNET is not being used and fallback features are disabled.")

    target_dim = int(getattr(config, "EXPECTED_EMBED_DIM", 1024))
    embs = []
    for instance in windows:
        waveform = instance.waveform.squeeze(0)
        n_fft = int(config.N_FFT)
        hop_length = max(1, n_fft // 4)
        window = torch.hann_window(n_fft, device=waveform.device)
        spectrum = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        mag = torch.log1p(torch.abs(spectrum))
        pooled = torch.mean(mag, dim=1)
        if pooled.numel() >= target_dim:
            embedding = pooled[:target_dim]
        else:
            embedding = torch.nn.functional.pad(pooled, (0, target_dim - pooled.numel()))
        embs.append(embedding.detach().cpu().numpy())

    starts = np.asarray([w.start_sec for w in windows], dtype=np.float32)
    ends = np.asarray([w.end_sec for w in windows], dtype=np.float32)
    return np.stack(embs, axis=0), starts, ends


@dataclass
class BagData:
    bag_id: str
    audio_path: str
    code: str
    label_code: str
    embeddings: torch.Tensor
    instance_windows: List[InstanceWindow]
    strong_events: List[Tuple[float, float, str]]
    is_negative: bool = False


def _load_embeddings_for_audio(audio_path: str, windows: List[InstanceWindow]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = config.EMBEDDING_SOURCE
    _debug(f"[birdnet] embedding_source={source}")

    if source == "birdnet_analyzer":
        _require_birdnet_analyzer_import()
        npz_path = embedding_cache_path(audio_path, config.BIRDNET_CACHE_DIR)
        if not npz_path.exists() and config.ALLOW_AUTO_EXTRACT:
            _debug(f"[birdnet] cache miss: {npz_path}")
            _auto_extract_single_audio_direct(audio_path)
        if not npz_path.exists():
            raise RuntimeError(
                "BirdNET embeddings not found in cache. "
                f"Expected: {npz_path}. Run extract_embeddings.py first or set ALLOW_AUTO_EXTRACT=True."
            )

        embeddings, starts, ends = load_segment_embeddings(npz_path)
        _debug(f"[birdnet] cache hit: {npz_path}")
        _debug(f"[birdnet] embedding_matrix_shape={embeddings.shape}")
        setattr(config, "EXPECTED_EMBED_DIM", int(embeddings.shape[1]))
        return embeddings, starts, ends

    if source == "precomputed_txt":
        txt_path = Path(config.PRECOMPUTED_TXT_DIR) / f"{Path(audio_path).stem}.birdnet.embeddings.txt"
        if not txt_path.exists():
            raise RuntimeError(f"Precomputed txt embedding file not found: {txt_path}")
        embeddings, starts, ends = load_precomputed_txt_embeddings(
            txt_path,
            hop_seconds=config.HOP_SECONDS,
            window_seconds=config.WINDOW_SECONDS,
        )
        _debug(f"[birdnet] loaded txt file={txt_path}")
        _debug(f"[birdnet] embedding_matrix_shape={embeddings.shape}")
        setattr(config, "EXPECTED_EMBED_DIM", int(embeddings.shape[1]))
        return embeddings, starts, ends

    raise RuntimeError(f"Unsupported EMBEDDING_SOURCE: {source}")


def build_bags(df) -> List[BagData]:
    bags: List[BagData] = []
    for row in df.to_dict(orient="records"):
        bag_id = str(
            row.get("Numero de catalogo")
            or row.get("Arquivo do registro")
            or row.get("audio_path")
            or "unknown"
        )
        audio_path = str(row["audio_path"])
        code = str(row["Code"])
        label_code = str(row.get("label_code", code))
        strong_events = row.get("strong_events", [])
        is_negative = bool(row.get("is_negative", False))

        mono_waveform, sample_rate = load_audio_mono(audio_path)
        windows = split_into_windows(
            bag_id=bag_id,
            mono_waveform=mono_waveform,
            sample_rate=sample_rate,
            window_seconds=config.WINDOW_SECONDS,
            hop_seconds=config.HOP_SECONDS,
        )
        if not windows:
            continue

        try:
            embeddings_np, starts, ends = _load_embeddings_for_audio(audio_path, windows)
        except Exception:
            if not config.USE_FALLBACK_FEATURES:
                raise RuntimeError("BirdNET kullanılmıyor")
            embeddings_np, starts, ends = _fallback_features_for_debug(windows)
            _debug("[birdnet] WARNING: fallback STFT features are active (debug only).")

        if embeddings_np.ndim != 2:
            raise RuntimeError(f"BirdNET evidence failed: invalid embedding shape {embeddings_np.shape}")

        seg_count = embeddings_np.shape[0]
        if seg_count == 0:
            raise RuntimeError("BirdNET evidence failed: no segments in embedding matrix")
        _debug(
            f"[birdnet] segment_times_first_last=({starts[0]:.2f},{ends[0]:.2f}) -> ({starts[-1]:.2f},{ends[-1]:.2f}) "
            f"window={config.WINDOW_SECONDS}s hop={config.HOP_SECONDS}s"
        )

        if seg_count != len(windows):
            raise RuntimeError(
                f"Segment count mismatch: embeddings={seg_count}, windows={len(windows)} for {audio_path}"
            )

        embeddings = torch.from_numpy(embeddings_np).float()
        bags.append(
            BagData(
                bag_id=bag_id,
                audio_path=audio_path,
                code=code,
                label_code=label_code,
                embeddings=embeddings,
                instance_windows=windows,
                strong_events=strong_events,
                is_negative=is_negative,
            )
        )
    return bags


class JointMilModel(nn.Module):
    """Stage-1 model: joint weak multi-label training with Linear(in_dim, n_classes)."""

    def __init__(self, in_dim: int, n_classes: int, pooling: str = "lin") -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
        self.pooling = pooling

    def forward_segment_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings)  # [n_instances, n_classes]

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inst_logits = self.forward_segment_logits(embeddings)
        inst_probs = torch.sigmoid(inst_logits)

        if self.pooling == "max":
            bag_probs = torch.max(inst_probs, dim=0).values
        elif self.pooling == "avg":
            bag_probs = torch.mean(inst_probs, dim=0)
        elif self.pooling == "exp":
            numerator = torch.sum(inst_probs * torch.exp(inst_probs), dim=0)
            denominator = torch.sum(torch.exp(inst_probs), dim=0) + 1e-8
            bag_probs = numerator / denominator
        else:  # lin
            numerator = torch.sum(inst_probs * inst_probs, dim=0)
            denominator = torch.sum(inst_probs, dim=0) + 1e-8
            bag_probs = numerator / denominator

        bag_logits = torch.logit(torch.clamp(bag_probs, min=1e-6, max=1 - 1e-6))
        return bag_logits, inst_logits


class ClassSpecificHead(nn.Module):
    """Stage-2 model: per-class head Linear(in_dim, 1) for strong-label retraining."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.linear(embeddings).squeeze(-1)


@dataclass
class SnippetScore:
    class_code: str
    bag_id: str
    audio_path: str
    instance_index: int
    start_sec: float
    end_sec: float
    score: float


def build_species_index(species_codes: List[str]) -> Dict[str, int]:
    return {code: idx for idx, code in enumerate(species_codes)}


def build_weak_target(code: str, species_to_idx: Dict[str, int], n_classes: int) -> torch.Tensor:
    """Build bag-level weak target only.

    Stage-1 supervision remains bag-level and never uses snippet-level labels.
    """
    target = torch.zeros(n_classes, dtype=torch.float32)
    if code is None or str(code).strip() == "":
        return target
    if not config.MULTI_LABEL:
        if code in species_to_idx:
            target[species_to_idx[code]] = 1.0
        return target

    # Placeholder for future true multi-label bags when bag metadata provides multiple class codes.
    # For current FNJV setup, `code` is a single weak bag label.
    if code in species_to_idx:
        target[species_to_idx[code]] = 1.0
    return target


def build_bag_target(
    bag: BagData,
    species_to_idx: Dict[str, int],
    n_classes: int,
    use_strong_if_available: bool = False,
) -> torch.Tensor:
    if use_strong_if_available and bag.strong_events:
        target = torch.zeros(n_classes, dtype=torch.float32)
        for _start, _end, label in bag.strong_events:
            if label in species_to_idx:
                target[species_to_idx[label]] = 1.0
        if torch.any(target > 0):
            return target
    return build_weak_target(bag.label_code, species_to_idx, n_classes)


def _compute_pos_weight_from_train(
    train_bags: List[BagData],
    species_codes: List[str],
) -> torch.Tensor:
    species_to_idx = build_species_index(species_codes)
    y = _bag_target_matrix(train_bags, species_codes, use_strong_if_available=False)
    if y.size == 0:
        return torch.ones(len(species_codes), dtype=torch.float32)
    pos = y.sum(axis=0)
    neg = float(y.shape[0]) - pos
    pos = np.maximum(pos, 1.0)
    raw = neg / pos
    clamp_max = float(getattr(config, "WEIGHTED_BCE_POS_WEIGHT_CLAMP", 20.0))
    raw = np.clip(raw, 1.0, clamp_max)
    return torch.tensor(raw, dtype=torch.float32)


def _build_loss_fn(
    loss_name: str,
    species_codes: List[str],
    train_bags: Optional[List[BagData]] = None,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    name = str(loss_name).strip().lower()

    if name == "weighted_bce":
        if train_bags is None:
            raise ValueError("weighted_bce requires train_bags to compute pos_weight")
        pos_weight = _compute_pos_weight_from_train(train_bags, species_codes)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        return lambda logits, target: criterion(logits, target)

    if name == "focal":
        gamma = float(getattr(config, "FOCAL_GAMMA", 2.0))
        alpha = float(getattr(config, "FOCAL_ALPHA", 0.25))

        def focal_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            probs = torch.sigmoid(logits)
            pt = probs * target + (1.0 - probs) * (1.0 - target)
            alpha_factor = alpha * target + (1.0 - alpha) * (1.0 - target)
            modulating = (1.0 - pt) ** gamma
            return (alpha_factor * modulating * bce).mean()

        return focal_loss

    if name == "asl":
        gamma_pos = float(getattr(config, "ASL_GAMMA_POS", 0.0))
        gamma_neg = float(getattr(config, "ASL_GAMMA_NEG", 4.0))
        clip = float(getattr(config, "ASL_CLIP", 0.05))

        def asl_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            xs_pos = torch.sigmoid(logits)
            xs_neg = 1.0 - xs_pos
            if clip > 0:
                xs_neg = torch.clamp(xs_neg + clip, max=1.0)

            eps = 1e-8
            los_pos = target * torch.log(torch.clamp(xs_pos, min=eps))
            los_neg = (1.0 - target) * torch.log(torch.clamp(xs_neg, min=eps))

            if gamma_pos > 0 or gamma_neg > 0:
                pt = xs_pos * target + xs_neg * (1.0 - target)
                one_sided_gamma = gamma_pos * target + gamma_neg * (1.0 - target)
                one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
                los_pos = los_pos * one_sided_w
                los_neg = los_neg * one_sided_w

            return (-(los_pos + los_neg)).mean()

        return asl_loss

    # default bce
    criterion = nn.BCEWithLogitsLoss()
    return lambda logits, target: criterion(logits, target)


def _score_entropy(frame_pred: np.ndarray) -> float:
    eps = 1e-8
    p = np.clip(frame_pred.astype(np.float64), eps, 1.0 - eps)
    h = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(np.mean(h)) if h.size else 0.0


def _kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return np.zeros_like(grid, dtype=np.float64)
    std = np.std(v)
    bw = 1.06 * max(std, 1e-3) * (v.size ** (-1.0 / 5.0))
    bw = max(bw, 1e-3)
    diff = (grid[:, None] - v[None, :]) / bw
    dens = np.exp(-0.5 * diff * diff).sum(axis=1) / (v.size * bw * np.sqrt(2.0 * np.pi))
    return dens


def _save_epoch_distribution_artifacts(
    epoch_idx: int,
    model: JointMilModel,
    bags: List[BagData],
    class_names: List[str],
    history: Dict[str, List[float]],
) -> None:
    if importlib.util.find_spec('matplotlib') is None:
        _debug('[epoch-artifacts] matplotlib unavailable, skipping epoch plots')
        return
    import matplotlib.pyplot as plt

    epoch_dir = Path('outputs') / 'epochs' / f'epoch_{epoch_idx:04d}'
    epoch_dir.mkdir(parents=True, exist_ok=True)
    color_map = _build_class_color_map(class_names)

    by_class: Dict[str, List[float]] = {c: [] for c in class_names}
    all_scores: List[float] = []
    entropies: List[float] = []

    model.eval()
    with torch.no_grad():
        for bag in bags:
            if bag.embeddings.shape[0] == 0:
                continue
            probs = torch.sigmoid(model.forward_segment_logits(bag.embeddings)).cpu().numpy()
            all_scores.extend(probs.reshape(-1).tolist())
            entropies.append(_score_entropy(probs))
            for c_idx, cname in enumerate(class_names):
                by_class[cname].extend(probs[:, c_idx].tolist())

    # per-class hist and kde
    grid = np.linspace(0.0, 1.0, 300, dtype=np.float64)
    for cname in class_names:
        vals = np.asarray(by_class[cname], dtype=np.float64)
        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        ax_h.hist(vals, bins=30, range=(0.0, 1.0), color=color_map[cname], alpha=0.8)
        ax_h.set_title(f'Histogram - {cname} (epoch {epoch_idx})')
        ax_h.set_xlim(0.0, 1.0)
        ax_h.grid(True, alpha=0.3)
        fig_h.tight_layout()
        fig_h.savefig(epoch_dir / f'hist_{cname}.png', dpi=150)
        plt.close(fig_h)

        fig_k, ax_k = plt.subplots(figsize=(8, 4))
        ax_k.plot(grid, _kde_curve(vals, grid), color=color_map[cname], linewidth=2.0)
        ax_k.set_title(f'KDE - {cname} (epoch {epoch_idx})')
        ax_k.set_xlim(0.0, 1.0)
        ax_k.grid(True, alpha=0.3)
        fig_k.tight_layout()
        fig_k.savefig(epoch_dir / f'kde_{cname}.png', dpi=150)
        plt.close(fig_k)

    # combined hist
    fig_ch, ax_ch = plt.subplots(figsize=(10, 5))
    for cname in class_names:
        vals = np.asarray(by_class[cname], dtype=np.float64)
        ax_ch.hist(vals, bins=30, range=(0.0, 1.0), density=True, histtype='step', linewidth=1.5, color=color_map[cname], label=cname)
    ax_ch.set_title(f'Histogram Combined (epoch {epoch_idx})')
    ax_ch.set_xlim(0.0, 1.0)
    ax_ch.grid(True, alpha=0.3)
    ax_ch.legend(fontsize=7)
    fig_ch.tight_layout()
    fig_ch.savefig(epoch_dir / 'hist_combined.png', dpi=150)
    plt.close(fig_ch)

    # combined kde
    fig_ck, ax_ck = plt.subplots(figsize=(10, 5))
    for cname in class_names:
        vals = np.asarray(by_class[cname], dtype=np.float64)
        ax_ck.plot(grid, _kde_curve(vals, grid), color=color_map[cname], linewidth=1.8, label=cname)
    ax_ck.set_title(f'KDE Combined (epoch {epoch_idx})')
    ax_ck.set_xlim(0.0, 1.0)
    ax_ck.grid(True, alpha=0.3)
    ax_ck.legend(fontsize=7)
    fig_ck.tight_layout()
    fig_ck.savefig(epoch_dir / 'kde_combined.png', dpi=150)
    plt.close(fig_ck)

    # cumulative curves up to epoch
    ep = np.arange(1, len(history['train_loss']) + 1)
    current_entropy = float(np.mean(entropies)) if entropies else 0.0
    entropy_hist = list(history.get('epoch_entropy', [])) + [current_entropy]

    fig_e, ax_e = plt.subplots(figsize=(8, 4))
    ax_e.plot(ep, entropy_hist, color='black', linewidth=2)
    ax_e.set_title('Entropy vs Epochs')
    ax_e.set_xlabel('Epoch')
    ax_e.set_ylabel('Entropy')
    ax_e.grid(True, alpha=0.3)
    fig_e.tight_layout()
    fig_e.savefig(epoch_dir / 'entropy_vs_epochs.png', dpi=150)
    plt.close(fig_e)

    fig_l, ax_l = plt.subplots(figsize=(8, 4))
    ax_l.plot(ep, history['train_loss'], label='train_loss', color='tab:blue')
    ax_l.plot(ep, history['val_loss'], label='val_loss', color='tab:orange')
    ax_l.set_title('Loss vs Epochs')
    ax_l.set_xlabel('Epoch')
    ax_l.set_ylabel('Loss')
    ax_l.grid(True, alpha=0.3)
    ax_l.legend()
    fig_l.tight_layout()
    fig_l.savefig(epoch_dir / 'loss_vs_epochs.png', dpi=150)
    plt.close(fig_l)

    fig_p, ax_p = plt.subplots(figsize=(8, 4))
    ax_p.plot(ep, history['train_micro_f1'], label='train_micro_f1', color='tab:green')
    ax_p.plot(ep, history['val_micro_f1'], label='val_micro_f1', color='tab:red')
    ax_p.plot(ep, history['train_macro_f1'], label='train_macro_f1', color='tab:purple')
    ax_p.plot(ep, history['val_macro_f1'], label='val_macro_f1', color='tab:brown')
    ax_p.set_title('Performance vs Epochs')
    ax_p.set_xlabel('Epoch')
    ax_p.set_ylabel('F1')
    ax_p.set_ylim(-0.1, 1.1)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(fontsize=8)
    fig_p.tight_layout()
    fig_p.savefig(epoch_dir / 'performance_vs_epochs.png', dpi=150)
    plt.close(fig_p)

    # all-in-one subplot
    fig_all, axes = plt.subplots(3, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    for cname in class_names:
        vals = np.asarray(by_class[cname], dtype=np.float64)
        ax1.hist(vals, bins=30, range=(0.0, 1.0), density=True, histtype='step', linewidth=1.2, color=color_map[cname], label=cname)
        ax2.plot(grid, _kde_curve(vals, grid), color=color_map[cname], linewidth=1.4, label=cname)
    ax1.set_title('Combined Histogram')
    ax2.set_title('Combined KDE')
    ax1.set_xlim(0.0, 1.0); ax2.set_xlim(0.0, 1.0)
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    ax1.legend(fontsize=6); ax2.legend(fontsize=6)

    ax3.plot(ep, entropy_hist, color='black', linewidth=2)
    ax3.set_title('Entropy vs Epochs')
    ax3.grid(True, alpha=0.3)

    ax4.plot(ep, history['train_loss'], label='train')
    ax4.plot(ep, history['val_loss'], label='val')
    ax4.set_title('Loss vs Epochs')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=7)

    ax5.plot(ep, history['train_micro_f1'], label='train_micro')
    ax5.plot(ep, history['val_micro_f1'], label='val_micro')
    ax5.plot(ep, history['train_macro_f1'], label='train_macro')
    ax5.plot(ep, history['val_macro_f1'], label='val_macro')
    ax5.set_title('Performance vs Epochs')
    ax5.set_ylim(-0.1, 1.1)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=7)

    # distribution summary
    ax6.hist(np.asarray(all_scores, dtype=np.float64), bins=30, range=(0.0, 1.0), color='gray', alpha=0.8)
    ax6.set_title('All Scores Histogram')
    ax6.set_xlim(0.0, 1.0)
    ax6.grid(True, alpha=0.3)

    fig_all.tight_layout()
    fig_all.savefig(epoch_dir / 'all_in_one_subplot.png', dpi=150)
    plt.close(fig_all)

    history.setdefault('epoch_entropy', []).append(current_entropy)


def _bag_target_matrix(
    bags: List[BagData],
    species_codes: List[str],
    use_strong_if_available: bool = False,
) -> np.ndarray:
    species_to_idx = build_species_index(species_codes)
    y = np.zeros((len(bags), len(species_codes)), dtype=np.float32)
    for i, bag in enumerate(bags):
        y[i] = build_bag_target(
            bag,
            species_to_idx,
            len(species_codes),
            use_strong_if_available=use_strong_if_available,
        ).numpy()
    return y


def _build_per_class_metric_rows(y_true: np.ndarray, y_pred: np.ndarray, species_codes: List[str]) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    if y_true.size == 0:
        return rows
    for c_idx, cname in enumerate(species_codes):
        yt = y_true[:, c_idx].astype(np.int32)
        yp = y_pred[:, c_idx].astype(np.int32)
        p = precision_score(yt, yp, zero_division=0)
        r = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        rows[cname] = {
            'micro_precision': float(p),
            'micro_recall': float(r),
            'micro_f1': float(f1),
            'micro_er': float(1.0 - f1),
            'macro_precision': float(p),
            'macro_recall': float(r),
            'macro_f1': float(f1),
            'macro_er': float(1.0 - f1),
        }
    return rows


def _format_metric_block(metrics: Dict[str, float], include_macro: bool = True) -> Dict[str, float]:
    out = {
        'micro_precision': round(float(metrics.get('micro_precision', 0.0)), 4),
        'micro_recall': round(float(metrics.get('micro_recall', 0.0)), 4),
        'micro_f1': round(float(metrics.get('micro_f1', 0.0)), 4),
        'micro_er': round(float(metrics.get('micro_er', 0.0)), 4),
    }
    if include_macro:
        out.update({
            'macro_precision': round(float(metrics.get('macro_precision', 0.0)), 4),
            'macro_recall': round(float(metrics.get('macro_recall', 0.0)), 4),
            'macro_f1': round(float(metrics.get('macro_f1', 0.0)), 4),
            'macro_er': round(float(metrics.get('macro_er', 0.0)), 4),
        })
    return out


def _eval_bag_metrics(
    model: JointMilModel,
    bags: List[BagData],
    species_codes: List[str],
    use_strong_if_available: bool = False,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    criterion = loss_fn if loss_fn is not None else _build_loss_fn("bce", species_codes)
    y_true = _bag_target_matrix(bags, species_codes, use_strong_if_available=use_strong_if_available)
    y_prob = []
    losses = []
    species_to_idx = build_species_index(species_codes)

    model.eval()
    with torch.no_grad():
        for bag in bags:
            target = build_bag_target(
                bag,
                species_to_idx,
                len(species_codes),
                use_strong_if_available=use_strong_if_available,
            )
            bag_logits, _ = model(bag.embeddings)
            losses.append(float(criterion(bag_logits, target).item()))
            y_prob.append(torch.sigmoid(bag_logits).cpu().numpy())

    y_prob_np = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob_np >= float(config.threshold)).astype(np.int32)
    y_true_i = y_true.astype(np.int32)

    micro_f1 = f1_score(y_true_i.reshape(-1), y_pred.reshape(-1), zero_division=0)
    macro_f1 = f1_score(y_true_i, y_pred, average='macro', zero_division=0)
    micro_p = precision_score(y_true_i.reshape(-1), y_pred.reshape(-1), zero_division=0)
    macro_p = precision_score(y_true_i, y_pred, average='macro', zero_division=0)
    micro_r = recall_score(y_true_i.reshape(-1), y_pred.reshape(-1), zero_division=0)
    macro_r = recall_score(y_true_i, y_pred, average='macro', zero_division=0)

    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'micro_precision': float(micro_p),
        'macro_precision': float(macro_p),
        'micro_recall': float(micro_r),
        'macro_recall': float(macro_r),
        'micro_er': float(1.0 - micro_f1),
        'macro_er': float(1.0 - macro_f1),
        'per_class': _build_per_class_metric_rows(y_true_i, y_pred, species_codes),
    }


def _label_from_events(window: InstanceWindow, class_code: str, events: List[Tuple[float, float, str]]) -> int:
    if not events:
        return 0
    for start, end, label in events:
        if label != class_code:
            continue
        if max(window.start_sec, float(start)) < min(window.end_sec, float(end)):
            return 1
    return 0


def _eval_snippet_metrics(
    model: JointMilModel,
    bags: List[BagData],
    species_codes: List[str],
    use_strong_if_available: bool = False,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, float]:
    y_true = []
    y_pred = []
    losses = []
    criterion = loss_fn if loss_fn is not None else _build_loss_fn("bce", species_codes)
    species_to_idx = build_species_index(species_codes)

    model.eval()
    overlap_debug_logged = False
    with torch.no_grad():
        for bag in bags:
            bag_target = build_bag_target(
                bag,
                species_to_idx,
                len(species_codes),
                use_strong_if_available=use_strong_if_available,
            )
            bag_logits, _ = model(bag.embeddings)
            inst_logits = model.forward_segment_logits(bag.embeddings)
            losses.append(float(criterion(bag_logits, bag_target).item()))
            inst_prob = torch.sigmoid(inst_logits).cpu().numpy()
            pred = (inst_prob >= float(config.threshold)).astype(np.int32)
            y_pred.append(pred)

            true_rows = []
            for w in bag.instance_windows:
                row = [_label_from_events(w, c, bag.strong_events) for c in species_codes]
                true_rows.append(row)

            if VERBOSE_DEBUG and (not overlap_debug_logged) and bag.instance_windows:
                print(f"[eval-overlap] example bag={bag.bag_id} audio={bag.audio_path}")
                for idx, w in enumerate(bag.instance_windows[:10]):
                    row = true_rows[idx]
                    print(
                        f"[eval-overlap] window_{idx}: [{w.start_sec:.2f},{w.end_sec:.2f}] target={row}"
                    )
                print(f"[eval-overlap] strong_events={bag.strong_events[:10]}")
                overlap_debug_logged = True
            y_true.append(np.asarray(true_rows, dtype=np.int32))

    if not y_true:
        out = {k: 0.0 for k in ['loss','micro_f1','macro_f1','micro_precision','macro_precision','micro_recall','macro_recall','micro_er','macro_er']}
        out['per_class'] = {}
        return out

    yt = np.concatenate(y_true, axis=0)
    yp = np.concatenate(y_pred, axis=0)

    micro_f1 = f1_score(yt.reshape(-1), yp.reshape(-1), zero_division=0)
    macro_f1 = f1_score(yt, yp, average='macro', zero_division=0)
    micro_p = precision_score(yt.reshape(-1), yp.reshape(-1), zero_division=0)
    macro_p = precision_score(yt, yp, average='macro', zero_division=0)
    micro_r = recall_score(yt.reshape(-1), yp.reshape(-1), zero_division=0)
    macro_r = recall_score(yt, yp, average='macro', zero_division=0)

    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'micro_f1': float(micro_f1),
        'macro_f1': float(macro_f1),
        'micro_precision': float(micro_p),
        'macro_precision': float(macro_p),
        'micro_recall': float(micro_r),
        'macro_recall': float(macro_r),
        'micro_er': float(1.0 - micro_f1),
        'macro_er': float(1.0 - macro_f1),
        'per_class': _build_per_class_metric_rows(yt, yp, species_codes),
    }


def _plot_history(history: Dict[str, List[float]]) -> None:
    if importlib.util.find_spec('matplotlib') is None:
        _debug('[plot] matplotlib unavailable, skipping plots')
        return
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig1, axes1 = plt.subplots(3, 1, figsize=(8, 10))
    axes1[0].plot(epochs, history['train_loss'], label='train_loss')
    axes1[0].plot(epochs, history['val_loss'], label='val_loss')
    axes1[0].set_title('Loss')
    axes1[1].plot(epochs, history['train_micro_f1'], label='train_micro_f1')
    axes1[1].plot(epochs, history['val_micro_f1'], label='val_micro_f1')
    axes1[1].plot(epochs, history['train_macro_f1'], label='train_macro_f1')
    axes1[1].plot(epochs, history['val_macro_f1'], label='val_macro_f1')
    axes1[1].set_ylim(-0.1, 1.1)
    axes1[1].set_title('F1')
    axes1[2].plot(epochs, history['train_micro_precision'], label='train_micro_precision')
    axes1[2].plot(epochs, history['val_micro_precision'], label='val_micro_precision')
    axes1[2].plot(epochs, history['train_macro_precision'], label='train_macro_precision')
    axes1[2].plot(epochs, history['val_macro_precision'], label='val_macro_precision')
    axes1[2].set_ylim(-0.1, 1.1)
    axes1[2].set_title('Precision')
    for ax in axes1:
        ax.grid(True)
        ax.legend()
    fig1.tight_layout()
    fig1.savefig('metrics_f1_precision.png')
    plt.close(fig1)

    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 8))
    axes2[0].plot(epochs, history['train_micro_recall'], label='train_micro_recall')
    axes2[0].plot(epochs, history['val_micro_recall'], label='val_micro_recall')
    axes2[0].plot(epochs, history['train_macro_recall'], label='train_macro_recall')
    axes2[0].plot(epochs, history['val_macro_recall'], label='val_macro_recall')
    axes2[0].set_ylim(-0.1, 1.1)
    axes2[0].set_title('Recall')
    axes2[1].plot(epochs, history['train_micro_er'], label='train_micro_er')
    axes2[1].plot(epochs, history['val_micro_er'], label='val_micro_er')
    axes2[1].plot(epochs, history['train_macro_er'], label='train_macro_er')
    axes2[1].plot(epochs, history['val_macro_er'], label='val_macro_er')
    axes2[1].set_ylim(-0.1, 1.1)
    axes2[1].set_title('ER')
    for ax in axes2:
        ax.grid(True)
        ax.legend()
    fig2.tight_layout()
    fig2.savefig('metrics_recall_er.png')
    plt.close(fig2)



def _build_class_color_map(class_names: List[str]) -> Dict[str, tuple]:
    global CLASS_COLOR_MAP
    if importlib.util.find_spec('matplotlib') is None:
        CLASS_COLOR_MAP = {name: (0.0, 0.0, 0.0, 1.0) for name in class_names}
        return CLASS_COLOR_MAP

    import matplotlib.pyplot as plt

    c_count = len(class_names)
    if c_count == 0:
        CLASS_COLOR_MAP = {}
        return CLASS_COLOR_MAP

    if "glasbey" in set(plt.colormaps()):
        cmap = plt.get_cmap("glasbey")
        colors = [cmap(i / max(1, c_count - 1)) for i in range(c_count)]
    else:
        # Deterministic high-separation colors in HSV space.
        colors = []
        for i in range(c_count):
            h = float(i) / float(c_count)
            s_v = 0.9
            v_v = 0.9
            r, g, b = colorsys.hsv_to_rgb(h, s_v, v_v)
            colors.append((r, g, b, 1.0))

    CLASS_COLOR_MAP = {class_names[i]: colors[i] for i in range(c_count)}
    return CLASS_COLOR_MAP


def _build_frame_true_for_bag(bag: BagData, class_names: List[str]) -> np.ndarray:
    rows = []
    for w in bag.instance_windows:
        rows.append([_label_from_events(w, c, bag.strong_events) for c in class_names])
    if not rows:
        return np.zeros((0, len(class_names)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _build_frame_pred_for_bag(model: JointMilModel, bag: BagData) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inst_logits = model.forward_segment_logits(bag.embeddings)
        return torch.sigmoid(inst_logits).cpu().numpy().astype(np.float32)


def plot_two_panel_localization(
    frame_true: Optional[np.ndarray],
    frame_pred: np.ndarray,
    class_names: List[str],
    window_seconds: float,
    hop_seconds: float,
    audio_duration_seconds: float,
    out_path: Optional[str] = None,
    colored_gt: bool = False,
):
    if importlib.util.find_spec('matplotlib') is None:
        _debug('[viz] matplotlib unavailable, skipping two-panel visualization')
        return None
    import matplotlib.pyplot as plt

    if frame_pred.ndim != 2:
        raise ValueError(f"frame_pred must be 2D [T,C], got {frame_pred.shape}")
    t_count, c_count = frame_pred.shape
    if len(class_names) != c_count:
        raise ValueError(f"class_names length mismatch: len(class_names)={len(class_names)} vs C={c_count}")

    if frame_true is not None:
        if frame_true.ndim != 2:
            raise ValueError(f"frame_true must be 2D [T,C], got {frame_true.shape}")
        if frame_true.shape != frame_pred.shape:
            raise ValueError(f"frame_true/frame_pred shape mismatch: {frame_true.shape} vs {frame_pred.shape}")

    time_centers = np.arange(t_count, dtype=np.float32) * float(hop_seconds) + float(window_seconds) / 2.0
    class_color_map = _build_class_color_map(class_names)

    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # add spacer rows between classes for readability
    spacer_label = "....."
    y_labels: List[str] = []
    row_centers: List[float] = []
    expanded_rows = c_count * 2 - 1 if c_count > 0 else 0

    if frame_true is None or frame_true.size == 0:
        ax_gt.text(0.5, 0.5, 'GT unavailable', transform=ax_gt.transAxes, ha='center', va='center')
        for i, cname in enumerate(class_names):
            y_labels.append(cname)
            row_centers.append(i * 2 + 0.5)
            if i < c_count - 1:
                y_labels.append(spacer_label)
                row_centers.append(i * 2 + 1.5)
        if row_centers:
            ax_gt.set_yticks(row_centers)
            ax_gt.set_yticklabels(y_labels, fontsize=8)
        ax_gt.set_ylim(0, max(expanded_rows, 1))
    else:
        expanded = np.zeros((expanded_rows, t_count), dtype=np.float32)
        for i in range(c_count):
            expanded[i * 2, :] = frame_true[:, i]
            y_labels.append(class_names[i])
            row_centers.append(i * 2 + 0.5)
            if i < c_count - 1:
                y_labels.append(spacer_label)
                row_centers.append(i * 2 + 1.5)

        extent = [float(time_centers[0]), float(time_centers[-1]), 0, max(expanded_rows, 1)]
        if colored_gt:
            rgb = np.ones((expanded_rows, t_count, 3), dtype=np.float32)
            for i, cname in enumerate(class_names):
                color = np.asarray(class_color_map[cname][:3], dtype=np.float32)
                mask = expanded[i * 2, :] >= 0.5
                rgb[i * 2, mask, :] = color
            ax_gt.imshow(rgb, aspect='auto', origin='lower', extent=extent)
        else:
            ax_gt.imshow(expanded, cmap="Greys", vmin=0, vmax=1, aspect="auto", origin="lower", extent=extent)

        ax_gt.set_yticks(row_centers)
        ax_gt.set_yticklabels(y_labels, fontsize=8)
    ax_gt.set_title("Ground Truth (strong labels) - all classes")

    for c in range(c_count):
        class_name = class_names[c]
        ax_pred.plot(time_centers, frame_pred[:, c], color=class_color_map[class_name], label=class_name, linewidth=1.5)

    ax_pred.set_title("Model snippet-wise predictions - all classes")
    ax_pred.set_ylim(-0.1, 1.1)
    ax_pred.set_xlim(0.0, float(audio_duration_seconds))
    ax_pred.grid(True, alpha=0.3)
    ax_pred.set_xlabel("Time (s)")
    ax_pred.set_ylabel("Presence score")
    ax_pred.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)

    fig.tight_layout()
    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_path



def _save_split_visualizations(
    split_name: str,
    bags: List[BagData],
    model: JointMilModel,
    class_names: List[str],
    output_dir: Path,
) -> None:
    if not bags:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    used_names: set[str] = set()
    neg_count = sum(1 for b in bags if bool(getattr(b, "is_negative", False)))
    print(f"[{split_name}] bags={len(bags)} negatives={neg_count}")
    for bag in bags:
        if bag.embeddings.shape[0] == 0:
            continue

        stem = Path(bag.audio_path).stem
        prefix = "NEG_" if bool(getattr(bag, "is_negative", False)) else ""
        candidate = f"{prefix}{stem}.png"
        if candidate in used_names:
            parent_name = Path(bag.audio_path).parent.name
            candidate = f"{parent_name}_{stem}.png"
            suffix_idx = 1
            while candidate in used_names:
                candidate = f"{parent_name}_{stem}_{suffix_idx}.png"
                suffix_idx += 1
        used_names.add(candidate)

        frame_pred = _build_frame_pred_for_bag(model, bag)
        frame_true = _build_frame_true_for_bag(bag, class_names) if bag.strong_events else None
        audio_duration_seconds = float(bag.instance_windows[-1].end_sec) if bag.instance_windows else 0.0
        out_png = output_dir / candidate
        saved_path = plot_two_panel_localization(
            frame_true=frame_true,
            frame_pred=frame_pred,
            class_names=class_names,
            window_seconds=float(config.WINDOW_SECONDS),
            hop_seconds=float(config.HOP_SECONDS),
            audio_duration_seconds=audio_duration_seconds,
            out_path=str(out_png),
            colored_gt=bool(getattr(config, "COLORED_GT", False)),
        )
        if saved_path:
            _debug(f"[viz] {split_name} two-panel plot saved: {saved_path}")


def _export_eval_predictions(
    model: JointMilModel,
    val_bags: List[BagData],
    test_bags: List[BagData],
    species_codes: List[str],
    run_name: str,
    output_dir: Path,
) -> None:
    bag_rows: List[dict] = []
    segment_rows: List[dict] = []
    model.eval()
    with torch.no_grad():
        for subset_name, bags in [("validation", val_bags), ("test", test_bags)]:
            for bag in bags:
                if bag.embeddings.shape[0] == 0:
                    continue
                bag_logits, inst_logits = model(bag.embeddings)
                bag_probs = torch.sigmoid(bag_logits).detach().cpu().numpy()
                inst_probs = torch.sigmoid(inst_logits).detach().cpu().numpy()

                bag_row = {
                    "subset": subset_name,
                    "filename": Path(bag.audio_path).name,
                }
                for idx, cname in enumerate(species_codes):
                    bag_row[cname] = float(bag_probs[idx])
                bag_rows.append(bag_row)

                for i, window in enumerate(bag.instance_windows):
                    seg_row = {
                        "subset": subset_name,
                        "filename": Path(bag.audio_path).name,
                        "segment": f"{window.start_sec:g}-{window.end_sec:g}",
                    }
                    for idx, cname in enumerate(species_codes):
                        seg_row[cname] = float(inst_probs[i, idx])
                    segment_rows.append(seg_row)

    output_dir.mkdir(parents=True, exist_ok=True)
    bag_df = pd.DataFrame(bag_rows)
    seg_df = pd.DataFrame(segment_rows)
    bag_csv = output_dir / f"{run_name}_bag_level_predictions.csv"
    bag_xlsx = output_dir / f"{run_name}_bag_level_predictions.xlsx"
    seg_csv = output_dir / f"{run_name}_segment_level_predictions.csv"
    seg_xlsx = output_dir / f"{run_name}_segment_level_predictions.xlsx"
    bag_df.to_csv(bag_csv, index=False)
    seg_df.to_csv(seg_csv, index=False)
    try:
        bag_df.to_excel(bag_xlsx, index=False)
        seg_df.to_excel(seg_xlsx, index=False)
    except Exception as exc:
        print(f"[export] warning: could not write Excel prediction exports: {exc}")


def train_joint_weak_model(
    train_bags: List[BagData],
    species_codes: List[str],
    in_dim: int,
    val_bags: List[BagData] | None = None,
) -> tuple[JointMilModel, Dict[str, List[float]]]:
    model = JointMilModel(in_dim=in_dim, n_classes=len(species_codes), pooling=config.MIL_POOLING)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_name = str(getattr(config, "LOSS_NAME", "bce")).lower()
    criterion = _build_loss_fn(loss_name, species_codes, train_bags=train_bags)
    species_to_idx = build_species_index(species_codes)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_micro_f1': [], 'val_micro_f1': [],
        'train_macro_f1': [], 'val_macro_f1': [],
        'train_micro_precision': [], 'val_micro_precision': [],
        'train_macro_precision': [], 'val_macro_precision': [],
        'train_micro_recall': [], 'val_micro_recall': [],
        'train_macro_recall': [], 'val_macro_recall': [],
        'train_micro_er': [], 'val_micro_er': [],
        'train_macro_er': [], 'val_macro_er': [],
        'epoch_entropy': [],
    }

    best_micro = -1.0
    best_macro = -1.0
    epochs_without_improve = 0

    for epoch in range(config.EPOCHS):
        total_loss = 0.0
        random.shuffle(train_bags)
        model.train()

        for bag in train_bags:
            target = build_weak_target(bag.label_code, species_to_idx, len(species_codes))
            optimizer.zero_grad()
            bag_logits, _ = model(bag.embeddings)
            loss = criterion(bag_logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_metrics = _eval_bag_metrics(model, train_bags, species_codes, loss_fn=criterion)
        if val_bags:
            val_metrics = _eval_bag_metrics(
                model,
                val_bags,
                species_codes,
                use_strong_if_available=True,
                loss_fn=criterion,
            )
        else:
            val_metrics = {k: 0.0 for k in train_metrics}

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        for key in ['micro_f1','macro_f1','micro_precision','macro_precision','micro_recall','macro_recall','micro_er','macro_er']:
            history[f'train_{key}'].append(train_metrics[key])
            history[f'val_{key}'].append(val_metrics[key])

        print(
            f"Epoch {epoch + 1}: "
            f"train loss={train_metrics['loss']:.4f}, "
            f"micro-F1={train_metrics['micro_f1']:.4f}, "
            f"macro-F1={train_metrics['macro_f1']:.4f}, "
            f"ER={train_metrics['micro_er']:.4f}"
        )
        print(
            f"           val   loss={val_metrics['loss']:.4f}, "
            f"micro-F1={val_metrics['micro_f1']:.4f}, "
            f"macro-F1={val_metrics['macro_f1']:.4f}, "
            f"ER={val_metrics['micro_er']:.4f}"
        )

        _save_epoch_distribution_artifacts(
            epoch_idx=epoch + 1,
            model=model,
            bags=train_bags,
            class_names=species_codes,
            history=history,
        )

        if VERBOSE_DEBUG:
            _plot_history(history)

        improved = False
        if val_metrics['micro_f1'] > best_micro:
            best_micro = val_metrics['micro_f1']
            torch.save(model.state_dict(), 'best_micro_model.pt')
            torch.save(model.state_dict(), 'best_micro_model_segment.pt')
            improved = True
        if val_metrics['macro_f1'] > best_macro:
            best_macro = val_metrics['macro_f1']
            torch.save(model.state_dict(), 'best_macro_model.pt')
            torch.save(model.state_dict(), 'best_macro_model_segment.pt')
            improved = True

        if improved:
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if (
            bool(getattr(config, 'ENABLE_EARLY_STOPPING', False))
            and (epoch + 1) >= int(getattr(config, 'EARLY_STOPPING_MIN_EPOCHS', 1))
            and epochs_without_improve >= int(getattr(config, 'EARLY_STOPPING_PATIENCE', 20))
        ):
            print(f"Early stopping at epoch {epoch + 1} (no val bag-level improvement).")
            break

    return model, history


def initialize_class_heads_from_joint(joint_model: JointMilModel, species_codes: List[str], in_dim: int) -> Dict[str, ClassSpecificHead]:
    heads: Dict[str, ClassSpecificHead] = {}
    for idx, code in enumerate(species_codes):
        head = ClassSpecificHead(in_dim=in_dim)
        with torch.no_grad():
            head.linear.weight.copy_(joint_model.linear.weight[idx : idx + 1, :])
            head.linear.bias.copy_(joint_model.linear.bias[idx : idx + 1])
        heads[code] = head
    return heads


def infer_snippet_scores_joint(
    model: JointMilModel,
    bags: List[BagData],
    species_codes: List[str],
) -> Dict[str, List[SnippetScore]]:
    by_class: Dict[str, List[SnippetScore]] = {code: [] for code in species_codes}
    model.eval()
    with torch.no_grad():
        for bag in bags:
            inst_logits = model.forward_segment_logits(bag.embeddings)
            inst_probs = torch.sigmoid(inst_logits)
            for class_idx, class_code in enumerate(species_codes):
                for i in range(inst_probs.shape[0]):
                    window = bag.instance_windows[i]
                    by_class[class_code].append(
                        SnippetScore(
                            class_code=class_code,
                            bag_id=bag.bag_id,
                            audio_path=bag.audio_path,
                            instance_index=i,
                            start_sec=window.start_sec,
                            end_sec=window.end_sec,
                            score=float(inst_probs[i, class_idx].item()),
                        )
                    )
    return by_class


def infer_single_class_scores(head: ClassSpecificHead, class_code: str, bags: List[BagData]) -> List[SnippetScore]:
    scores: List[SnippetScore] = []
    head.eval()
    with torch.no_grad():
        for bag in bags:
            probs = torch.sigmoid(head(bag.embeddings))
            for i in range(probs.shape[0]):
                window = bag.instance_windows[i]
                scores.append(
                    SnippetScore(
                        class_code=class_code,
                        bag_id=bag.bag_id,
                        audio_path=bag.audio_path,
                        instance_index=i,
                        start_sec=window.start_sec,
                        end_sec=window.end_sec,
                        score=float(probs[i].item()),
                    )
                )
    return scores


def uncertainty_value(score: float) -> float:
    return abs(score - config.UNCERTAINTY_CENTER)


def load_annotations() -> set[tuple[str, str, int]]:
    path = Path(config.ANNOTATION_STORE)
    if not path.exists():
        return set()

    existing = set()
    with path.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            existing.add((row["class_code"], row["bag_id"], int(row["instance_index"])))
    return existing


def append_annotation(class_code: str, snippet: SnippetScore, label: int) -> None:
    path = Path(config.ANNOTATION_STORE)
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        if write_header:
            writer.writerow(
                [
                    "class_code",
                    "bag_id",
                    "audio_path",
                    "instance_index",
                    "start_sec",
                    "end_sec",
                    "model_score",
                    "strong_label",
                ]
            )
        writer.writerow(
            [
                class_code,
                snippet.bag_id,
                snippet.audio_path,
                snippet.instance_index,
                snippet.start_sec,
                snippet.end_sec,
                snippet.score,
                label,
            ]
        )


def select_uncertain_unlabeled(snippets: List[SnippetScore], seen: set[tuple[str, str, int]]) -> List[SnippetScore]:
    candidates = [s for s in snippets if (s.class_code, s.bag_id, s.instance_index) not in seen]
    candidates.sort(key=lambda s: uncertainty_value(s.score))
    return candidates[: config.UNCERTAINTY_TOP_K]


def retrain_single_class_head(
    head: ClassSpecificHead,
    class_code: str,
    annotation_rows: List[dict],
    bag_lookup: Dict[Tuple[str, int], torch.Tensor],
) -> None:
    relevant_rows = [row for row in annotation_rows if row["class_code"] == class_code]
    if not relevant_rows:
        return

    optimizer = Adam(head.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    head.train()
    for _ in range(5):
        random.shuffle(relevant_rows)
        for row in relevant_rows:
            key = (row["bag_id"], int(row["instance_index"]))
            if key not in bag_lookup:
                continue

            emb = bag_lookup[key].unsqueeze(0)
            label = torch.tensor([float(row["strong_label"])], dtype=torch.float32)
            optimizer.zero_grad()
            logits = head(emb)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()


class AudioPlaybackController:
    """Simple local audio playback controller with play/pause support."""

    def __init__(self) -> None:
        self.backend_available = importlib.util.find_spec("simpleaudio") is not None
        self._simpleaudio = importlib.import_module("simpleaudio") if self.backend_available else None
        self._play_obj = None

    def play(self, wav_path: str) -> bool:
        if not self.backend_available:
            print("[gui-demo] Playback backend unavailable: install `simpleaudio` for Play/Pause.")
            return False
        self.stop()
        wave_obj = self._simpleaudio.WaveObject.from_wave_file(wav_path)
        self._play_obj = wave_obj.play()
        return True

    def pause(self) -> bool:
        if self._play_obj is None:
            return False
        self._play_obj.stop()
        self._play_obj = None
        return True

    def stop(self) -> None:
        if self._play_obj is not None:
            self._play_obj.stop()
            self._play_obj = None


def maybe_launch_gui(
    class_heads: Dict[str, ClassSpecificHead],
    snippet_scores: Dict[str, List[SnippetScore]],
    bag_lookup: Dict[Tuple[str, int], torch.Tensor],
    train_bags: List[BagData],
) -> None:
    if not config.GUI:
        return

    tkinter_available = importlib.util.find_spec("tkinter") is not None
    matplotlib_available = importlib.util.find_spec("matplotlib") is not None
    if not tkinter_available or not matplotlib_available:
        print("[gui-demo] GUI dependencies missing: tkinter and matplotlib are required.")
        return

    import tkinter as tk
    from tkinter import ttk

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk()
    root.title("WSSED Active Learning")
    root.geometry("1200x900")

    selected_class = tk.StringVar(value=list(class_heads.keys())[0])
    countdown_var = tk.StringVar(value="")

    top_frame = ttk.Frame(root)
    top_frame.pack(fill="x", padx=8, pady=8)

    ttk.Label(top_frame, text="Species").pack(side="left")
    dropdown = ttk.Combobox(top_frame, textvariable=selected_class, values=list(class_heads.keys()), state="readonly")
    dropdown.pack(side="left", padx=8)

    plot_frame = ttk.Frame(root)
    plot_frame.pack(fill="both", expand=True, padx=8, pady=8)

    fig, (ax_hist, ax_spec) = plt.subplots(2, 1, figsize=(9, 7))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    list_frame = ttk.Frame(root)
    list_frame.pack(fill="both", expand=True, padx=8, pady=8)

    snippet_list = tk.Listbox(list_frame, height=10)
    snippet_list.pack(fill="both", expand=True)

    action_frame = ttk.Frame(root)
    action_frame.pack(fill="x", padx=8, pady=8)

    per_class_annotation_count: Dict[str, int] = {code: 0 for code in class_heads}
    current_snippets: List[SnippetScore] = []
    playback = AudioPlaybackController()
    current_clip_path: Optional[str] = None

    def update_countdown_label() -> None:
        class_code = selected_class.get()
        current_count = per_class_annotation_count[class_code]
        remaining = config.RETRAIN_EVERY_N - (current_count % config.RETRAIN_EVERY_N)
        if remaining == config.RETRAIN_EVERY_N:
            remaining = 0
        countdown_var.set(
            f"Class={class_code} | New annotations since last retrain: {current_count} | "
            f"Auto retrain in: {remaining}"
        )

    def get_selected_snippet() -> Optional[SnippetScore]:
        idx = snippet_list.curselection()
        if not idx:
            return None
        if idx[0] >= len(current_snippets):
            return None
        return current_snippets[idx[0]]

    def build_snippet_wav(snippet: SnippetScore) -> Optional[str]:
        waveform, sr = load_audio_mono(snippet.audio_path)
        start = int(snippet.start_sec * sr)
        end = int(snippet.end_sec * sr)
        clip = waveform[:, start:end].squeeze(0)
        if clip.numel() == 0:
            return None

        clip = clip / (torch.max(torch.abs(clip)) + 1e-8)
        pcm = (clip.numpy() * 32767.0).astype(np.int16)

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        with wave.open(temp_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(pcm.tobytes())
        return temp_file.name

    def render_selected_spectrogram() -> None:
        snippet = get_selected_snippet()
        if snippet is None:
            return

        waveform, sr = load_audio_mono(snippet.audio_path)
        start = int(snippet.start_sec * sr)
        end = int(snippet.end_sec * sr)
        clip = waveform[:, start:end]
        n_fft = int(config.N_FFT)
        hop_length = max(1, n_fft // 4)
        window = torch.hann_window(n_fft)
        spec = torch.log1p(
            torch.abs(
                torch.stft(
                    clip.squeeze(0),
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=n_fft,
                    window=window,
                    return_complex=True,
                )
            )
        ).numpy()

        ax_spec.clear()
        ax_spec.imshow(spec, aspect="auto", origin="lower")
        ax_spec.set_title(f"{snippet.bag_id} | {snippet.start_sec:.1f}-{snippet.end_sec:.1f}s")
        canvas.draw_idle()

    def refresh_ui() -> None:
        class_code = selected_class.get()
        seen = load_annotations()
        pool = select_uncertain_unlabeled(snippet_scores[class_code], seen)

        nonlocal current_snippets
        current_snippets = pool

        snippet_list.delete(0, tk.END)
        for item in pool:
            snippet_list.insert(
                tk.END,
                f"{item.bag_id} | idx={item.instance_index} | {item.start_sec:.1f}-{item.end_sec:.1f}s | score={item.score:.3f}",
            )

        ax_hist.clear()
        all_scores = [s.score for s in snippet_scores[class_code]]
        counts, bins, _ = ax_hist.hist(all_scores, bins=20)
        ax_hist.set_title(f"Score histogram: {class_code}")
        ax_hist.set_xlabel("Presence score")
        ax_hist.set_ylabel("Snippet count")

        ax_spec.clear()
        ax_spec.set_title("Selected snippet spectrogram")
        ax_spec.set_xlabel("Time")
        ax_spec.set_ylabel("Frequency")

        fig.canvas.mpl_disconnect(getattr(refresh_ui, "_hist_cid", None)) if hasattr(refresh_ui, "_hist_cid") else None

        def on_hist_click(event) -> None:
            if event.inaxes != ax_hist or event.xdata is None or len(current_snippets) == 0:
                return
            target_score = float(event.xdata)
            best_idx = min(range(len(current_snippets)), key=lambda idx: abs(current_snippets[idx].score - target_score))
            snippet_list.selection_clear(0, tk.END)
            snippet_list.selection_set(best_idx)
            snippet_list.activate(best_idx)
            render_selected_spectrogram()

        refresh_ui._hist_cid = fig.canvas.mpl_connect("button_press_event", on_hist_click)

        update_countdown_label()
        canvas.draw_idle()

    def play_selected() -> None:
        nonlocal current_clip_path
        snippet = get_selected_snippet()
        if snippet is None:
            print("[gui-demo] Play failed: no snippet selected.")
            return
        current_clip_path = build_snippet_wav(snippet)
        if current_clip_path is None:
            print("[gui-demo] Play failed: could not build snippet wav.")
            return
        ok = playback.play(current_clip_path)
        if ok:
            print("[gui-demo] Play OK.")

    def pause_selected() -> None:
        ok = playback.pause()
        if ok:
            print("[gui-demo] Pause OK.")
        else:
            print("[gui-demo] Pause no-op.")

    def annotate(label: int) -> None:
        snippet = get_selected_snippet()
        if snippet is None:
            return
        class_code = selected_class.get()
        append_annotation(class_code, snippet, label)
        per_class_annotation_count[class_code] += 1

        if per_class_annotation_count[class_code] % config.RETRAIN_EVERY_N == 0:
            trigger_retrain()

        refresh_ui()

    def load_annotation_rows() -> List[dict]:
        path = Path(config.ANNOTATION_STORE)
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as file_handle:
            return list(csv.DictReader(file_handle))

    def trigger_retrain() -> None:
        class_code = selected_class.get()
        rows = load_annotation_rows()
        retrain_single_class_head(class_heads[class_code], class_code, rows, bag_lookup)
        snippet_scores[class_code] = infer_single_class_scores(class_heads[class_code], class_code, train_bags)
        per_class_annotation_count[class_code] = 0
        print(f"[gui-demo] Retrain completed for class={class_code}.")
        refresh_ui()

    def run_gui_demo_flow() -> None:
        print("[gui-demo] Demo flow start.")
        refresh_ui()
        if len(current_snippets) == 0:
            print("[gui-demo] FAIL: No snippets available for demo flow.")
            return

        snippet_list.selection_clear(0, tk.END)
        snippet_list.selection_set(0)
        snippet_list.activate(0)
        render_selected_spectrogram()
        print("[gui-demo] PASS: Snippet select + spectrogram render.")

        play_selected()
        pause_selected()

        before_lines = 0
        annotation_path = Path(config.ANNOTATION_STORE)
        if annotation_path.exists():
            before_lines = len(annotation_path.read_text(encoding="utf-8").splitlines())
        annotate(1)
        after_lines = 0
        if annotation_path.exists():
            after_lines = len(annotation_path.read_text(encoding="utf-8").splitlines())

        if after_lines > before_lines:
            print("[gui-demo] PASS: Annotation record written.")
        else:
            print("[gui-demo] FAIL: Annotation record was not written.")

        print("[gui-demo] INFO: Countdown updated after annotation.")
        trigger_retrain()
        print("[gui-demo] PASS: Manual retrain trigger executed.")
        print("[gui-demo] PASS: Scores and histogram refresh executed.")

    ttk.Button(action_frame, text="Play", command=play_selected).pack(side="left", padx=4)
    ttk.Button(action_frame, text="Pause", command=pause_selected).pack(side="left", padx=4)
    ttk.Button(action_frame, text="Presence", command=lambda: annotate(1)).pack(side="left", padx=4)
    ttk.Button(action_frame, text="Absence", command=lambda: annotate(0)).pack(side="left", padx=4)
    ttk.Button(action_frame, text="Retrain now", command=trigger_retrain).pack(side="left", padx=4)
    ttk.Label(action_frame, textvariable=countdown_var).pack(side="left", padx=12)

    dropdown.bind("<<ComboboxSelected>>", lambda _evt: refresh_ui())
    snippet_list.bind("<<ListboxSelect>>", lambda _evt: render_selected_spectrogram())

    run_gui_demo_flow()
    root.mainloop()


def _ensure_embedding_cache_for_wav_lists(split_to_wavs: Dict[str, List[str]]) -> None:
    if config.EMBEDDING_SOURCE != "birdnet_analyzer":
        return

    cache_dir = Path(config.BIRDNET_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split_name, wavs in split_to_wavs.items():
        hit = 0
        miss = 0
        seen = set()
        for wav in wavs:
            if wav in seen:
                continue
            seen.add(wav)
            cache_path = embedding_cache_path(wav, config.BIRDNET_CACHE_DIR)
            if cache_path.exists():
                try:
                    arr = np.load(cache_path)["embeddings"]
                    _debug(f"CACHE_HIT split={split_name} file={wav} shape={arr.shape}")
                except Exception:
                    _debug(f"CACHE_HIT split={split_name} file={wav} shape=<unknown>")
                hit += 1
                continue

            miss += 1
            if not config.ALLOW_AUTO_EXTRACT:
                raise RuntimeError(f"Missing cache for split={split_name} file={wav} and ALLOW_AUTO_EXTRACT=False")

            cmd = [
                "python",
                "extract_embeddings.py",
                "--single-wav",
                str(wav),
                "--output-dir",
                str(config.BIRDNET_CACHE_DIR),
            ]
            _debug(f"CACHE_MISS split={split_name} file={wav} -> extracting via: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Auto extraction failed for split={split_name} file={wav}.\n"
                    f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                )
        summary[split_name] = (hit, miss)

    _debug(
        "CACHE_SUMMARY "
        + " | ".join(
            f"{k}: hit={v[0]} miss={v[1]}" for k, v in summary.items()
        )
    )




def _collect_summary_rows(label: str, level: str, metrics: Dict[str, float], species_codes: List[str]) -> List[Tuple[str, float]]:
    rows: List[Tuple[str, float]] = []
    for k in ["micro_precision", "micro_recall", "micro_f1", "micro_er", "macro_precision", "macro_recall", "macro_f1", "macro_er"]:
        rows.append((f"{label}/{level}/overall/{k}", float(metrics.get(k, 0.0))))
    per_class = metrics.get("per_class", {})
    for cname in species_codes:
        row = per_class.get(cname, {})
        for k in ["micro_precision", "micro_recall", "micro_f1", "micro_er"]:
            rows.append((f"{label}/{level}/{cname}/{k}", float(row.get(k, 0.0))))
    return rows


def run_pipeline(bundle: DatasetBundle) -> List[Tuple[str, float]]:
    _debug("Preparing split-selected sources...")
    train_source = bundle.train_df
    val_source = bundle.val_df
    test_source = bundle.test_df

    if config.SMOKE_MODE:
        train_source = train_source.head(config.SMOKE_MAX_BAGS).copy()
        val_source = val_source.head(config.SMOKE_MAX_BAGS).copy()
        test_source = test_source.head(config.SMOKE_MAX_BAGS).copy()
        _debug(f"[smoke] enabled: max_bags={config.SMOKE_MAX_BAGS}")
    run_name = f"original({len(train_source)})_synthetic(0)_BirdNET_{str(config.LOSS_NAME)}"

    split_to_wavs = {
        "train": train_source["audio_path"].dropna().astype(str).drop_duplicates().tolist() if len(train_source) else [],
        "val": val_source["audio_path"].dropna().astype(str).drop_duplicates().tolist() if len(val_source) else [],
        "test": test_source["audio_path"].dropna().astype(str).drop_duplicates().tolist() if len(test_source) else [],
    }
    _ensure_embedding_cache_for_wav_lists(split_to_wavs)

    _debug("Building training bags...")
    train_bags = build_bags(train_source)
    if not train_bags:
        raise RuntimeError("No train bags were built. Check dataset paths and metadata filtering.")

    in_dim = int(train_bags[0].embeddings.shape[1])
    for bag in train_bags:
        if int(bag.embeddings.shape[1]) != in_dim:
            raise RuntimeError(
                f"Inconsistent embedding dim across bags: expected {in_dim}, got {bag.embeddings.shape[1]} for {bag.audio_path}"
            )
    _debug(f"[birdnet] BirdNET embeddings used, in_dim={in_dim}")
    if config.EMBEDDING_SOURCE == "birdnet_analyzer" and int(in_dim) != 1024:
        raise RuntimeError(
            f"FAIL: EMBEDDING_SOURCE=birdnet_analyzer requires in_dim==1024, got {in_dim}"
        )

    _debug("Building validation bags for evaluation-only (no optimizer updates)...")
    val_bags = build_bags(val_source) if len(val_source) > 0 else []

    _debug(f"Training a joint weak MIL model for {len(bundle.species_codes)} classes...")
    original_epochs = config.EPOCHS
    try:
        if config.SMOKE_MODE:
            config.EPOCHS = config.SMOKE_EPOCHS
            _debug(f"[smoke] epochs override: {config.EPOCHS}")
        joint_model, history = train_joint_weak_model(
            train_bags,
            bundle.species_codes,
            in_dim=in_dim,
            val_bags=val_bags,
        )
    finally:
        config.EPOCHS = original_epochs

    test_bags = build_bags(test_source) if len(test_source) > 0 else []
    ckpt_specs = [
        ("best_micro_model.pt", "best_micro_model_segment.pt", "best_micro_model"),
        ("best_macro_model.pt", "best_macro_model_segment.pt", "best_macro_model"),
    ]
    seed_rows: List[Tuple[str, float]] = []
    if test_bags:
        has_segment_gt = any(bool(bag.strong_events) for bag in test_bags)
        for bag_ckpt_path, segment_ckpt_path, ckpt_label in ckpt_specs:
            if not Path(bag_ckpt_path).exists():
                continue

            bag_model = JointMilModel(in_dim=in_dim, n_classes=len(bundle.species_codes), pooling=config.MIL_POOLING)
            bag_model.load_state_dict(torch.load(bag_ckpt_path, map_location="cpu"))
            bag_metrics = _eval_bag_metrics(
                bag_model,
                test_bags,
                bundle.species_codes,
                use_strong_if_available=True,
            )
            print(f"[{ckpt_label}] TEST Bag-level")
            print(_format_metric_block(bag_metrics))
            seed_rows.extend(_collect_summary_rows(ckpt_label, "bag", bag_metrics, bundle.species_codes))
            for cname in bundle.species_codes:
                row = bag_metrics.get("per_class", {}).get(cname, {})
                print(f"{cname} - {_format_metric_block(row, include_macro=False)}")

            if has_segment_gt:
                segment_load_path = segment_ckpt_path if Path(segment_ckpt_path).exists() else bag_ckpt_path
                segment_model = JointMilModel(in_dim=in_dim, n_classes=len(bundle.species_codes), pooling=config.MIL_POOLING)
                segment_model.load_state_dict(torch.load(segment_load_path, map_location="cpu"))
                segment_metrics = _eval_snippet_metrics(
                    segment_model,
                    test_bags,
                    bundle.species_codes,
                    use_strong_if_available=True,
                )
                print(f"[{ckpt_label}] TEST Segment-level")
                print(_format_metric_block(segment_metrics))
                seed_rows.extend(_collect_summary_rows(ckpt_label, "segment", segment_metrics, bundle.species_codes))
                for cname in bundle.species_codes:
                    row = segment_metrics.get("per_class", {}).get(cname, {})
                    print(f"{cname} - {_format_metric_block(row, include_macro=False)}")
            else:
                print(f"[{ckpt_label}] TEST Segment-level: SKIP (no GT)")

    viz_ckpt = "best_micro_model_segment.pt" if Path("best_micro_model_segment.pt").exists() else (
        "best_micro_model.pt" if Path("best_micro_model.pt").exists() else None
    )
    if viz_ckpt is not None:
        viz_model = JointMilModel(in_dim=in_dim, n_classes=len(bundle.species_codes), pooling=config.MIL_POOLING)
        viz_model.load_state_dict(torch.load(viz_ckpt, map_location="cpu"))
        _save_split_visualizations(
            split_name="validation",
            bags=val_bags,
            model=viz_model,
            class_names=bundle.species_codes,
            output_dir=Path("outputs") / "validation",
        )
        _save_split_visualizations(
            split_name="test",
            bags=test_bags,
            model=viz_model,
            class_names=bundle.species_codes,
            output_dir=Path("outputs") / "test",
        )
        _export_eval_predictions(
            model=viz_model,
            val_bags=val_bags,
            test_bags=test_bags,
            species_codes=bundle.species_codes,
            run_name=run_name,
            output_dir=Path("outputs"),
        )
    else:
        _export_eval_predictions(
            model=joint_model,
            val_bags=val_bags,
            test_bags=test_bags,
            species_codes=bundle.species_codes,
            run_name=run_name,
            output_dir=Path("outputs"),
        )

    _debug("Running snippet-level inference from the joint model...")
    snippet_scores = infer_snippet_scores_joint(joint_model, train_bags, bundle.species_codes)

    _debug("Initializing per-class strong supervision heads from joint model weights...")
    class_heads = initialize_class_heads_from_joint(joint_model, bundle.species_codes, in_dim=in_dim)

    bag_lookup: Dict[Tuple[str, int], torch.Tensor] = {}
    for bag in train_bags:
        for idx in range(bag.embeddings.shape[0]):
            bag_lookup[(bag.bag_id, idx)] = bag.embeddings[idx]

    maybe_launch_gui(class_heads, snippet_scores, bag_lookup, train_bags)
    return seed_rows


def main() -> None:
    config.SEED = 42
    seed_everything(config.SEED)
    print_config()

    bundle = load_datasets()
    if VERBOSE_DEBUG:
        summary = summarize_bundle(bundle)
        print("Dataset summary:")
        for key, value in summary.items():
            print(f"  - {key}: {value}")

    run_pipeline(bundle)


if __name__ == "__main__":
    main()
