import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---- dynamic config resolution -----------------------------------------------
# The GPU server sets WSSED_CONFIG=config_<job_id> in the subprocess environment
# so each training job uses its own generated hyperparameter file instead of the
# repo-default config.py.  Inserting the module into sys.modules["config"] ensures
# that subsequent `import config` statements in DataHandler.py and other files
# all see the same per-job module.
import importlib.util as _ilu
_wssed_cfg = os.environ.get("WSSED_CONFIG", "config")
if _wssed_cfg != "config" and "config" not in sys.modules:
    _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{_wssed_cfg}.py")
    _spec = _ilu.spec_from_file_location("config", _cfg_path)
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["config"] = _mod
    _spec.loader.exec_module(_mod)
del _ilu, _wssed_cfg
# ---- end dynamic config resolution -------------------------------------------

import config
from MainClasses.Models import Baseline, CDur, TALNet
from MainClasses.MILPooling import MILPooling
from MainClasses.loc_vad import activity_detection


@dataclass
class AudioSegment:
    audio_path: str
    start_sec: float
    end_sec: float
    label: np.ndarray
    segment_id: str
    duration_sec: Optional[float] = None


class AudioSegmentDataset(Dataset):
    def __init__(
        self,
        segments: List[AudioSegment],
        sample_rate: int,
        n_mels: int,
        n_fft: int,
        hop_length: int,
        segment_seconds: int,
        fixed_frames: Optional[int] = None,
    ):
        self.segments = segments
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_seconds = segment_seconds
        if fixed_frames is not None:
            self.seq_len = fixed_frames
        else:
            self.seq_len = int(np.floor((segment_seconds * sample_rate - n_fft) / hop_length) + 1)

    def __len__(self) -> int:
        return len(self.segments)

    def _load_segment(self, audio_path: str, start_sec: float, end_sec: float) -> np.ndarray:
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        start_sample = int(start_sec * self.sample_rate)
        end_sample = int(end_sec * self.sample_rate)
        segment = audio[start_sample:end_sample]
        target_len = int((end_sec - start_sec) * self.sample_rate)
        if len(segment) < target_len:
            padding = np.zeros(target_len - len(segment), dtype=segment.dtype)
            segment = np.concatenate([segment, padding])
        return segment

    def _extract_logmel(self, audio: np.ndarray) -> np.ndarray:
        melspec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            win_length=self.n_fft,
            center=False,
            window="hamming",
        )
        logmelspec = librosa.power_to_db(melspec)
        return logmelspec.T

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, int, str, float, float]:
        seg = self.segments[idx]
        audio = self._load_segment(seg.audio_path, seg.start_sec, seg.end_sec)
        feats = self._extract_logmel(audio)
        frames_len = feats.shape[0]
        return (
            torch.from_numpy(feats).float(),
            torch.from_numpy(seg.label).float(),
            seg.segment_id,
            frames_len,
            seg.audio_path,
            seg.start_sec,
            seg.end_sec,
        )


class CNNBiGRU(nn.Module):
    def __init__(self, n_classes: int, pool_style: str, seq_len: int, n_mels: int = 64):
        super().__init__()
        self.pool_style = pool_style
        self.pool = MILPooling(n_classes=n_classes, seq_len=seq_len).get_pool(pool_style)
        self.name = f"cnn_bigru_{pool_style}"
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.gru = nn.GRU(
            input_size=(n_mels // 4) * 64,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.out = nn.Linear(256, n_classes)

    def forward(
        self, inputs: torch.Tensor, upsample: bool = False, mask: Optional[torch.Tensor] = None, **_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = inputs.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).flatten(2)
        x, _ = self.gru(x)
        y_frames = torch.sigmoid(self.out(x)).clamp(1e-7, 1.0)
        y_clip = pool_with_mask(self.pool, y_frames, mask)
        return y_clip, y_frames


class CNNTransformer(nn.Module):
    def __init__(self, n_classes: int, pool_style: str, seq_len: int, n_mels: int = 64):
        super().__init__()
        self.pool_style = pool_style
        self.pool = MILPooling(n_classes=n_classes, seq_len=seq_len).get_pool(pool_style)
        self.name = f"cnn_transformer_{pool_style}"
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=(n_mels // 4) * 64,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = nn.Linear((n_mels // 4) * 64, n_classes)

    def forward(
        self, inputs: torch.Tensor, upsample: bool = False, mask: Optional[torch.Tensor] = None, **_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = inputs.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(2)
        x = self.transformer(x)
        y_frames = torch.sigmoid(self.out(x)).clamp(1e-7, 1.0)
        y_clip = pool_with_mask(self.pool, y_frames, mask)
        return y_clip, y_frames


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_frames(duration_sec: float, sample_rate: int, n_fft: int, hop_length: int) -> int:
    frames = int(np.floor((duration_sec * sample_rate - n_fft) / hop_length) + 1)
    return max(frames, 1)


def pool_with_mask(pool, y_frames, mask=None):
    if mask is None:
        return pool(y_frames)
    try:
        return pool(y_frames, mask)
    except TypeError:
        return pool(y_frames)


def pool_requires_fixed_seq_len(pool_style: str) -> bool:
    return pool_style in {"attention_pool", "hi_pool", "hi_pool_plus", "hi_pool_fixed"}


def collate_default(batch):
    feats, labels, ids, _lengths, paths, starts, ends = zip(*batch)
    return torch.stack(feats), torch.stack(labels), list(ids), None, list(paths), list(starts), list(ends)


def collate_full_bag_pad(batch, fixed_frames: int, pad_mode: str):
    feats, labels, ids, lengths, paths, starts, ends = zip(*batch)
    padded = []
    masks = []
    for feat, length in zip(feats, lengths):
        if length >= fixed_frames:
            feat_pad = feat[:fixed_frames]
            mask = torch.ones(fixed_frames, dtype=torch.float32)
        else:
            if pad_mode == "repeat":
                reps = fixed_frames // length
                remainder = fixed_frames % length
                tiles = [feat] * reps
                if remainder > 0:
                    tiles.append(feat[:remainder])
                feat_pad = torch.cat(tiles, dim=0)
                mask = None
            else:
                pad_len = fixed_frames - length
                pad = torch.zeros((pad_len, feat.shape[1]), dtype=feat.dtype)
                feat_pad = torch.cat([feat, pad], dim=0)
                mask = torch.cat([torch.ones(length), torch.zeros(pad_len)])
        padded.append(feat_pad)
        masks.append(mask)

    x = torch.stack(padded)
    y = torch.stack(labels)
    if pad_mode == "silence":
        mask = torch.stack(masks)
    else:
        mask = None
    return x, y, list(ids), mask, list(paths), list(starts), list(ends)


def get_max_duration(segments: List[AudioSegment]) -> float:
    if not segments:
        return 0.0
    return max(
        seg.duration_sec if seg.duration_sec is not None else (seg.end_sec - seg.start_sec)
        for seg in segments
    )


def load_strong_labels(csv_path: str, allowed_levels: List[str]) -> Dict[str, List[Dict[str, object]]]:
    df = pd.read_csv(csv_path)
    if allowed_levels:
        df = df[df["level"].isin(allowed_levels)]
    df["file_stem"] = df["file_name"].str.replace(".txt", "", regex=False)
    events: Dict[str, List[Dict[str, object]]] = {}
    for _, row in df.iterrows():
        events.setdefault(row["file_stem"], []).append(
            {
                "start": float(row["start_second"]),
                "end": float(row["end_second"]),
                "label": str(row["label"]),
            }
        )
    return events


def strong_label_vector(
    events: List[Dict[str, object]],
    class_columns: List[str],
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    label_vec = np.zeros(len(class_columns), dtype=np.float32)
    if not events:
        return label_vec
    for event in events:
        if event["end"] > start_sec and event["start"] < end_sec:
            if event["label"] in class_columns:
                idx = class_columns.index(event["label"])
                label_vec[idx] = 1.0
    return label_vec


def anuraset_file_stem(audio_path: str) -> str:
    return os.path.splitext(os.path.basename(audio_path))[0]
def get_class_columns(metadata: pd.DataFrame, target_species: List[str]) -> List[str]:
    start_idx = metadata.columns.get_loc("subset") + 1
    all_species = list(metadata.columns[start_idx:])
    if not target_species:
        return all_species
    return [col for col in all_species if col in target_species]


def build_anuraset_segments(
    root_path: str,
    metadata: pd.DataFrame,
    class_columns: List[str],
    segment_seconds: int,
    overlap_bags: bool,
    hop_seconds: int,
    full_bag: bool,
    subset: str,
) -> List[AudioSegment]:
    subset_meta = metadata[metadata["subset"] == subset]
    segments: List[AudioSegment] = []
    grouped = subset_meta.groupby(["site", "fname"])
    for (site, fname), group in grouped:
        file_label = group[class_columns].max().values.astype(np.float32)
        audio_path = os.path.join(root_path, site, f"{fname}.wav")
        duration = 60
        if full_bag:
            segments.append(
                AudioSegment(
                    audio_path=audio_path,
                    start_sec=0,
                    end_sec=duration,
                    label=file_label,
                    segment_id=f"{fname}_full",
                    duration_sec=duration,
                )
            )
        else:
            step = hop_seconds if overlap_bags else segment_seconds
            start_sec = 0
            while start_sec + segment_seconds <= duration:
                end_sec = start_sec + segment_seconds
                label_rows = group[(group["min_t"] >= start_sec) & (group["max_t"] <= end_sec)]
                if label_rows.empty:
                    label = np.zeros(len(class_columns), dtype=np.float32)
                else:
                    label = label_rows[class_columns].max().values.astype(np.float32)
                segment_id = f"{fname}_{start_sec:02d}_{end_sec:02d}"
                segments.append(
                    AudioSegment(
                        audio_path=audio_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        label=label,
                        segment_id=segment_id,
                        duration_sec=segment_seconds,
                    )
                )
                start_sec += step
    return segments


def build_fnjv_segments(
    root_path: str,
    class_columns: List[str],
    segment_seconds: int,
    overlap_bags: bool,
    hop_seconds: int,
    full_bag: bool,
) -> Tuple[List[AudioSegment], List[str]]:
    metadata_path = os.path.join(root_path, "metadata_filtered_filled.csv")
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata["Code"].ne("IGNORE")]
    file_to_codes: Dict[str, set] = {}
    for _, row in metadata.iterrows():
        fname = row["Arquivo do registro"]
        code = row["Code"]
        file_to_codes.setdefault(fname, set()).add(code)

    segments: List[AudioSegment] = []
    codes = sorted({code for codes in file_to_codes.values() for code in codes})
    class_columns = [col for col in class_columns if col in codes]
    for fname, codes_for_file in file_to_codes.items():
        audio_path = os.path.join(root_path, fname)
        duration = librosa.get_duration(path=audio_path)
        label = np.array([1.0 if col in codes_for_file else 0.0 for col in class_columns], dtype=np.float32)
        if full_bag:
            segments.append(
                AudioSegment(
                    audio_path=audio_path,
                    start_sec=0,
                    end_sec=duration,
                    label=label,
                    segment_id=f"{os.path.splitext(fname)[0]}_full",
                    duration_sec=duration,
                )
            )
        else:
            step = hop_seconds if overlap_bags else segment_seconds
            start_sec = 0
            if duration < segment_seconds:
                end_sec = segment_seconds
                segment_id = f"{os.path.splitext(fname)[0]}_{start_sec:02d}_{end_sec:02d}"
                segments.append(
                    AudioSegment(
                        audio_path=audio_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        label=label,
                        segment_id=segment_id,
                        duration_sec=segment_seconds,
                    )
                )
            else:
                while start_sec + segment_seconds <= duration:
                    end_sec = start_sec + segment_seconds
                    segment_id = f"{os.path.splitext(fname)[0]}_{start_sec:02d}_{end_sec:02d}"
                    segments.append(
                        AudioSegment(
                            audio_path=audio_path,
                            start_sec=start_sec,
                            end_sec=end_sec,
                            label=label,
                            segment_id=segment_id,
                            duration_sec=segment_seconds,
                        )
                    )
                    start_sec += step
    return segments, codes


def split_segments(
    segments: List[AudioSegment],
    validation_split: float,
    test_split: float,
    seed: int,
) -> Tuple[List[AudioSegment], List[AudioSegment], List[AudioSegment]]:
    random.Random(seed).shuffle(segments)
    total = len(segments)
    val_count = int(total * validation_split)
    test_count = int(total * test_split)
    val_segments = segments[:val_count]
    test_segments = segments[val_count:val_count + test_count]
    train_segments = segments[val_count + test_count:]
    return train_segments, val_segments, test_segments


def select_target_segments(
    segments: List[AudioSegment],
    negative_split: float,
    seed: int,
    include_negatives: bool,
) -> Tuple[List[AudioSegment], List[AudioSegment]]:
    target_segments = [segment for segment in segments if np.any(segment.label)]
    negative_segments = [segment for segment in segments if not np.any(segment.label)]
    if include_negatives and negative_split > 0:
        random.Random(seed).shuffle(negative_segments)
        extra_count = int(len(negative_segments) * negative_split)
        extra_segments = negative_segments[:extra_count]
        remaining_negatives = negative_segments[extra_count:]
    else:
        extra_segments = []
        remaining_negatives = negative_segments
    return target_segments + extra_segments, remaining_negatives


def compute_error_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Tuple[float, float]:
    y_hat = (y_pred >= threshold).astype(np.int32)
    tp = (y_hat * y_true).sum(axis=0)
    fp = (y_hat * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_hat) * y_true).sum(axis=0)
    er_per_class = []
    for k in range(y_true.shape[1]):
        s = min(tp[k] + fn[k], tp[k] + fp[k]) - tp[k]
        d = max(0.0, fn[k] - fp[k])
        i = max(0.0, fp[k] - fn[k])
        er = (s + d + i) / (tp[k] + fn[k] + 1e-10)
        er_per_class.append(er)
    macro_er = float(np.mean(er_per_class))
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    s = min(total_tp + total_fn, total_tp + total_fp) - total_tp
    d = max(0.0, total_fn - total_fp)
    i = max(0.0, total_fp - total_fn)
    micro_er = float((s + d + i) / (total_tp + total_fn + 1e-10))
    return micro_er, macro_er


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> Dict[str, float]:
    y_hat = (y_pred >= threshold).astype(np.int32)
    tp = (y_hat * y_true).sum(axis=0)
    fp = (y_hat * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_hat) * y_true).sum(axis=0)
    precision_per = tp / (tp + fp + 1e-10)
    recall_per = tp / (tp + fn + 1e-10)
    f1_per = 2 * precision_per * recall_per / (precision_per + recall_per + 1e-10)

    macro_precision = float(np.mean(precision_per))
    macro_recall = float(np.mean(recall_per))
    macro_f1 = float(np.mean(f1_per))

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_precision = float(total_tp / (total_tp + total_fp + 1e-10))
    micro_recall = float(total_tp / (total_tp + total_fn + 1e-10))
    micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10))

    micro_er, macro_er = compute_error_rate(y_true, y_pred, threshold)

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_er": micro_er,
        "macro_er": macro_er,
    }


def compute_class_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> np.ndarray:
    y_hat = (y_pred >= threshold).astype(np.int32)
    tp = (y_hat * y_true).sum(axis=0)
    fp = (y_hat * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_hat) * y_true).sum(axis=0)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1


def compute_class_f1_binary(y_true: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    tp = (y_hat * y_true).sum(axis=0)
    fp = (y_hat * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_hat) * y_true).sum(axis=0)
    return 2 * tp / (2 * tp + fp + fn + 1e-10)


def compute_macro_f1_binary(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    class_f1 = compute_class_f1_binary(y_true, y_hat)
    return float(np.mean(class_f1)) if class_f1.size else 0.0


def apply_tagging_thresholds(
    clip_out: np.ndarray,
    thresholds: Dict[str, np.ndarray],
    system: str,
    per_class_systems: Optional[List[str]] = None,
) -> np.ndarray:
    if per_class_systems is None:
        if system == "single":
            return (clip_out >= thresholds["tag_threshold"]).astype(np.float32)
        low = thresholds["tag_threshold_low"]
        high = thresholds["tag_threshold_high"]
        mid = (low + high) / 2.0
        return (clip_out >= mid).astype(np.float32)
    y_hat = np.zeros_like(clip_out, dtype=np.float32)
    for k, mode in enumerate(per_class_systems):
        if mode == "double":
            low = thresholds["tag_threshold_low"][k]
            high = thresholds["tag_threshold_high"][k]
            mid = (low + high) / 2.0
            y_hat[:, k] = (clip_out[:, k] >= mid).astype(np.float32)
        else:
            y_hat[:, k] = (clip_out[:, k] >= thresholds["tag_threshold"][k]).astype(np.float32)
    return y_hat


def apply_localization_thresholds(
    frame_out: np.ndarray,
    class_columns: List[str],
    thresholds: Dict[str, np.ndarray],
    system: str,
    per_class_systems: Optional[List[str]] = None,
) -> np.ndarray:
    if per_class_systems is None:
        if system == "single":
            return (frame_out >= thresholds["loc_threshold"]).astype(np.float32)
        n_frames, n_classes = frame_out.shape
        binary = np.zeros((n_frames, n_classes), dtype=np.float32)
        for k in range(n_classes):
            pairs = activity_detection(
                x=frame_out[:, k],
                thres=float(thresholds["loc_threshold_high"][k]),
                low_thres=float(thresholds["loc_threshold_low"][k]),
                n_smooth=thresholds["smooth"],
                n_salt=thresholds["smooth"],
            )
            for onset, offset in pairs:
                onset = max(0, min(n_frames, int(onset)))
                offset = max(0, min(n_frames, int(offset)))
                if offset > onset:
                    binary[onset:offset, k] = 1.0
        return binary
    n_frames, n_classes = frame_out.shape
    binary = np.zeros((n_frames, n_classes), dtype=np.float32)
    for k in range(n_classes):
        if per_class_systems[k] == "double":
            pairs = activity_detection(
                x=frame_out[:, k],
                thres=float(thresholds["loc_threshold_high"][k]),
                low_thres=float(thresholds["loc_threshold_low"][k]),
                n_smooth=thresholds["smooth"],
                n_salt=thresholds["smooth"],
            )
            for onset, offset in pairs:
                onset = max(0, min(n_frames, int(onset)))
                offset = max(0, min(n_frames, int(offset)))
                if offset > onset:
                    binary[onset:offset, k] = 1.0
        else:
            binary[:, k] = (frame_out[:, k] >= thresholds["loc_threshold"][k]).astype(np.float32)
    return binary


def cache_model_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    strong_events: Optional[Dict[str, List[Dict[str, object]]]],
    class_columns: List[str],
    localization_mode: str,
    block_seconds: float,
    frame_duration: float,
    pool_style: str,
) -> Dict[str, np.ndarray]:
    model.eval()
    clip_preds = []
    clip_targets = []
    frame_preds = []
    frame_targets = []
    with torch.no_grad():
        for x_data, y_data, _ids, mask, paths, starts, ends in loader:
            x_data = x_data.to(device)
            if mask is not None:
                mask = mask.to(device)
                clip_out, frame_out = model(x_data, mask=mask)
            else:
                clip_out, frame_out = model(x_data)
            clip_out_np = clip_out.cpu().numpy()
            frame_out_np = frame_out.cpu().numpy()
            mask_np = mask.cpu().numpy() if mask is not None else None
            clip_preds.append(clip_out_np)
            clip_targets.append(y_data.cpu().numpy())
            if strong_events is None:
                continue
            for i in range(len(paths)):
                file_stem = anuraset_file_stem(paths[i])
                events = strong_events.get(file_stem, [])
                frames_len = frame_out_np[i].shape[0]
                if mask_np is not None:
                    frames_len = int(mask_np[i].sum())
                    frames_len = max(frames_len, 1)
                    frame_pred = frame_out_np[i][:frames_len]
                else:
                    frame_pred = frame_out_np[i]
                frame_true = localization_frame_labels(
                    events,
                    class_columns,
                    float(starts[i]),
                    float(ends[i]),
                    frames_len,
                )
                if localization_mode == "block":
                    block_frames = max(1, int(round(block_seconds / frame_duration)))
                    frame_pred = blockify_frames(frame_pred, block_frames, pool_style)
                    frame_true = blockify_binary_labels(frame_true, block_frames)
                frame_preds.append(frame_pred)
                frame_targets.append(frame_true)
    clip_out_all = np.concatenate(clip_preds, axis=0) if clip_preds else np.zeros((0, 0))
    clip_true_all = np.concatenate(clip_targets, axis=0) if clip_targets else np.zeros((0, 0))
    frame_out_all = np.concatenate(frame_preds, axis=0) if frame_preds else np.zeros((0, 0))
    frame_true_all = np.concatenate(frame_targets, axis=0) if frame_targets else np.zeros((0, 0))
    return {
        "clip_out": clip_out_all,
        "clip_true": clip_true_all,
        "frame_out": frame_out_all,
        "frame_true": frame_true_all,
    }


def iterative_threshold_optimization(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Dict[str, np.ndarray],
    system: str,
    step: float,
    max_rounds: int,
    objective: str,
    class_columns: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], float]:
    best_thresholds = {k: np.array(v, copy=True) for k, v in thresholds.items()}
    if objective == "tagging":
        y_hat = apply_tagging_thresholds(y_pred, best_thresholds, system)
    else:
        y_hat = apply_localization_thresholds(y_pred, class_columns or [], best_thresholds, system)
    best_score = compute_macro_f1_binary(y_true, y_hat)

    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for idx in range(y_true.shape[1]):
            if system == "single":
                key = "tag_threshold" if objective == "tagging" else "loc_threshold"
                current = best_thresholds[key][idx]
                for delta in (-step, step):
                    value = float(np.clip(current + delta, 0.0, 1.0))
                    candidate_thresholds = {k: np.array(v, copy=True) for k, v in best_thresholds.items()}
                    candidate_thresholds[key][idx] = value
                    if objective == "tagging":
                        y_hat = apply_tagging_thresholds(y_pred, candidate_thresholds, system)
                    else:
                        y_hat = apply_localization_thresholds(y_pred, class_columns or [], candidate_thresholds, system)
                    score = compute_macro_f1_binary(y_true, y_hat)
                    if score > best_score:
                        best_score = score
                        best_thresholds = candidate_thresholds
                        improved = True
                        current = value
            else:
                low_key = "tag_threshold_low" if objective == "tagging" else "loc_threshold_low"
                high_key = "tag_threshold_high" if objective == "tagging" else "loc_threshold_high"
                low_current = best_thresholds[low_key][idx]
                high_current = best_thresholds[high_key][idx]
                for delta in (-step, step):
                    low_value = float(np.clip(low_current + delta, 0.0, 1.0))
                    if low_value <= high_current:
                        candidate_thresholds = {k: np.array(v, copy=True) for k, v in best_thresholds.items()}
                        candidate_thresholds[low_key][idx] = low_value
                        candidate_thresholds[high_key][idx] = high_current
                        if objective == "tagging":
                            y_hat = apply_tagging_thresholds(y_pred, candidate_thresholds, system)
                        else:
                            y_hat = apply_localization_thresholds(
                                y_pred, class_columns or [], candidate_thresholds, system
                            )
                        score = compute_macro_f1_binary(y_true, y_hat)
                        if score > best_score:
                            best_score = score
                            best_thresholds = candidate_thresholds
                            improved = True
                            low_current = low_value
                    high_value = float(np.clip(high_current + delta, 0.0, 1.0))
                    if low_current <= high_value:
                        candidate_thresholds = {k: np.array(v, copy=True) for k, v in best_thresholds.items()}
                        candidate_thresholds[low_key][idx] = low_current
                        candidate_thresholds[high_key][idx] = high_value
                        if objective == "tagging":
                            y_hat = apply_tagging_thresholds(y_pred, candidate_thresholds, system)
                        else:
                            y_hat = apply_localization_thresholds(
                                y_pred, class_columns or [], candidate_thresholds, system
                            )
                        score = compute_macro_f1_binary(y_true, y_hat)
                        if score > best_score:
                            best_score = score
                            best_thresholds = candidate_thresholds
                            improved = True
                            high_current = high_value
    return best_thresholds, best_score


def localization_frame_labels(
    events: List[Dict[str, object]],
    class_columns: List[str],
    start_sec: float,
    end_sec: float,
    frame_count: int,
) -> np.ndarray:
    labels = np.zeros((frame_count, len(class_columns)), dtype=np.float32)
    if not events or frame_count <= 0:
        return labels
    duration = max(end_sec - start_sec, 1e-6)
    frames_per_sec = frame_count / duration
    for event in events:
        if event["end"] <= start_sec or event["start"] >= end_sec:
            continue
        if event["label"] not in class_columns:
            continue
        class_idx = class_columns.index(event["label"])
        onset = max(event["start"], start_sec) - start_sec
        offset = min(event["end"], end_sec) - start_sec
        start_frame = int(np.floor(onset * frames_per_sec))
        end_frame = int(np.ceil(offset * frames_per_sec))
        start_frame = max(0, min(frame_count, start_frame))
        end_frame = max(0, min(frame_count, end_frame))
        if end_frame > start_frame:
            labels[start_frame:end_frame, class_idx] = 1.0
    return labels


def blockify_frames(
    frame_scores: np.ndarray,
    block_frames: int,
    pool_style: str,
) -> np.ndarray:
    if block_frames <= 1:
        return frame_scores
    total_frames = frame_scores.shape[0]
    block_count = total_frames // block_frames
    if block_count == 0:
        return frame_scores
    trimmed = frame_scores[: block_count * block_frames]
    blocks = trimmed.reshape(block_count, block_frames, frame_scores.shape[1])
    if pool_style == "max_pool":
        return blocks.max(axis=1)
    if pool_style == "avg_pool":
        return blocks.mean(axis=1)
    if pool_style == "linear_pool":
        numerator = (blocks ** 2).sum(axis=1)
        denominator = blocks.sum(axis=1) + 1e-8
        return numerator / denominator
    if pool_style == "exp_pool":
        exp_blocks = np.exp(blocks)
        numerator = (blocks * exp_blocks).sum(axis=1)
        denominator = exp_blocks.sum(axis=1) + 1e-8
        return numerator / denominator
    return blocks.mean(axis=1)


def blockify_binary_labels(frame_labels: np.ndarray, block_frames: int) -> np.ndarray:
    if block_frames <= 1:
        return frame_labels
    total_frames = frame_labels.shape[0]
    block_count = total_frames // block_frames
    if block_count == 0:
        return frame_labels
    trimmed = frame_labels[: block_count * block_frames]
    blocks = trimmed.reshape(block_count, block_frames, frame_labels.shape[1])
    return blocks.max(axis=1)


def evaluate_localization(
    model: nn.Module,
    loader: DataLoader,
    strong_events: Dict[str, List[Dict[str, object]]],
    class_columns: List[str],
    threshold: float,
    device: torch.device,
    localization_mode: str,
    block_seconds: float,
    frame_duration: float,
    pool_style: str,
) -> Tuple[Dict[str, float], np.ndarray]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x_data, _y_data, _ids, mask, paths, starts, ends in loader:
            x_data = x_data.to(device)
            if mask is not None:
                mask = mask.to(device)
                _y_pred, y_frames = model(x_data, mask=mask)
            else:
                _y_pred, y_frames = model(x_data)
            y_frames = y_frames.cpu().numpy()
            mask_np = mask.cpu().numpy() if mask is not None else None
            for i in range(len(paths)):
                file_stem = anuraset_file_stem(paths[i])
                events = strong_events.get(file_stem, [])
                frames_len = y_frames[i].shape[0]
                if mask_np is not None:
                    frames_len = int(mask_np[i].sum())
                    frames_len = max(frames_len, 1)
                    frame_pred = y_frames[i][:frames_len]
                else:
                    frame_pred = y_frames[i]
                frame_true = localization_frame_labels(
                    events,
                    class_columns,
                    float(starts[i]),
                    float(ends[i]),
                    frames_len,
                )
                if localization_mode == "block":
                    block_frames = max(1, int(round(block_seconds / frame_duration)))
                    frame_pred = blockify_frames(frame_pred, block_frames, pool_style)
                    frame_true = blockify_binary_labels(frame_true, block_frames)
                preds.append(frame_pred)
                targets.append(frame_true)
    if not preds:
        return {
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }, np.zeros(len(class_columns), dtype=np.float32)
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = compute_metrics(y_true, y_pred, threshold)
    class_f1 = compute_class_f1(y_true, y_pred, threshold)
    return metrics, class_f1


def binarize_with_activity_detection(
    frame_pred: np.ndarray,
    class_columns: List[str],
    thresholds: Dict[str, np.ndarray],
    system: str,
    per_class_systems: Optional[List[str]] = None,
) -> np.ndarray:
    if per_class_systems is None:
        if system == "single":
            return (frame_pred >= thresholds["loc_threshold"]).astype(np.float32)
        n_frames, n_classes = frame_pred.shape
        binary = np.zeros((n_frames, n_classes), dtype=np.float32)
        for k in range(n_classes):
            pairs = activity_detection(
                x=frame_pred[:, k],
                thres=float(thresholds["loc_threshold_high"][k]),
                low_thres=float(thresholds["loc_threshold_low"][k]),
                n_smooth=thresholds["smooth"],
                n_salt=thresholds["smooth"],
            )
            for onset, offset in pairs:
                onset = max(0, min(n_frames, int(onset)))
                offset = max(0, min(n_frames, int(offset)))
                if offset > onset:
                    binary[onset:offset, k] = 1.0
        return binary
    n_frames, n_classes = frame_pred.shape
    binary = np.zeros((n_frames, n_classes), dtype=np.float32)
    for k in range(n_classes):
        if per_class_systems[k] == "double":
            pairs = activity_detection(
                x=frame_pred[:, k],
                thres=float(thresholds["loc_threshold_high"][k]),
                low_thres=float(thresholds["loc_threshold_low"][k]),
                n_smooth=thresholds["smooth"],
                n_salt=thresholds["smooth"],
            )
            for onset, offset in pairs:
                onset = max(0, min(n_frames, int(onset)))
                offset = max(0, min(n_frames, int(offset)))
                if offset > onset:
                    binary[onset:offset, k] = 1.0
        else:
            binary[:, k] = (frame_pred[:, k] >= thresholds["loc_threshold"][k]).astype(np.float32)
    return binary


def visualize_predictions(
    model: nn.Module,
    loader: DataLoader,
    strong_events: Dict[str, List[Dict[str, object]]],
    class_columns: List[str],
    thresholds: Dict[str, np.ndarray],
    output_root: str,
    prefix: str,
    max_per_class: int = 5,
    localization_mode: str = "frame",
    block_seconds: float = 1.0,
    frame_duration: float = 0.05,
    pool_style: str = "avg_pool",
    tagging_system: str = "single",
    localization_system: str = "single",
    per_class_tagging_systems: Optional[List[str]] = None,
    per_class_localization_systems: Optional[List[str]] = None,
) -> None:
    model.eval()
    selections = {name: {"correct": [], "wrong": []} for name in class_columns}
    with torch.no_grad():
        for x_data, y_data, _ids, mask, paths, starts, ends in loader:
            device = next(model.parameters()).device
            x_data = x_data.to(device)
            if mask is not None:
                mask = mask.to(x_data.device)
                clip_out, frame_out = model(x_data, mask=mask)
            else:
                clip_out, frame_out = model(x_data)
            clip_out = clip_out.cpu().numpy()
            frame_out = frame_out.cpu().numpy()
            mask_np = mask.cpu().numpy() if mask is not None else None
            y_true = y_data.cpu().numpy()
            tag_preds = apply_tagging_thresholds(
                clip_out,
                thresholds,
                tagging_system,
                per_class_systems=per_class_tagging_systems,
            )
            for i in range(len(paths)):
                file_stem = anuraset_file_stem(paths[i])
                events = strong_events.get(file_stem, [])
                frame_pred = frame_out[i]
                if mask_np is not None:
                    valid_len = int(mask_np[i].sum())
                    valid_len = max(valid_len, 1)
                    frame_pred = frame_pred[:valid_len]
                frame_true = localization_frame_labels(
                    events,
                    class_columns,
                    float(starts[i]),
                    float(ends[i]),
                    frame_pred.shape[0],
                )
                if localization_mode == "block":
                    block_frames = max(1, int(round(block_seconds / frame_duration)))
                    frame_pred = blockify_frames(frame_pred, block_frames, pool_style)
                    frame_true = blockify_binary_labels(frame_true, block_frames)
                for class_idx, class_name in enumerate(class_columns):
                    if len(selections[class_name]["correct"]) >= max_per_class and len(
                        selections[class_name]["wrong"]
                    ) >= max_per_class:
                        continue
                    pred = tag_preds[i, class_idx] >= 0.5
                    target = y_true[i, class_idx] >= 0.5
                    bucket = "correct" if pred == target else "wrong"
                    if len(selections[class_name][bucket]) >= max_per_class:
                        continue
                    selections[class_name][bucket].append(
                        {
                            "class_idx": class_idx,
                            "class_name": class_name,
                            "audio_path": paths[i],
                            "start": float(starts[i]),
                            "end": float(ends[i]),
                            "clip_out": clip_out[i],
                            "frame_pred": frame_pred,
                            "frame_true": frame_true,
                            "target": y_true[i],
                        }
                    )

    for class_name, buckets in selections.items():
        for bucket_name, samples in buckets.items():
            out_dir = os.path.join(output_root, class_name, bucket_name)
            os.makedirs(out_dir, exist_ok=True)
            for sample in samples:
                frame_pred = sample["frame_pred"]
                frame_true = sample["frame_true"]
                target = sample["target"]
                active_classes = np.where(target >= 0.5)[0].tolist()
                if not active_classes:
                    active_classes = [sample["class_idx"]]
                active_names = [class_columns[idx] for idx in active_classes]
                clip_len = max(sample["end"] - sample["start"], 1e-6)
                seq_len = frame_pred.shape[0]
                t_sec = np.arange(seq_len) * (clip_len / seq_len)

                def build_spacer_rows(data: np.ndarray, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
                    k = len(labels)
                    if k == 1:
                        rows = 5
                        layout = ["...", "...", labels[0], "...", "..."]
                        out = np.zeros((rows, data.shape[1]), dtype=data.dtype)
                        out[2] = data[0]
                        return out, layout
                    rows = 2 * k + 1
                    out = np.zeros((rows, data.shape[1]), dtype=data.dtype)
                    layout = ["..."] * rows
                    for idx in range(k):
                        row_idx = 1 + 2 * idx
                        out[row_idx] = data[idx]
                        layout[row_idx] = labels[idx]
                    return out, layout

                fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
                tag_label = tagging_system
                loc_label = localization_system
                if per_class_tagging_systems is not None:
                    tag_label = per_class_tagging_systems[sample["class_idx"]]
                if per_class_localization_systems is not None:
                    loc_label = per_class_localization_systems[sample["class_idx"]]
                fig.suptitle(f"Tagging={tag_label} | Localization={loc_label}", fontsize=12)
                gt_data = frame_true[:, active_classes].T
                gt_plot, y_labels = build_spacer_rows(gt_data, active_names)
                im0 = axes[0].imshow(
                    gt_plot,
                    aspect="auto",
                    origin="lower",
                    extent=[0, clip_len, 0, gt_plot.shape[0]],
                    vmin=0,
                    vmax=1,
                    cmap="Greys",
                )
                axes[0].set_yticks(np.arange(gt_plot.shape[0]) + 0.5)
                axes[0].set_yticklabels(y_labels)
                axes[0].set_title("Ground Truth (strong labels)")

                for idx in active_classes:
                    axes[1].plot(t_sec, frame_pred[:, idx], label=class_columns[idx])
                class_idx = sample["class_idx"]
                active_loc_system = localization_system
                if per_class_localization_systems is not None:
                    active_loc_system = per_class_localization_systems[class_idx]
                if active_loc_system == "single":
                    loc_value = float(thresholds["loc_threshold"][class_idx])
                    axes[1].axhline(loc_value, color="red", linestyle="--", linewidth=1)
                else:
                    axes[1].axhline(
                        float(thresholds["loc_threshold_high"][class_idx]),
                        color="red",
                        linestyle="--",
                        linewidth=1,
                    )
                    axes[1].axhline(
                        float(thresholds["loc_threshold_low"][class_idx]),
                        color="orange",
                        linestyle="--",
                        linewidth=1,
                    )
                axes[1].set_ylim(0, 1)
                axes[1].set_title("Frame-wise probabilities (line plot)")
                axes[1].legend(loc="upper right")

                pred_data = frame_pred[:, active_classes].T
                pred_plot, _ = build_spacer_rows(pred_data, active_names)
                im2 = axes[2].imshow(
                    pred_plot,
                    aspect="auto",
                    origin="lower",
                    extent=[0, clip_len, 0, pred_plot.shape[0]],
                    vmin=0,
                    vmax=1,
                    cmap="Greys",
                )
                axes[2].set_yticks(np.arange(pred_plot.shape[0]) + 0.5)
                axes[2].set_yticklabels(y_labels)
                axes[2].set_title("Frame-wise probabilities (heatmap)")

                binary = binarize_with_activity_detection(
                    frame_pred,
                    class_columns,
                    thresholds,
                    localization_system,
                    per_class_systems=per_class_localization_systems,
                )
                binary_data = binary[:, active_classes].T
                binary_plot, _ = build_spacer_rows(binary_data, active_names)
                im3 = axes[3].imshow(
                    binary_plot,
                    aspect="auto",
                    origin="lower",
                    extent=[0, clip_len, 0, binary_plot.shape[0]],
                    vmin=0,
                    vmax=1,
                    cmap="Greys",
                )
                axes[3].set_yticks(np.arange(binary_plot.shape[0]) + 0.5)
                axes[3].set_yticklabels(y_labels)
                axes[3].set_title(
                    "Frame-wise detections (after activity_detection)" if active_loc_system == "double"
                    else "Frame-wise detections (single threshold)"
                )
                axes[3].set_xlabel("Time (s)")

                fig.colorbar(im3, ax=[axes[0], axes[2], axes[3]], location="right", fraction=0.02, pad=0.02)
                audio_name = os.path.basename(sample["audio_path"]).replace(".wav", "")
                fig_path = os.path.join(out_dir, f"{audio_name}_{prefix}_{class_name}.png")
                fig.savefig(fig_path)
                plt.close(fig)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float,
    return_arrays: bool = False,
) -> Tuple[float, Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for x_data, y_data, _, mask, _paths, _starts, _ends in loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            if mask is not None:
                mask = mask.to(device)
                y_pred, _ = model(x_data, mask=mask)
            else:
                y_pred, _ = model(x_data)
            loss = loss_fn(y_pred, y_data).mean()
            losses.append(loss.item())
            preds.append(y_pred.cpu().numpy())
            targets.append(y_data.cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics = compute_metrics(y_true, y_pred, threshold)
    if return_arrays:
        return float(np.mean(losses)), metrics, y_true, y_pred
    return float(np.mean(losses)), metrics, None, None


def evaluate_tagging_cached(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Dict[str, np.ndarray],
    system: str,
) -> Tuple[float, np.ndarray]:
    y_hat = apply_tagging_thresholds(y_pred, thresholds, system)
    class_f1 = compute_class_f1_binary(y_true, y_hat)
    return float(np.mean(class_f1)) if class_f1.size else 0.0, class_f1


def evaluate_localization_cached(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_columns: List[str],
    thresholds: Dict[str, np.ndarray],
    system: str,
) -> Tuple[float, np.ndarray]:
    y_hat = apply_localization_thresholds(y_pred, class_columns, thresholds, system)
    class_f1 = compute_class_f1_binary(y_true, y_hat)
    return float(np.mean(class_f1)) if class_f1.size else 0.0, class_f1


def compute_metrics_binary(y_true: np.ndarray, y_hat: np.ndarray) -> Dict[str, float]:
    tp = (y_hat * y_true).sum(axis=0)
    fp = (y_hat * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_hat) * y_true).sum(axis=0)
    precision_per = tp / (tp + fp + 1e-10)
    recall_per = tp / (tp + fn + 1e-10)
    f1_per = 2 * precision_per * recall_per / (precision_per + recall_per + 1e-10)
    macro_precision = float(np.mean(precision_per)) if precision_per.size else 0.0
    macro_recall = float(np.mean(recall_per)) if recall_per.size else 0.0
    macro_f1 = float(np.mean(f1_per)) if f1_per.size else 0.0

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()
    micro_precision = float(total_tp / (total_tp + total_fp + 1e-10))
    micro_recall = float(total_tp / (total_tp + total_fn + 1e-10))
    micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-10))

    s = min(total_tp + total_fn, total_tp + total_fp) - total_tp
    d = max(0.0, total_fn - total_fp)
    i = max(0.0, total_fp - total_fn)
    micro_er = float((s + d + i) / (total_tp + total_fn + 1e-10))

    macro_er_list = []
    for k in range(y_true.shape[1]):
        s = min(tp[k] + fn[k], tp[k] + fp[k]) - tp[k]
        d = max(0.0, fn[k] - fp[k])
        i = max(0.0, fp[k] - fn[k])
        er = (s + d + i) / (tp[k] + fn[k] + 1e-10)
        macro_er_list.append(er)
    macro_er = float(np.mean(macro_er_list)) if macro_er_list else 0.0

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_er": micro_er,
        "macro_er": macro_er,
    }


def print_threshold_table(
    title: str,
    thresholds: Dict[str, np.ndarray],
    class_columns: List[str],
    system: str,
    objective: str,
) -> None:
    print(title)
    if system == "single":
        key = "tag_threshold" if objective == "tagging" else "loc_threshold"
        for class_name, value in zip(class_columns, thresholds[key]):
            print(f"  {class_name}: {float(value):.3f}")
    else:
        low_key = "tag_threshold_low" if objective == "tagging" else "loc_threshold_low"
        high_key = "tag_threshold_high" if objective == "tagging" else "loc_threshold_high"
        for class_name, low_value, high_value in zip(
            class_columns, thresholds[low_key], thresholds[high_key]
        ):
            print(f"  {class_name}: low={float(low_value):.3f}, high={float(high_value):.3f}")


def print_eval_block(
    title: str,
    metrics: Dict[str, float],
    class_columns: List[str],
    thresholds: Dict[str, np.ndarray],
    class_f1: np.ndarray,
    system: str,
    objective: str,
) -> None:
    print(title)
    print(metrics)
    if system == "single":
        key = "tag_threshold" if objective == "tagging" else "loc_threshold"
        for class_name, th_value, f1_value in zip(class_columns, thresholds[key], class_f1):
            print(f"{class_name}: th={float(th_value):.3f} - f1={float(f1_value):.2f}")
    else:
        low_key = "tag_threshold_low" if objective == "tagging" else "loc_threshold_low"
        high_key = "tag_threshold_high" if objective == "tagging" else "loc_threshold_high"
        for class_name, low_value, high_value, f1_value in zip(
            class_columns, thresholds[low_key], thresholds[high_key], class_f1
        ):
            print(
                f"{class_name}: low={float(low_value):.3f}, high={float(high_value):.3f} - "
                f"f1={float(f1_value):.2f}"
            )


def plot_training_history(history: Dict[str, List[float]], output_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    def plot_metric(ax, train_key, test_key, title, ylabel, ylim=None):
        ax.plot(epochs, history[train_key], label="train")
        ax.plot(epochs, history[test_key], label="test")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        if ylim is not None:
            ax.set_ylim(ylim)

    def common_ylim(keys: List[str]) -> Tuple[float, float]:
        values = [val for key in keys for val in history[key]]
        if not values:
            return 0.0, 1.0
        return min(values), max(values)

    plot_metric(axes[0], "train_loss", "test_loss", "Loss", "Loss")

    f1_ylim = common_ylim(["train_micro_f1", "test_micro_f1", "train_macro_f1", "test_macro_f1"])
    plot_metric(axes[1], "train_micro_f1", "test_micro_f1", "Micro F1", "F1", f1_ylim)
    plot_metric(axes[2], "train_macro_f1", "test_macro_f1", "Macro F1", "F1", f1_ylim)

    precision_ylim = common_ylim([
        "train_micro_precision",
        "test_micro_precision",
        "train_macro_precision",
        "test_macro_precision",
    ])
    plot_metric(axes[3], "train_micro_precision", "test_micro_precision", "Micro Precision", "Precision", precision_ylim)
    plot_metric(axes[4], "train_macro_precision", "test_macro_precision", "Macro Precision", "Precision", precision_ylim)

    recall_ylim = common_ylim([
        "train_micro_recall",
        "test_micro_recall",
        "train_macro_recall",
        "test_macro_recall",
    ])
    plot_metric(axes[5], "train_micro_recall", "test_micro_recall", "Micro Recall", "Recall", recall_ylim)
    plot_metric(axes[6], "train_macro_recall", "test_macro_recall", "Macro Recall", "Recall", recall_ylim)

    er_ylim = common_ylim([
        "train_micro_er",
        "test_micro_er",
        "train_macro_er",
        "test_macro_er",
    ])
    plot_metric(axes[7], "train_micro_er", "test_micro_er", "Micro ER", "ER", er_ylim)
    plot_metric(axes[8], "train_macro_er", "test_macro_er", "Macro ER", "ER", er_ylim)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_model(model_name: str, n_classes: int, pool_style: str, seq_len: int, n_mels: int) -> nn.Module:
    if model_name == "Baseline":
        return Baseline(pool_style=pool_style, n_classes=n_classes, seq_len=seq_len)
    if model_name == "CNN-biGRU":
        return CNNBiGRU(n_classes=n_classes, pool_style=pool_style, seq_len=seq_len, n_mels=n_mels)
    if model_name == "CNN-Transformer":
        return CNNTransformer(n_classes=n_classes, pool_style=pool_style, seq_len=seq_len, n_mels=n_mels)
    if model_name == "CDur":
        return CDur(pool_style=pool_style, n_classes=n_classes, seq_len=seq_len)
    if model_name == "TALNet":
        return TALNet(pool_style=pool_style, n_classes=n_classes, seq_len=seq_len)
    raise ValueError(f"Unsupported model name: {model_name}")


def run_birdnet_pipeline() -> None:
    def _normalize_dataset_name_for_birdnet(name: str) -> str:
        normalized = str(name).strip().upper()
        if normalized == "ANURASET":
            return "ANURASET"
        if normalized == "FNJV":
            fnjv_root = str(getattr(config, "FNJV_ROOT", ""))
            return "FNJV_458" if "458" in fnjv_root else "FNJV_578"
        return normalized

    repo_root = Path(__file__).resolve().parent
    focal_dir = repo_root / "focal-data"
    birdnet_main_path = focal_dir / "main.py"
    if not birdnet_main_path.exists():
        raise FileNotFoundError(f"BirdNET main.py not found: {birdnet_main_path}")

    focal_path = str(focal_dir)
    if focal_path not in sys.path:
        sys.path.insert(0, focal_path)

    import importlib.util

    # BirdNET dataset loader expects explicit FNJV variant names (FNJV_458/FNJV_578).
    config.DATASET_TRAIN = _normalize_dataset_name_for_birdnet(getattr(config, "DATASET_TRAIN", ""))
    config.DATASET_VAL = _normalize_dataset_name_for_birdnet(getattr(config, "DATASET_VAL", ""))
    config.DATASET_TEST = _normalize_dataset_name_for_birdnet(getattr(config, "DATASET_TEST", ""))

    def _select_subset_csv(default_csv: str, subset_csv: str) -> str:
        return subset_csv if os.path.exists(subset_csv) else default_csv

    # BirdNET AnuraSet loader requires a `subset` column.
    # Prefer explicit *_WITH_SUBSET csv files if present, fallback to classic paths.
    config.STRONG_LABELS_458 = _select_subset_csv(
        str(getattr(config, "STRONG_LABELS_458", "")),
        str(getattr(config, "STRONG_LABELS_458_WITH_SUBSET", "")),
    )
    config.STRONG_LABELS_578 = _select_subset_csv(
        str(getattr(config, "STRONG_LABELS_578", "")),
        str(getattr(config, "STRONG_LABELS_578_WITH_SUBSET", "")),
    )

    spec = importlib.util.spec_from_file_location("focal_data_main", birdnet_main_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load BirdNET pipeline from {birdnet_main_path}")

    module = importlib.util.module_from_spec(spec)
    # dataclasses (and some other decorators/introspection code) expect the module
    # to be present in sys.modules while class bodies are being executed.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError("focal-data/main.py does not expose a main() entrypoint.")
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(focal_dir))
        module.main()
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    pipeline_mode = getattr(config, "PIPELINE", "auto")
    model_name_for_dispatch = str(getattr(config, "MODEL_NAME", "")).strip().lower()
    use_birdnet = pipeline_mode == "birdnet" or (pipeline_mode == "auto" and model_name_for_dispatch == "birdnet")

    if use_birdnet:
        print(">>> [dispatch] Running unified BirdNET pipeline from focal-data/main.py")
        run_birdnet_pipeline()
        raise SystemExit(0)

    set_seed(config.SEED)

    ANURASET_ROOT = config.ANURASET_ROOT
    FNJV_ROOT = config.FNJV_ROOT

    DATASET_TRAIN = config.DATASET_TRAIN
    DATASET_VAL = config.DATASET_VAL
    DATASET_TEST = config.DATASET_TEST

    POOLING = config.POOLING
    BAG_SECONDS = config.BAG_SECONDS
    FULL_BAG_METHOD = config.FULL_BAG_METHOD
    PAD_MODE = config.PAD_MODE

    MODEL_NAME = config.MODEL_NAME
    EPOCHS = config.EPOCHS
    BATCH_SIZE = config.BATCH_SIZE
    NUM_WORKERS = config.NUM_WORKERS
    LEARNING_RATE = config.LEARNING_RATE

    VALIDATION_SPLIT = config.VALIDATION_SPLIT
    TEST_SPLIT = config.TEST_SPLIT
    APPLY_VALIDATION_SPLIT = config.APPLY_VALIDATION_SPLIT
    APPLY_TEST_SPLIT = config.APPLY_TEST_SPLIT

    TARGET_SPECIES = config.TARGET_SPECIES
    INCLUDE_NEGATIVE_SPLITS = config.INCLUDE_NEGATIVE_SPLITS
    USE_CLASS_SPECIFIC_THRESHOLD_TUNING = config.USE_CLASS_SPECIFIC_THRESHOLD_TUNING
    STRONG_LABELS_458 = config.STRONG_LABELS_458
    STRONG_LABELS_578 = config.STRONG_LABELS_578
    STRONG_LABEL_LEVELS = config.STRONG_LABEL_LEVELS
    ANURASET_EVAL = config.ANURASET_EVAL
    LOCALIZATION_MODE = config.LOCALIZATION_MODE
    BLOCK_SECONDS = config.BLOCK_SECONDS

    pool_map = {
        "max": "max_pool",
        "mean": "avg_pool",
        "linear": "linear_pool",
        "exp": "exp_pool",
        "att": "attention_pool",
        "auto": "auto_pool",
        "power": "power_pool",
        "hi": "hi_pool",
        "hi_plus": "hi_pool_plus",
        "hi_fixed": "hi_pool_fixed",
    }
    pool_style = pool_map[POOLING]

    sample_rate = config.sample_rate
    n_mels = config.n_mels
    n_fft = config.n_fft
    hop_length = config.hop_length
    OVERLAP_BAGS = config.OVERLAP_BAGS
    HOP_SECONDS = config.HOP_SECONDS
    full_bag = BAG_SECONDS == "full"

    if full_bag and FULL_BAG_METHOD == "batch":
        BATCH_SIZE = 1

    if full_bag and FULL_BAG_METHOD == "batch" and pool_requires_fixed_seq_len(pool_style):
        print(">>> [config] Full-bag batch mode: switching pooling to avg_pool for variable length input.")
        pool_style = "avg_pool"

    metadata_path = os.path.join(ANURASET_ROOT, "metadata.csv")
    anuraset_metadata = pd.read_csv(metadata_path)
    class_columns = get_class_columns(anuraset_metadata, TARGET_SPECIES)

    fnjv_segments = []
    fnjv_codes: List[str] = []
    if DATASET_TRAIN == "FNJV" or DATASET_VAL == "FNJV" or DATASET_TEST == "FNJV":
        fnjv_segments, fnjv_codes = build_fnjv_segments(
            FNJV_ROOT,
            class_columns,
            BAG_SECONDS,
            OVERLAP_BAGS,
            HOP_SECONDS,
            full_bag,
        )
        if TARGET_SPECIES:
            class_columns = [col for col in class_columns if col in TARGET_SPECIES]
        if DATASET_TRAIN == "FNJV":
            class_columns = [col for col in class_columns if col in fnjv_codes]

    train_segments = []
    val_segments = []
    test_segments = []

    anura_train_segments = build_anuraset_segments(
        ANURASET_ROOT,
        anuraset_metadata,
        class_columns,
        BAG_SECONDS,
        OVERLAP_BAGS,
        HOP_SECONDS,
        full_bag,
        subset="train",
    )
    anura_test_segments = build_anuraset_segments(
        ANURASET_ROOT,
        anuraset_metadata,
        class_columns,
        BAG_SECONDS,
        OVERLAP_BAGS,
        HOP_SECONDS,
        full_bag,
        subset="test",
    )

    if APPLY_VALIDATION_SPLIT:
        if TARGET_SPECIES:
            anura_val_segments, anura_train_negatives = select_target_segments(
                anura_train_segments,
                VALIDATION_SPLIT,
                seed=config.SEED,
                include_negatives=INCLUDE_NEGATIVE_SPLITS,
            )
            anura_train_segments = anura_train_negatives
        else:
            anura_train_segments, anura_val_segments, _ = split_segments(
                anura_train_segments,
                VALIDATION_SPLIT,
                0.0,
                seed=config.SEED,
            )
    else:
        anura_val_segments = anura_train_segments

    if APPLY_TEST_SPLIT:
        if TARGET_SPECIES:
            anura_test_segments, _ = select_target_segments(
                anura_test_segments,
                TEST_SPLIT,
                seed=config.SEED,
                include_negatives=INCLUDE_NEGATIVE_SPLITS,
            )
        else:
            _, _, anura_test_segments = split_segments(
                anura_test_segments,
                0.0,
                TEST_SPLIT,
                seed=config.SEED,
            )

    fnjv_id = "578" if "578" in FNJV_ROOT else "458"
    strong_labels_path = STRONG_LABELS_578 if fnjv_id == "578" else STRONG_LABELS_458
    strong_events = None
    if os.path.exists(strong_labels_path):
        strong_events = load_strong_labels(strong_labels_path, STRONG_LABEL_LEVELS)

    if APPLY_VALIDATION_SPLIT or APPLY_TEST_SPLIT:
        fnjv_train_segments, fnjv_val_segments, fnjv_test_segments = split_segments(
            fnjv_segments,
            VALIDATION_SPLIT if APPLY_VALIDATION_SPLIT else 0.0,
            TEST_SPLIT if APPLY_TEST_SPLIT else 0.0,
            seed=config.SEED,
        )
    else:
        fnjv_train_segments = fnjv_segments
        fnjv_val_segments = fnjv_segments
        fnjv_test_segments = fnjv_segments

    if DATASET_TRAIN == "AnuraSet":
        train_segments = anura_train_segments
    else:
        train_segments = fnjv_train_segments

    if DATASET_VAL == "AnuraSet":
        val_segments = anura_val_segments
    else:
        val_segments = fnjv_val_segments

    if DATASET_TEST == "AnuraSet":
        test_segments = anura_test_segments
    else:
        test_segments = fnjv_test_segments

    if strong_events is not None:
        if DATASET_VAL == "AnuraSet":
            updated_segments = []
            for seg in val_segments:
                stem = anuraset_file_stem(seg.audio_path)
                events = strong_events.get(stem, [])
                label = strong_label_vector(events, class_columns, seg.start_sec, seg.end_sec)
                updated_segments.append(
                    AudioSegment(
                        audio_path=seg.audio_path,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        label=label,
                        segment_id=seg.segment_id,
                        duration_sec=seg.duration_sec,
                    )
                )
            val_segments = updated_segments
        if DATASET_TEST == "AnuraSet":
            updated_segments = []
            for seg in test_segments:
                stem = anuraset_file_stem(seg.audio_path)
                events = strong_events.get(stem, [])
                label = strong_label_vector(events, class_columns, seg.start_sec, seg.end_sec)
                updated_segments.append(
                    AudioSegment(
                        audio_path=seg.audio_path,
                        start_sec=seg.start_sec,
                        end_sec=seg.end_sec,
                        label=label,
                        segment_id=seg.segment_id,
                        duration_sec=seg.duration_sec,
                    )
                )
            test_segments = updated_segments

    if full_bag and FULL_BAG_METHOD == "pad":
        max_duration_sec = max(
            get_max_duration(train_segments + val_segments + test_segments),
            0.0,
        )
        fixed_frames = (
            compute_frames(max_duration_sec, sample_rate, n_fft, hop_length)
            if max_duration_sec > 0
            else None
        )
        if fixed_frames is None:
            raise ValueError("Full-bag pad mode requires at least one segment to compute max length.")
    else:
        fixed_frames = None

    segment_seconds = BAG_SECONDS if isinstance(BAG_SECONDS, int) else 1

    train_dataset = AudioSegmentDataset(
        train_segments,
        sample_rate,
        n_mels,
        n_fft,
        hop_length,
        segment_seconds,
        fixed_frames=fixed_frames,
    )
    val_dataset = AudioSegmentDataset(
        val_segments,
        sample_rate,
        n_mels,
        n_fft,
        hop_length,
        segment_seconds,
        fixed_frames=fixed_frames,
    )
    test_dataset = AudioSegmentDataset(
        test_segments,
        sample_rate,
        n_mels,
        n_fft,
        hop_length,
        segment_seconds,
        fixed_frames=fixed_frames,
    )

    num_classes = len(class_columns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_seq_len = fixed_frames if fixed_frames is not None else train_dataset.seq_len
    model = build_model(MODEL_NAME, num_classes, pool_style, model_seq_len, n_mels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    loss_fn = nn.BCELoss()

    if full_bag and FULL_BAG_METHOD == "pad":
        collate_fn = lambda batch: collate_full_bag_pad(batch, fixed_frames, PAD_MODE)
    else:
        collate_fn = collate_default

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    output_dir = os.path.join(
        "TALNet",
        "outputs",
        f"{DATASET_TRAIN}_{POOLING}_{BAG_SECONDS}sec_{EPOCHS}epoch_{BATCH_SIZE}batch",
    )
    os.makedirs(output_dir, exist_ok=True)
    metrics_plot_path = os.path.join(output_dir, "training_metrics.png")
    best_macro_path = os.path.join(output_dir, "best_model_macro.pt")
    best_micro_path = os.path.join(output_dir, "best_model_micro.pt")

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_micro_f1": [],
        "test_micro_f1": [],
        "train_macro_f1": [],
        "test_macro_f1": [],
        "train_micro_precision": [],
        "test_micro_precision": [],
        "train_macro_precision": [],
        "test_macro_precision": [],
        "train_micro_recall": [],
        "test_micro_recall": [],
        "train_macro_recall": [],
        "test_macro_recall": [],
        "train_micro_er": [],
        "test_micro_er": [],
        "train_macro_er": [],
        "test_macro_er": [],
    }

    best_macro_f1 = -1.0
    best_micro_f1 = -1.0

    threshold = config.threshold

    print(">>> [config] SEED=", config.SEED)
    print(">>> [config] ANURASET_ROOT=", ANURASET_ROOT)
    print(">>> [config] FNJV_ROOT=", FNJV_ROOT)
    print(">>> [config] DATASET_TRAIN=", DATASET_TRAIN)
    print(">>> [config] DATASET_VAL=", DATASET_VAL)
    print(">>> [config] DATASET_TEST=", DATASET_TEST)
    print(">>> [config] POOLING=", POOLING)
    print(">>> [config] BAG_SECONDS=", BAG_SECONDS)
    print(">>> [config] FULL_BAG_METHOD=", FULL_BAG_METHOD)
    print(">>> [config] PAD_MODE=", PAD_MODE)
    print(">>> [config] MODEL_NAME=", MODEL_NAME)
    print(">>> [config] EPOCHS=", EPOCHS)
    print(">>> [config] BATCH_SIZE=", BATCH_SIZE)
    print(">>> [config] NUM_WORKERS=", NUM_WORKERS)
    print(">>> [config] LEARNING_RATE=", LEARNING_RATE)
    print(">>> [config] VALIDATION_SPLIT=", VALIDATION_SPLIT)
    print(">>> [config] TEST_SPLIT=", TEST_SPLIT)
    print(">>> [config] APPLY_VALIDATION_SPLIT=", APPLY_VALIDATION_SPLIT)
    print(">>> [config] APPLY_TEST_SPLIT=", APPLY_TEST_SPLIT)
    print(">>> [config] TARGET_SPECIES=", TARGET_SPECIES)
    print(">>> [config] INCLUDE_NEGATIVE_SPLITS=", INCLUDE_NEGATIVE_SPLITS)
    print(">>> [config] USE_CLASS_SPECIFIC_THRESHOLD_TUNING=", USE_CLASS_SPECIFIC_THRESHOLD_TUNING)
    print(">>> [config] STRONG_LABEL_LEVELS=", STRONG_LABEL_LEVELS)
    print(">>> [config] ANURASET_EVAL=", ANURASET_EVAL)
    print(">>> [config] LOCALIZATION_MODE=", LOCALIZATION_MODE)
    print(">>> [config] BLOCK_SECONDS=", BLOCK_SECONDS)
    print(">>> [config] sample_rate=", sample_rate)
    print(">>> [config] n_mels=", n_mels)
    print(">>> [config] n_fft=", n_fft)
    print(">>> [config] hop_length=", hop_length)
    print(">>> [config] threshold=", threshold)
    print(">>> [config] OVERLAP_BAGS=", OVERLAP_BAGS)
    print(">>> [config] HOP_SECONDS=", HOP_SECONDS)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        train_preds = []
        train_targets = []
        for x_data, y_data, _, mask, _paths, _starts, _ends in train_loader:
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            if mask is not None:
                mask = mask.to(device)
                y_pred, _ = model(x_data, mask=mask)
            else:
                y_pred, _ = model(x_data)
            loss = loss_fn(y_pred, y_data).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds.append(y_pred.detach().cpu().numpy())
            train_targets.append(y_data.detach().cpu().numpy())

        scheduler.step()
        train_loss = float(np.mean(train_losses))
        train_metrics = compute_metrics(
            np.concatenate(train_targets, axis=0),
            np.concatenate(train_preds, axis=0),
            threshold,
        )

        test_loss, test_metrics, _, _ = evaluate_model(model, val_loader, loss_fn, device, threshold)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        for key in [
            "micro_f1",
            "macro_f1",
            "micro_precision",
            "macro_precision",
            "micro_recall",
            "macro_recall",
            "micro_er",
            "macro_er",
        ]:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"test_{key}"].append(test_metrics[key])

        plot_training_history(history, metrics_plot_path)

        if test_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = test_metrics["macro_f1"]
            torch.save(model.state_dict(), best_macro_path)

        if test_metrics["micro_f1"] > best_micro_f1:
            best_micro_f1 = test_metrics["micro_f1"]
            torch.save(model.state_dict(), best_micro_path)

        print(
            "Epoch {epoch}: train loss={train_loss:.4f}, micro-P={train_micro_p:.4f}, "
            "micro-R={train_micro_r:.4f}, micro-F1={train_micro_f1:.4f}, "
            "macro-P={train_macro_p:.4f}, macro-R={train_macro_r:.4f}, "
            "macro-F1={train_macro_f1:.4f}, ER={train_er:.4f}".format(
                epoch=epoch,
                train_loss=train_loss,
                train_micro_p=train_metrics["micro_precision"],
                train_micro_r=train_metrics["micro_recall"],
                train_micro_f1=train_metrics["micro_f1"],
                train_macro_p=train_metrics["macro_precision"],
                train_macro_r=train_metrics["macro_recall"],
                train_macro_f1=train_metrics["macro_f1"],
                train_er=train_metrics["macro_er"],
            )
        )
        print(
            "           val loss={val_loss:.4f}, micro-P={val_micro_p:.4f}, "
            "micro-R={val_micro_r:.4f}, micro-F1={val_micro_f1:.4f}, "
            "macro-P={val_macro_p:.4f}, macro-R={val_macro_r:.4f}, "
            "macro-F1={val_macro_f1:.4f}, ER={val_er:.4f}".format(
                val_loss=test_loss,
                val_micro_p=test_metrics["micro_precision"],
                val_micro_r=test_metrics["micro_recall"],
                val_micro_f1=test_metrics["micro_f1"],
                val_macro_p=test_metrics["macro_precision"],
                val_macro_r=test_metrics["macro_recall"],
                val_macro_f1=test_metrics["macro_f1"],
                val_er=test_metrics["macro_er"],
            )
        )

    best_macro_model = build_model(MODEL_NAME, num_classes, pool_style, model_seq_len, n_mels).to(device)
    best_macro_model.load_state_dict(torch.load(best_macro_path, map_location=device))

    best_micro_model = build_model(MODEL_NAME, num_classes, pool_style, model_seq_len, n_mels).to(device)
    best_micro_model.load_state_dict(torch.load(best_micro_path, map_location=device))

    frame_duration = hop_length / sample_rate

    if USE_CLASS_SPECIFIC_THRESHOLD_TUNING:
        tuning_step = 0.05
        max_rounds = 50
        base_thresholds = {
            "tag_threshold": np.full(num_classes, threshold, dtype=np.float32),
            "tag_threshold_low": np.full(num_classes, threshold, dtype=np.float32),
            "tag_threshold_high": np.full(num_classes, threshold, dtype=np.float32),
            "loc_threshold": np.full(num_classes, ANURASET_EVAL["loc_threshold_high"], dtype=np.float32),
            "loc_threshold_low": np.full(num_classes, ANURASET_EVAL["loc_threshold_low"], dtype=np.float32),
            "loc_threshold_high": np.full(num_classes, ANURASET_EVAL["loc_threshold_high"], dtype=np.float32),
            "smooth": ANURASET_EVAL["smooth"],
        }

        def run_tuning_for_model(model: nn.Module, name: str) -> None:
            val_cache = cache_model_outputs(
                model,
                val_loader,
                device,
                strong_events if DATASET_VAL == "AnuraSet" else None,
                class_columns,
                LOCALIZATION_MODE,
                BLOCK_SECONDS,
                frame_duration,
                pool_style,
            )
            test_cache = cache_model_outputs(
                model,
                test_loader,
                device,
                strong_events if DATASET_TEST == "AnuraSet" else None,
                class_columns,
                LOCALIZATION_MODE,
                BLOCK_SECONDS,
                frame_duration,
                pool_style,
            )

            print(f"{name} Tagging Threshold Tuning (single)")
            tag_single, tag_single_score = iterative_threshold_optimization(
                val_cache["clip_true"],
                val_cache["clip_out"],
                {"tag_threshold": base_thresholds["tag_threshold"]},
                system="single",
                step=tuning_step,
                max_rounds=max_rounds,
                objective="tagging",
            )
            print_threshold_table(
                f"{name} Tagging Thresholds (single)",
                tag_single,
                class_columns,
                system="single",
                objective="tagging",
            )

            print(f"{name} Tagging Threshold Tuning (double)")
            tag_double, tag_double_score = iterative_threshold_optimization(
                val_cache["clip_true"],
                val_cache["clip_out"],
                {
                    "tag_threshold_low": base_thresholds["tag_threshold_low"],
                    "tag_threshold_high": base_thresholds["tag_threshold_high"],
                },
                system="double",
                step=tuning_step,
                max_rounds=max_rounds,
                objective="tagging",
            )
            print_threshold_table(
                f"{name} Tagging Thresholds (double)",
                tag_double,
                class_columns,
                system="double",
                objective="tagging",
            )

            tag_single_f1, _ = evaluate_tagging_cached(
                test_cache["clip_true"], test_cache["clip_out"], tag_single, system="single"
            )
            tag_double_f1, _ = evaluate_tagging_cached(
                test_cache["clip_true"], test_cache["clip_out"], tag_double, system="double"
            )
            tag_single_hat = apply_tagging_thresholds(test_cache["clip_out"], tag_single, "single")
            tag_double_hat = apply_tagging_thresholds(test_cache["clip_out"], tag_double, "double")
            tag_single_metrics = compute_metrics_binary(test_cache["clip_true"], tag_single_hat)
            tag_double_metrics = compute_metrics_binary(test_cache["clip_true"], tag_double_hat)
            tag_single_class_f1 = compute_class_f1_binary(test_cache["clip_true"], tag_single_hat)
            tag_double_class_f1 = compute_class_f1_binary(test_cache["clip_true"], tag_double_hat)
            print_eval_block(
                f"{name} Tagging (single)",
                tag_single_metrics,
                class_columns,
                tag_single,
                tag_single_class_f1,
                system="single",
                objective="tagging",
            )
            print_eval_block(
                f"{name} Tagging (double)",
                tag_double_metrics,
                class_columns,
                tag_double,
                tag_double_class_f1,
                system="double",
                objective="tagging",
            )

            loc_single_f1 = 0.0
            loc_double_f1 = 0.0
            loc_single = {}
            loc_double = {}
            loc_single_score = 0.0
            loc_double_score = 0.0
            loc_single_class_f1 = np.zeros(len(class_columns), dtype=np.float32)
            loc_double_class_f1 = np.zeros(len(class_columns), dtype=np.float32)
            if val_cache["frame_true"].size and test_cache["frame_true"].size:
                print(f"{name} Localization Threshold Tuning (single)")
                loc_single, loc_single_score = iterative_threshold_optimization(
                    val_cache["frame_true"],
                    val_cache["frame_out"],
                    {"loc_threshold": base_thresholds["loc_threshold"]},
                    system="single",
                    step=tuning_step,
                    max_rounds=max_rounds,
                    objective="localization",
                    class_columns=class_columns,
                )
                print_threshold_table(
                    f"{name} Localization Thresholds (single)",
                    loc_single,
                    class_columns,
                    system="single",
                    objective="localization",
                )

                print(f"{name} Localization Threshold Tuning (double)")
                loc_double, loc_double_score = iterative_threshold_optimization(
                    val_cache["frame_true"],
                    val_cache["frame_out"],
                    {
                        "loc_threshold_low": base_thresholds["loc_threshold_low"],
                        "loc_threshold_high": base_thresholds["loc_threshold_high"],
                        "smooth": base_thresholds["smooth"],
                    },
                    system="double",
                    step=tuning_step,
                    max_rounds=max_rounds,
                    objective="localization",
                    class_columns=class_columns,
                )
                print_threshold_table(
                    f"{name} Localization Thresholds (double)",
                    loc_double,
                    class_columns,
                    system="double",
                    objective="localization",
                )

                loc_single_f1, _ = evaluate_localization_cached(
                    test_cache["frame_true"],
                    test_cache["frame_out"],
                    class_columns,
                    loc_single,
                    system="single",
                )
                loc_double_f1, _ = evaluate_localization_cached(
                    test_cache["frame_true"],
                    test_cache["frame_out"],
                    class_columns,
                    loc_double,
                    system="double",
                )
                loc_single_hat = apply_localization_thresholds(
                    test_cache["frame_out"], class_columns, loc_single, "single"
                )
                loc_double_hat = apply_localization_thresholds(
                    test_cache["frame_out"], class_columns, loc_double, "double"
                )
                loc_single_metrics = compute_metrics_binary(test_cache["frame_true"], loc_single_hat)
                loc_double_metrics = compute_metrics_binary(test_cache["frame_true"], loc_double_hat)
                loc_single_class_f1 = compute_class_f1_binary(test_cache["frame_true"], loc_single_hat)
                loc_double_class_f1 = compute_class_f1_binary(test_cache["frame_true"], loc_double_hat)
                print_eval_block(
                    f"{name} Localization (single)",
                    loc_single_metrics,
                    class_columns,
                    loc_single,
                    loc_single_class_f1,
                    system="single",
                    objective="localization",
                )
                print_eval_block(
                    f"{name} Localization (double)",
                    loc_double_metrics,
                    class_columns,
                    loc_double,
                    loc_double_class_f1,
                    system="double",
                    objective="localization",
                )

            tag_system = "single" if tag_single_score >= tag_double_score else "double"
            loc_system = "single" if loc_single_score >= loc_double_score else "double"
            per_class_tagging = [
                "double" if tag_double_class_f1[i] > tag_single_class_f1[i] else "single"
                for i in range(len(class_columns))
            ]
            per_class_loc = [
                "double" if loc_double_class_f1[i] > loc_single_class_f1[i] else "single"
                for i in range(len(class_columns))
            ]
            combined_thresholds = {
                "tag_threshold": tag_single.get("tag_threshold", base_thresholds["tag_threshold"]),
                "tag_threshold_low": tag_double.get("tag_threshold_low", base_thresholds["tag_threshold_low"]),
                "tag_threshold_high": tag_double.get("tag_threshold_high", base_thresholds["tag_threshold_high"]),
                "loc_threshold": loc_single.get("loc_threshold", base_thresholds["loc_threshold"]),
                "loc_threshold_low": loc_double.get("loc_threshold_low", base_thresholds["loc_threshold_low"]),
                "loc_threshold_high": loc_double.get("loc_threshold_high", base_thresholds["loc_threshold_high"]),
                "smooth": base_thresholds["smooth"],
            }

            if strong_events is not None and DATASET_TEST == "AnuraSet":
                visualize_predictions(
                    model,
                    test_loader,
                    strong_events,
                    class_columns,
                    combined_thresholds,
                    output_root="results/AnuraSet/viz",
                    prefix=f"{name.lower()}_{tag_system}_{loc_system}",
                    localization_mode=LOCALIZATION_MODE,
                    block_seconds=BLOCK_SECONDS,
                    frame_duration=frame_duration,
                    pool_style=pool_style,
                    tagging_system=tag_system,
                    localization_system=loc_system,
                    per_class_tagging_systems=per_class_tagging,
                    per_class_localization_systems=per_class_loc,
                )

        run_tuning_for_model(best_macro_model, "Best Macro Model")
        run_tuning_for_model(best_micro_model, "Best Micro Model")
    else:
        macro_test_loss, macro_test_metrics, macro_y_true, macro_y_pred = evaluate_model(
            best_macro_model, test_loader, loss_fn, device, threshold, return_arrays=True
        )
        macro_class_f1 = compute_class_f1(macro_y_true, macro_y_pred, threshold)

        micro_test_loss, micro_test_metrics, micro_y_true, micro_y_pred = evaluate_model(
            best_micro_model, test_loader, loss_fn, device, threshold, return_arrays=True
        )
        micro_class_f1 = compute_class_f1(micro_y_true, micro_y_pred, threshold)

        print("Best Macro Model Test Results")
        print(f"Loss: {macro_test_loss:.4f}")
        print(macro_test_metrics)
        print("Macro Model Class F1:")
        for class_name, f1_value in zip(class_columns, macro_class_f1):
            print(f"  {class_name}: {f1_value:.4f}")

        print("Best Micro Model Test Results")
        print(f"Loss: {micro_test_loss:.4f}")
        print(micro_test_metrics)
        print("Micro Model Class F1:")
        for class_name, f1_value in zip(class_columns, micro_class_f1):
            print(f"  {class_name}: {f1_value:.4f}")

        if strong_events is not None and DATASET_TEST == "AnuraSet":
            loc_metrics_macro, loc_class_f1_macro = evaluate_localization(
                best_macro_model,
                test_loader,
                strong_events,
                class_columns,
                threshold,
                device,
                LOCALIZATION_MODE,
                BLOCK_SECONDS,
                frame_duration,
                pool_style,
            )
            print("Macro Model Localization Metrics")
            print(loc_metrics_macro)
            print("Macro Model Localization Class F1:")
            for class_name, f1_value in zip(class_columns, loc_class_f1_macro):
                print(f"  {class_name}: {f1_value:.4f}")

            loc_metrics_micro, loc_class_f1_micro = evaluate_localization(
                best_micro_model,
                test_loader,
                strong_events,
                class_columns,
                threshold,
                device,
                LOCALIZATION_MODE,
                BLOCK_SECONDS,
                frame_duration,
                pool_style,
            )
            print("Micro Model Localization Metrics")
            print(loc_metrics_micro)
            print("Micro Model Localization Class F1:")
            for class_name, f1_value in zip(class_columns, loc_class_f1_micro):
                print(f"  {class_name}: {f1_value:.4f}")

            thresholds = {
                "tag_threshold": np.full(num_classes, threshold, dtype=np.float32),
                "loc_threshold": np.full(num_classes, ANURASET_EVAL["loc_threshold_high"], dtype=np.float32),
                "loc_threshold_low": np.full(num_classes, ANURASET_EVAL["loc_threshold_low"], dtype=np.float32),
                "loc_threshold_high": np.full(num_classes, ANURASET_EVAL["loc_threshold_high"], dtype=np.float32),
                "smooth": ANURASET_EVAL["smooth"],
            }
            visualize_predictions(
                best_macro_model,
                test_loader,
                strong_events,
                class_columns,
                thresholds,
                output_root="results/AnuraSet/viz",
                prefix="macro_fixed",
                localization_mode=LOCALIZATION_MODE,
                block_seconds=BLOCK_SECONDS,
                frame_duration=frame_duration,
                pool_style=pool_style,
                tagging_system="single",
                localization_system="single",
            )
            visualize_predictions(
                best_micro_model,
                test_loader,
                strong_events,
                class_columns,
                thresholds,
                output_root="results/AnuraSet/viz",
                prefix="micro_fixed",
                localization_mode=LOCALIZATION_MODE,
                block_seconds=BLOCK_SECONDS,
                frame_duration=frame_duration,
                pool_style=pool_style,
                tagging_system="single",
                localization_system="single",
            )
