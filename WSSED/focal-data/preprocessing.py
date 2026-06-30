"""Audio preprocessing helpers for bag-instance construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio

import config


@dataclass
class InstanceWindow:
    bag_id: str
    instance_index: int
    start_sec: float
    end_sec: float
    waveform: torch.Tensor
    sample_rate: int


def load_audio_mono(audio_path: str, target_sample_rate: int | None = None) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(audio_path)
    mono_waveform = torch.mean(waveform, dim=0, keepdim=True)
    desired_sr = target_sample_rate if target_sample_rate is not None else config.SAMPLE_RATE
    if sample_rate != desired_sr:
        mono_waveform = torchaudio.functional.resample(mono_waveform, sample_rate, desired_sr)
        sample_rate = desired_sr
    return mono_waveform, sample_rate


def build_birdnet_segment_batch(
    audio_path: str,
    window_seconds: float,
    hop_seconds: float,
    birdnet_sample_rate: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build BirdNET-ready batch with shape [n_segments, 144000] for 3s at 48kHz."""
    mono_waveform, sample_rate = load_audio_mono(audio_path, target_sample_rate=birdnet_sample_rate)
    windows = split_into_windows(
        bag_id="birdnet",
        mono_waveform=mono_waveform,
        sample_rate=sample_rate,
        window_seconds=window_seconds,
        hop_seconds=hop_seconds,
    )
    if not windows:
        raise RuntimeError(f"No windows built for BirdNET batch: {audio_path}")

    target_len = int(window_seconds * birdnet_sample_rate)
    rows = []
    starts = []
    ends = []
    for window in windows:
        segment = window.waveform.squeeze(0)
        if segment.numel() < target_len:
            segment = torch.nn.functional.pad(segment, (0, target_len - segment.numel()))
        elif segment.numel() > target_len:
            segment = segment[:target_len]
        rows.append(segment)
        starts.append(window.start_sec)
        ends.append(window.end_sec)

    batch = torch.stack(rows, dim=0).float()  # [n_segments, 144000]
    return batch, torch.tensor(starts, dtype=torch.float32), torch.tensor(ends, dtype=torch.float32)


def split_into_windows(
    bag_id: str,
    mono_waveform: torch.Tensor,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> List[InstanceWindow]:
    window_size = int(window_seconds * sample_rate)
    hop_size = int(hop_seconds * sample_rate)

    if mono_waveform.shape[1] < window_size:
        pad_size = window_size - mono_waveform.shape[1]
        mono_waveform = torch.nn.functional.pad(mono_waveform, (0, pad_size))

    windows: List[InstanceWindow] = []
    max_start = mono_waveform.shape[1] - window_size

    instance_idx = 0
    for start in range(0, max_start + 1, hop_size):
        end = start + window_size
        chunk = mono_waveform[:, start:end]
        windows.append(
            InstanceWindow(
                bag_id=bag_id,
                instance_index=instance_idx,
                start_sec=start / sample_rate,
                end_sec=end / sample_rate,
                waveform=chunk,
                sample_rate=sample_rate,
            )
        )
        instance_idx += 1

    return windows
