"""Offline BirdNET embedding extraction with caching (direct Python API, no DB/CLI)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

import config
from birdnet_direct import BirdNetDirectExtractor, smoke_check_single_wav
from embedding_io import embedding_cache_path, save_segment_embeddings
from preprocessing import build_birdnet_segment_batch


def iter_wavs(audio_dir: Path) -> Iterable[Path]:
    for p in sorted(audio_dir.rglob("*.wav")):
        yield p


def extract_for_audio(audio_path: Path, output_dir: Path, extractor: BirdNetDirectExtractor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = embedding_cache_path(str(audio_path), str(output_dir))
    if cache_path.exists():
        data = np.load(cache_path)
        embeddings = data["embeddings"]
        starts = data["starts"]
        ends = data["ends"]
        print(f"[extract] cache hit: {cache_path} shape={embeddings.shape}")
        return embeddings, starts, ends

    sample_batch, starts_tensor, ends_tensor = build_birdnet_segment_batch(
        audio_path=str(audio_path),
        window_seconds=config.WINDOW_SECONDS,
        hop_seconds=config.HOP_SECONDS,
        birdnet_sample_rate=config.BIRDNET_SR,
    )
    starts = starts_tensor.detach().cpu().numpy().astype(np.float32)
    ends = ends_tensor.detach().cpu().numpy().astype(np.float32)
    batch_np = sample_batch.detach().cpu().numpy().astype(np.float32)

    matrix = extractor.embed_batch(batch_np)
    if matrix.ndim != 2:
        raise RuntimeError(f"Unexpected embedding rank: {matrix.shape}")
    if matrix.shape[0] != batch_np.shape[0]:
        raise RuntimeError(f"Segment count mismatch: batch={batch_np.shape[0]} embeddings={matrix.shape[0]}")

    if not hasattr(config, "EXPECTED_EMBED_DIM"):
        setattr(config, "EXPECTED_EMBED_DIM", int(matrix.shape[1]))
    print(f"[extract] embedding_dim={matrix.shape[1]}")

    save_segment_embeddings(cache_path, matrix, starts, ends)
    print(
        f"[extract] cache miss -> wrote {cache_path} | shape={matrix.shape} "
        f"| window={config.WINDOW_SECONDS}s hop={config.HOP_SECONDS}s"
    )
    return matrix, starts, ends


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default="")
    parser.add_argument("--single-wav", default="")
    parser.add_argument("--output-dir", default=config.BIRDNET_CACHE_DIR)
    parser.add_argument("--smoke-wav", default="")
    args = parser.parse_args()

    extractor = BirdNetDirectExtractor()

    print(f"[extract] audio_dir={args.audio_dir}")
    print(f"[extract] single_wav={args.single_wav}")
    print(f"[extract] output_dir={args.output_dir}")

    if args.single_wav.strip():
        wav_path = Path(args.single_wav.strip())
        if not wav_path.exists():
            raise RuntimeError(f"single wav does not exist: {wav_path}")
        extract_for_audio(wav_path, Path(args.output_dir), extractor)
        smoke_target = args.smoke_wav.strip() or str(wav_path)
        smoke_check_single_wav(
            smoke_target,
            args.output_dir,
            lambda wav, cache: extract_for_audio(Path(wav), Path(cache), extractor),
        )
        return

    if not args.audio_dir.strip():
        raise RuntimeError("Either --single-wav or --audio-dir must be provided")

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise RuntimeError(f"audio_dir does not exist: {audio_dir}")

    wavs = list(iter_wavs(audio_dir))
    if not wavs:
        raise RuntimeError(f"No wav files found in {audio_dir}")

    for wav_path in wavs:
        extract_for_audio(wav_path, Path(args.output_dir), extractor)

    smoke_target = args.smoke_wav.strip() or str(wavs[0])
    smoke_check_single_wav(
        smoke_target,
        args.output_dir,
        lambda wav, cache: extract_for_audio(Path(wav), Path(cache), extractor),
    )


if __name__ == "__main__":
    main()
