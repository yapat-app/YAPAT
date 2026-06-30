"""Recursive BirdNET embedding extraction with audio preflight normalization and MP3 tail recovery.

Usage:
    python extract_birdnet_embeddings.py /path/to/dataset
    # or: WSSED_EXTRACT_DATASET_PATH=/path/to/dataset python extract_birdnet_embeddings.py

This script:

    - Recursively scans all audio files.
    - Detects mislabeled or problematic files, e.g. AAC stream with .mp3 extension.
    - Converts problematic files to standard WAV copies before extraction.
    - If MP3 extraction fails because of corrupt tail / illegal MPEG header,
      it trims the file before the decoder-reported bad offset and retries.
    - Does NOT modify original audio files.
    - Preserves original dataset folder structure in the embedding output folder.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import config
import embedding_io
import extract_embeddings as extract_embeddings_module
from birdnet_direct import BirdNetDirectExtractor
from job_hyperparameters import apply_extraction_hyperparameters, load_job_hyperparameters


def _resolve_dataset_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    path = os.environ.get("WSSED_EXTRACT_DATASET_PATH")
    if path:
        return path
    raise SystemExit(
        "Dataset path required.\n"
        "  python extract_birdnet_embeddings.py /path/to/dataset\n"
        "  WSSED_EXTRACT_DATASET_PATH=/path/to/dataset python extract_birdnet_embeddings.py"
    )


DATASET_PATH = _resolve_dataset_path()


AUDIO_EXTENSIONS = {
    ".wav",
    ".WAV",
    ".flac",
    ".FLAC",
    ".mp3",
    ".MP3",
    ".aac",
    ".AAC",
    ".m4a",
    ".M4A",
}

SKIP_EXISTING = True
CONTINUE_ON_ERROR = True

# Convert known problematic/mislabeled files to WAV before extraction.
NORMALIZE_PROBLEMATIC_AUDIO = True

NORMALIZED_SAMPLE_RATE = 48000
NORMALIZED_CHANNELS = 1

# MP3 corrupt-tail recovery.
ENABLE_MP3_TAIL_RECOVERY = True


DATASET_ROOT = Path(DATASET_PATH).resolve()

OUTPUT_ROOT = DATASET_ROOT.parent / f"{DATASET_ROOT.name}_birdnet_embeddings"

NORMALIZED_AUDIO_ROOT = (
    DATASET_ROOT.parent / f"{DATASET_ROOT.name}_normalized_audio_for_birdnet"
)

RECOVERED_AUDIO_ROOT = (
    DATASET_ROOT.parent / f"{DATASET_ROOT.name}_recovered_audio_for_birdnet"
)

FAILED_LOG_PATH = OUTPUT_ROOT / "failed_embeddings.txt"
NORMALIZATION_LOG_PATH = OUTPUT_ROOT / "audio_normalization_log.txt"
MP3_RECOVERY_LOG_PATH = OUTPUT_ROOT / "mp3_recovery_log.txt"


# Maps actual extraction source path -> original relative dataset path.
# This is required when BirdNET receives a normalized/recovered helper file,
# but the output .npz should still follow the original dataset structure.
EXTRACTION_SOURCE_TO_ORIGINAL_REL: Dict[Path, Path] = {}


@dataclass
class AudioProbeInfo:
    ok: bool
    codec_name: Optional[str]
    format_name: Optional[str]
    error: Optional[str] = None


@dataclass
class ExtractionJob:
    original_audio_path: Path
    extraction_audio_path: Path
    output_embedding_path: Path
    normalized: bool
    normalization_reason: str


@dataclass
class RecoveryResult:
    attempted: bool
    success: bool
    recovered_audio_path: Optional[Path]
    original_size: Optional[int]
    error_offset: Optional[int]
    trimmed_size: Optional[int]
    removed_tail_size: Optional[int]
    reason: str


def _is_audio_file(path: Path) -> bool:
    return path.is_file() and path.suffix in AUDIO_EXTENSIONS


def find_audio_files(dataset_root: Path) -> List[Path]:
    return sorted(path for path in dataset_root.rglob("*") if _is_audio_file(path))


def read_header(path: Path, n: int = 16) -> bytes:
    with path.open("rb") as file:
        return file.read(n)


def ffprobe_audio_info(path: Path) -> AudioProbeInfo:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=codec_name",
        "-show_entries",
        "format=format_name",
        "-of",
        "json",
        str(path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe was not found. Install ffmpeg/ffprobe and make sure it is in PATH."
        )

    if proc.returncode != 0:
        return AudioProbeInfo(
            ok=False,
            codec_name=None,
            format_name=None,
            error=proc.stderr.strip() or proc.stdout.strip(),
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return AudioProbeInfo(
            ok=False,
            codec_name=None,
            format_name=None,
            error=f"Could not parse ffprobe JSON output: {exc}",
        )

    streams = payload.get("streams", [])
    fmt = payload.get("format", {})

    codec_name = streams[0].get("codec_name") if streams else None
    format_name = fmt.get("format_name")

    return AudioProbeInfo(
        ok=True,
        codec_name=codec_name,
        format_name=format_name,
        error=None,
    )


def is_adts_aac_header(header: bytes) -> bool:
    return header.startswith(b"\xff\xf1") or header.startswith(b"\xff\xf9")


def should_normalize_audio(path: Path) -> tuple[bool, str]:
    suffix = path.suffix.lower()
    header = read_header(path, 16)
    probe = ffprobe_audio_info(path)

    if not probe.ok:
        return True, f"ffprobe_failed: {probe.error}"

    codec = (probe.codec_name or "").lower()
    fmt = (probe.format_name or "").lower()

    if suffix == ".mp3":
        if is_adts_aac_header(header):
            return True, "mp3_extension_but_adts_aac_header"

        if codec and codec != "mp3":
            return True, f"mp3_extension_but_codec_is_{codec}"

        if fmt and "mp3" not in fmt:
            return True, f"mp3_extension_but_container_is_{fmt}"

        return False, "mp3_ok"

    if suffix == ".aac":
        return True, "aac_input_normalized_to_wav"

    if suffix == ".m4a":
        return True, "m4a_input_normalized_to_wav"

    if suffix in {".wav", ".flac"}:
        return False, f"{suffix.lstrip('.')}_ok_codec_{codec or 'unknown'}"

    return True, f"unknown_extension_{suffix}"


def output_embedding_path_for_original_audio(
    original_audio_path: str | Path,
    output_root: str | Path,
) -> Path:
    audio = Path(original_audio_path).resolve()
    rel_path = audio.relative_to(DATASET_ROOT)
    return Path(output_root).resolve() / rel_path.with_suffix(".npz")


def normalized_audio_path_for_original_audio(original_audio_path: Path) -> Path:
    rel_path = original_audio_path.resolve().relative_to(DATASET_ROOT)
    source_ext = original_audio_path.suffix.lower().lstrip(".") or "noext"
    normalized_name = f"{original_audio_path.stem}__from_{source_ext}.wav"

    return NORMALIZED_AUDIO_ROOT / rel_path.parent / normalized_name


def recovered_mp3_path_for_original_audio(original_audio_path: Path, offset: int) -> Path:
    rel_path = original_audio_path.resolve().relative_to(DATASET_ROOT)
    recovered_name = f"{original_audio_path.stem}_trimmed_at_{offset}.mp3"
    return RECOVERED_AUDIO_ROOT / rel_path.parent / recovered_name


def embedding_cache_path_for_extraction_source(
    audio_path: str | Path,
    cache_dir: str | Path,
) -> Path:
    resolved = Path(audio_path).resolve()

    if resolved in EXTRACTION_SOURCE_TO_ORIGINAL_REL:
        original_rel = EXTRACTION_SOURCE_TO_ORIGINAL_REL[resolved]
        return OUTPUT_ROOT / original_rel.with_suffix(".npz")

    audio = resolved
    rel_path = audio.relative_to(DATASET_ROOT)
    return OUTPUT_ROOT / rel_path.with_suffix(".npz")


def patch_embedding_cache_path() -> None:
    embedding_io.embedding_cache_path = embedding_cache_path_for_extraction_source

    if hasattr(extract_embeddings_module, "embedding_cache_path"):
        extract_embeddings_module.embedding_cache_path = embedding_cache_path_for_extraction_source


def convert_to_standard_wav(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(NORMALIZED_CHANNELS),
        "-ar",
        str(NORMALIZED_SAMPLE_RATE),
        "-f",
        "wav",
        str(output_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg and make sure it is in PATH."
        )

    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg conversion failed.\n"
            f"Input: {input_path}\n"
            f"Output: {output_path}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            "ffmpeg finished without error, but output WAV was not created.\n"
            f"Input: {input_path}\n"
            f"Expected output: {output_path}"
        )


def run_torchaudio_load_probe(path: Path) -> subprocess.CompletedProcess:
    """Run torchaudio.load in a subprocess so decoder stderr can be captured.

    Some MPEG decoder messages are emitted by native libraries and are not
    reliably capturable inside the same Python process. A subprocess gives us
    the full stderr text, including messages like:

        Illegal Audio-MPEG-Header ... at offset 824640
    """
    code = (
        "import sys, torchaudio\n"
        "path = sys.argv[1]\n"
        "waveform, sr = torchaudio.load(path)\n"
        "print(tuple(waveform.shape), sr)\n"
    )

    return subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def parse_decoder_error_offset(text: str) -> Optional[int]:
    """Parse bad MPEG offset from decoder stderr/stdout."""
    patterns = [
        r"Illegal Audio-MPEG-Header .*? at offset (\d+)",
        r"offset\s+(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return int(match.group(1))

    return None


def is_recoverable_mp3_decoder_error(text: str) -> bool:
    markers = [
        "Illegal Audio-MPEG-Header",
        "Giving up resync",
        "Header missing",
        "Invalid data found when processing input",
        "Unspecified internal error",
        "your stream is not nice",
    ]

    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def trim_mp3_before_offset(original_audio_path: Path, offset: int) -> Path:
    original_size = original_audio_path.stat().st_size

    if offset <= 0 or offset >= original_size:
        raise RuntimeError(
            f"Invalid recovery offset={offset} for file size={original_size}: "
            f"{original_audio_path}"
        )

    recovered_path = recovered_mp3_path_for_original_audio(original_audio_path, offset)
    recovered_path.parent.mkdir(parents=True, exist_ok=True)

    data = original_audio_path.read_bytes()
    recovered_path.write_bytes(data[:offset])

    return recovered_path


def try_mp3_tail_recovery(original_audio_path: Path) -> RecoveryResult:
    """Try corrupt-tail recovery for MP3 files.

    Steps:
        1. Run torchaudio.load in subprocess.
        2. Parse bad decoder offset.
        3. Write data[:offset] to a trimmed MP3.
        4. Run torchaudio.load on the trimmed MP3.
    """
    if not ENABLE_MP3_TAIL_RECOVERY:
        return RecoveryResult(
            attempted=False,
            success=False,
            recovered_audio_path=None,
            original_size=None,
            error_offset=None,
            trimmed_size=None,
            removed_tail_size=None,
            reason="mp3_tail_recovery_disabled",
        )

    if original_audio_path.suffix.lower() != ".mp3":
        return RecoveryResult(
            attempted=False,
            success=False,
            recovered_audio_path=None,
            original_size=None,
            error_offset=None,
            trimmed_size=None,
            removed_tail_size=None,
            reason="not_an_mp3_file",
        )

    original_size = original_audio_path.stat().st_size

    probe = run_torchaudio_load_probe(original_audio_path)
    probe_text = "\n".join(
        part for part in [probe.stdout, probe.stderr] if part
    )

    if probe.returncode == 0:
        return RecoveryResult(
            attempted=True,
            success=False,
            recovered_audio_path=None,
            original_size=original_size,
            error_offset=None,
            trimmed_size=None,
            removed_tail_size=None,
            reason="original_mp3_loaded_successfully_in_recovery_probe_no_trim_needed",
        )

    if not is_recoverable_mp3_decoder_error(probe_text):
        return RecoveryResult(
            attempted=True,
            success=False,
            recovered_audio_path=None,
            original_size=original_size,
            error_offset=None,
            trimmed_size=None,
            removed_tail_size=None,
            reason="decoder_error_not_recognized_as_recoverable",
        )

    error_offset = parse_decoder_error_offset(probe_text)

    if error_offset is None:
        return RecoveryResult(
            attempted=True,
            success=False,
            recovered_audio_path=None,
            original_size=original_size,
            error_offset=None,
            trimmed_size=None,
            removed_tail_size=None,
            reason="could_not_parse_decoder_error_offset",
        )

    try:
        recovered_path = trim_mp3_before_offset(original_audio_path, error_offset)
    except Exception as exc:
        return RecoveryResult(
            attempted=True,
            success=False,
            recovered_audio_path=None,
            original_size=original_size,
            error_offset=error_offset,
            trimmed_size=None,
            removed_tail_size=None,
            reason=f"failed_to_write_trimmed_mp3: {exc}",
        )

    trimmed_size = recovered_path.stat().st_size
    removed_tail_size = original_size - trimmed_size

    retry_probe = run_torchaudio_load_probe(recovered_path)
    retry_text = "\n".join(
        part for part in [retry_probe.stdout, retry_probe.stderr] if part
    )

    if retry_probe.returncode != 0:
        return RecoveryResult(
            attempted=True,
            success=False,
            recovered_audio_path=recovered_path,
            original_size=original_size,
            error_offset=error_offset,
            trimmed_size=trimmed_size,
            removed_tail_size=removed_tail_size,
            reason=f"trimmed_mp3_still_failed_to_decode: {retry_text}",
        )

    return RecoveryResult(
        attempted=True,
        success=True,
        recovered_audio_path=recovered_path,
        original_size=original_size,
        error_offset=error_offset,
        trimmed_size=trimmed_size,
        removed_tail_size=removed_tail_size,
        reason=(
            "MP3 decoder found an illegal MPEG header near this offset. "
            "The file was trimmed before the problematic corrupt MP3 tail "
            "and decoding was retried successfully."
        ),
    )


def check_output_collisions(audio_files: List[Path]) -> None:
    by_output: Dict[Path, List[Path]] = {}

    for audio in audio_files:
        out_path = output_embedding_path_for_original_audio(audio, OUTPUT_ROOT)
        by_output.setdefault(out_path, []).append(audio)

    collisions = {
        out_path: sources
        for out_path, sources in by_output.items()
        if len(sources) > 1
    }

    if not collisions:
        return

    lines = [
        "Output filename collision detected.",
        "Two or more audio files would produce the same .npz file.",
        "Rename one of the source files or adjust the output naming rule.",
        "",
    ]

    for out_path, sources in collisions.items():
        lines.append(f"Output: {out_path}")
        for src in sources:
            lines.append(f"  - {src}")
        lines.append("")

    raise RuntimeError("\n".join(lines))


def write_log_headers() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for log_path in [FAILED_LOG_PATH, NORMALIZATION_LOG_PATH, MP3_RECOVERY_LOG_PATH]:
        if log_path.exists():
            log_path.unlink()

    with FAILED_LOG_PATH.open("w", encoding="utf-8") as file:
        file.write("Failed BirdNET embedding extraction files\n")
        file.write("=" * 80 + "\n\n")

    with NORMALIZATION_LOG_PATH.open("w", encoding="utf-8") as file:
        file.write("Audio normalization log\n")
        file.write("=" * 80 + "\n\n")

    with MP3_RECOVERY_LOG_PATH.open("w", encoding="utf-8") as file:
        file.write("MP3 corrupt-tail recovery log\n")
        file.write("=" * 80 + "\n\n")


def log_normalization(
    original_audio: Path,
    normalized_audio: Path,
    reason: str,
    action: str,
) -> None:
    with NORMALIZATION_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(f"Original: {original_audio.relative_to(DATASET_ROOT)}\n")
        file.write(f"Normalized: {normalized_audio.relative_to(NORMALIZED_AUDIO_ROOT)}\n")
        file.write(f"Reason: {reason}\n")
        file.write(f"Action: {action}\n")
        file.write("-" * 80 + "\n\n")


def log_mp3_recovery(
    original_audio: Path,
    recovery: RecoveryResult,
) -> None:
    with MP3_RECOVERY_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(f"Original file path: {original_audio}\n")
        file.write(f"Original relative path: {original_audio.relative_to(DATASET_ROOT)}\n")
        file.write(f"Recovery attempted: {recovery.attempted}\n")
        file.write(f"Recovery successful: {recovery.success}\n")

        if recovery.original_size is not None:
            file.write(f"Original MP3 size: {recovery.original_size} bytes\n")

        if recovery.error_offset is not None:
            file.write(f"Decoder failed near offset: {recovery.error_offset} bytes\n")

        if recovery.recovered_audio_path is not None:
            file.write(f"Trimmed file path: {recovery.recovered_audio_path}\n")

        if recovery.trimmed_size is not None:
            file.write(f"Trimmed file size: {recovery.trimmed_size} bytes\n")

        if recovery.removed_tail_size is not None:
            file.write(f"Removed tail size: {recovery.removed_tail_size} bytes\n")
            file.write(
                "Removed tail explanation: This removed tail is treated as the "
                "problematic/corrupt MP3 tail region detected near the decoder offset.\n"
            )

        file.write(f"Reason: {recovery.reason}\n")
        file.write("-" * 80 + "\n\n")


def log_failed_file(
    original_audio_path: Path,
    extraction_audio_path: Path,
    out_path: Path,
    exc: Exception,
) -> None:
    with FAILED_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(f"Original audio: {original_audio_path.relative_to(DATASET_ROOT)}\n")
        file.write(f"Extraction audio: {extraction_audio_path}\n")
        file.write(f"Expected output: {out_path.relative_to(OUTPUT_ROOT)}\n")
        file.write(f"Error type: {type(exc).__name__}\n")
        file.write(f"Error message: {exc}\n")
        file.write("Traceback:\n")
        file.write(traceback.format_exc())
        file.write("\n" + "-" * 80 + "\n\n")


def build_extraction_jobs(audio_files: List[Path]) -> List[ExtractionJob]:
    jobs: List[ExtractionJob] = []

    for index, original_audio in enumerate(audio_files, start=1):
        output_npz = output_embedding_path_for_original_audio(original_audio, OUTPUT_ROOT)

        if SKIP_EXISTING and output_npz.exists():
            extraction_audio = original_audio
            normalized = False
            reason = "embedding_exists_skip_preflight"
        else:
            normalize, reason = should_normalize_audio(original_audio)

            if NORMALIZE_PROBLEMATIC_AUDIO and normalize:
                normalized_audio = normalized_audio_path_for_original_audio(original_audio)

                if normalized_audio.exists():
                    action = "normalized_wav_already_exists"
                else:
                    action = "converted_to_standard_wav"
                    convert_to_standard_wav(original_audio, normalized_audio)

                log_normalization(
                    original_audio=original_audio,
                    normalized_audio=normalized_audio,
                    reason=reason,
                    action=action,
                )

                extraction_audio = normalized_audio
                normalized = True
            else:
                extraction_audio = original_audio
                normalized = False

        original_rel = original_audio.resolve().relative_to(DATASET_ROOT)
        EXTRACTION_SOURCE_TO_ORIGINAL_REL[extraction_audio.resolve()] = original_rel

        jobs.append(
            ExtractionJob(
                original_audio_path=original_audio,
                extraction_audio_path=extraction_audio,
                output_embedding_path=output_npz,
                normalized=normalized,
                normalization_reason=reason,
            )
        )

        rel = original_audio.relative_to(DATASET_ROOT)
        if normalized:
            print(f"[preflight {index}/{len(audio_files)}] normalize: {rel} ({reason})")
        else:
            print(f"[preflight {index}/{len(audio_files)}] ok: {rel} ({reason})")

    return jobs


def run_embedding_extraction_for_source(
    extraction_audio_path: Path,
    output_root: Path,
    extractor: BirdNetDirectExtractor,
) -> None:
    extract_embeddings_module.extract_for_audio(
        extraction_audio_path,
        output_root,
        extractor,
    )


def extract_one_job(
    job: ExtractionJob,
    extractor: BirdNetDirectExtractor,
) -> bool:
    job.output_embedding_path.parent.mkdir(parents=True, exist_ok=True)

    rel_original = job.original_audio_path.relative_to(DATASET_ROOT)
    rel_output = job.output_embedding_path.relative_to(OUTPUT_ROOT)

    print(f"    original   : {rel_original}")

    if job.normalized:
        print(f"    extraction : {job.extraction_audio_path}")
        print(f"    reason     : {job.normalization_reason}")

    print(f"    output     : {rel_output}")

    if SKIP_EXISTING and job.output_embedding_path.exists():
        print("    skip: embedding already exists")
        return True

    try:
        run_embedding_extraction_for_source(
            extraction_audio_path=job.extraction_audio_path,
            output_root=OUTPUT_ROOT,
            extractor=extractor,
        )

        if not job.output_embedding_path.exists():
            raise RuntimeError(
                "Embedding extraction finished, but expected output was not created.\n"
                f"Original audio: {job.original_audio_path}\n"
                f"Extraction audio: {job.extraction_audio_path}\n"
                f"Expected output: {job.output_embedding_path}"
            )

        print("    ok")
        return True

    except Exception as first_exc:
        print(f"    initial extraction FAILED: {type(first_exc).__name__}: {first_exc}")

        # Recovery should only run for real .mp3 sources that were not already
        # normalized to WAV. This keeps normal files unchanged.
        if job.original_audio_path.suffix.lower() == ".mp3":
            print("    attempting MP3 corrupt-tail recovery...")

            recovery = try_mp3_tail_recovery(job.original_audio_path)
            log_mp3_recovery(job.original_audio_path, recovery)

            if recovery.success and recovery.recovered_audio_path is not None:
                try:
                    recovered_audio = recovery.recovered_audio_path.resolve()

                    original_rel = job.original_audio_path.resolve().relative_to(DATASET_ROOT)
                    EXTRACTION_SOURCE_TO_ORIGINAL_REL[recovered_audio] = original_rel

                    print(f"    recovery ok: {recovered_audio}")
                    print(
                        f"    original size={recovery.original_size} bytes, "
                        f"offset={recovery.error_offset} bytes, "
                        f"removed tail={recovery.removed_tail_size} bytes"
                    )

                    run_embedding_extraction_for_source(
                        extraction_audio_path=recovered_audio,
                        output_root=OUTPUT_ROOT,
                        extractor=extractor,
                    )

                    if not job.output_embedding_path.exists():
                        raise RuntimeError(
                            "Recovered extraction finished, but expected output was not created.\n"
                            f"Original audio: {job.original_audio_path}\n"
                            f"Recovered audio: {recovered_audio}\n"
                            f"Expected output: {job.output_embedding_path}"
                        )

                    print("    ok after MP3 recovery")
                    return True

                except Exception as recovery_exc:
                    print(
                        f"    recovery extraction FAILED: "
                        f"{type(recovery_exc).__name__}: {recovery_exc}"
                    )
                    log_failed_file(
                        original_audio_path=job.original_audio_path,
                        extraction_audio_path=job.extraction_audio_path,
                        out_path=job.output_embedding_path,
                        exc=recovery_exc,
                    )

                    if CONTINUE_ON_ERROR:
                        return False

                    raise

            print(f"    recovery failed: {recovery.reason}")

        log_failed_file(
            original_audio_path=job.original_audio_path,
            extraction_audio_path=job.extraction_audio_path,
            out_path=job.output_embedding_path,
            exc=first_exc,
        )

        if CONTINUE_ON_ERROR:
            return False

        raise


def extract_all_embeddings() -> None:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {DATASET_ROOT}")

    if not DATASET_ROOT.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {DATASET_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    NORMALIZED_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    RECOVERED_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)

    patch_embedding_cache_path()

    audio_files = find_audio_files(DATASET_ROOT)

    if not audio_files:
        print(f"No audio files found under: {DATASET_ROOT}")
        return

    check_output_collisions(audio_files)
    write_log_headers()

    print("=" * 80)
    print("BirdNET embedding extraction with audio preflight and MP3 recovery")
    print("=" * 80)
    print(f"Dataset path          : {DATASET_ROOT}")
    print(f"Embedding output path : {OUTPUT_ROOT}")
    print(f"Normalized audio path : {NORMALIZED_AUDIO_ROOT}")
    print(f"Recovered audio path  : {RECOVERED_AUDIO_ROOT}")
    print(f"Failed log            : {FAILED_LOG_PATH}")
    print(f"Normalization log     : {NORMALIZATION_LOG_PATH}")
    print(f"MP3 recovery log      : {MP3_RECOVERY_LOG_PATH}")
    print(f"Audio files           : {len(audio_files)}")
    print("=" * 80)

    job_hparams = load_job_hyperparameters()
    applied = apply_extraction_hyperparameters(job_hparams)
    if applied:
        print("[config] extraction hyperparameters from job request:")
        for entry in applied:
            print(f"  {entry}")
        print(
            f"[config] WINDOW_SECONDS={config.WINDOW_SECONDS} "
            f"HOP_SECONDS={config.HOP_SECONDS} "
            f"BIRDNET_SR={config.BIRDNET_SR}"
        )
        print("=" * 80)

    print("\nPreflight: checking headers/codecs and normalizing problematic files...")
    jobs = build_extraction_jobs(audio_files)

    normalized_count = sum(1 for job in jobs if job.normalized)

    print("=" * 80)
    print("Preflight completed.")
    print(f"Total files          : {len(jobs)}")
    print(f"Normalized files     : {normalized_count}")
    print(f"Direct extraction    : {len(jobs) - normalized_count}")
    print("=" * 80)

    extractor = BirdNetDirectExtractor()

    success_count = 0
    failed_count = 0
    skipped_existing_count = 0

    print("\nStarting BirdNET embedding extraction...")

    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}]")

        existed_before = job.output_embedding_path.exists()

        ok = extract_one_job(
            job=job,
            extractor=extractor,
        )

        if ok:
            if existed_before and SKIP_EXISTING:
                skipped_existing_count += 1
            else:
                success_count += 1
        else:
            failed_count += 1

    print("=" * 80)
    print("Extraction completed.")
    print(f"Embeddings saved to  : {OUTPUT_ROOT}")
    print(f"Normalized audio path: {NORMALIZED_AUDIO_ROOT}")
    print(f"Recovered audio path : {RECOVERED_AUDIO_ROOT}")
    print(f"Successfully created : {success_count}")
    print(f"Skipped existing     : {skipped_existing_count}")
    print(f"Failed files         : {failed_count}")
    print(f"Failed log           : {FAILED_LOG_PATH}")
    print(f"Normalization log    : {NORMALIZATION_LOG_PATH}")
    print(f"MP3 recovery log     : {MP3_RECOVERY_LOG_PATH}")
    print("=" * 80)

    if failed_count > 0:
        print("")
        print("Some files failed. Check failed_embeddings.txt for details.")
        print("Training can still continue with successfully extracted .npz files.")


def main() -> None:
    extract_all_embeddings()


if __name__ == "__main__":
    main()