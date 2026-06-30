"""FNJV + AnuraSet dataset loading and split utilities for transfer-learning workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import config


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    species_codes: List[str]
    train_root: Path
    val_root: Path | None
    test_root: Path | None


def _resolve_fnjv_root(dataset_name: str) -> Tuple[Path, List[str]]:
    if dataset_name == "FNJV_458":
        return Path(config.FNJV_458_ROOT), list(config.TARGET_SPECIES_458)
    if dataset_name == "FNJV_578":
        return Path(config.FNJV_578_ROOT), list(config.TARGET_SPECIES_578)
    raise ValueError(f"Unsupported FNJV dataset name: {dataset_name}")


def _resolve_anuraset_targets_from_config() -> Tuple[List[str], Path]:
    names = [
        str(getattr(config, "DATASET_TRAIN", "")).upper(),
        str(getattr(config, "DATASET_VAL", "")).upper(),
        str(getattr(config, "DATASET_TEST", "")).upper(),
    ]
    has_458 = any("458" in n for n in names)
    has_578 = any("578" in n for n in names)
    if has_458 and not has_578:
        return list(config.TARGET_SPECIES_458), Path(config.STRONG_LABELS_458)
    # default to 578 overlap set
    return list(config.TARGET_SPECIES_578), Path(config.STRONG_LABELS_578)


def _sample_negative_rows(df_non_target: pd.DataFrame) -> pd.DataFrame:
    if not config.USE_NEGATIVE_SAMPLES or df_non_target.empty:
        return pd.DataFrame(columns=df_non_target.columns)
    frac = max(0.0, min(1.0, float(config.NEGATIVE_SAMPLE_RATIO)))
    if frac == 0.0:
        return pd.DataFrame(columns=df_non_target.columns)
    return df_non_target.sample(frac=frac, random_state=config.SEED).reset_index(drop=True)


def _class_stratified_fraction(df: pd.DataFrame, ratio: float, purpose: str) -> pd.DataFrame:
    ratio = float(ratio)
    if df.empty or ratio <= 0:
        return df.head(0).copy()
    if ratio >= 1:
        out = df.copy()
    else:
        pieces = []
        for label, grp in df.groupby("label_code"):
            n = max(1, int(len(grp) * ratio))
            n = min(n, len(grp))
            pieces.append(grp.sample(n=n, random_state=config.SEED))
        out = pd.concat(pieces, ignore_index=True) if pieces else df.head(0).copy()
    _log_split_counts(purpose, out)
    return out.reset_index(drop=True)


def _log_split_counts(name: str, df: pd.DataFrame) -> None:
    counts = df["label_code"].value_counts().to_dict() if not df.empty and "label_code" in df.columns else {}
    print(f"SPLIT_COUNTS {name}: total={len(df)} per_class={counts}")


def _prepare_fnjv_dataframe(
    dataset_root: Path,
    target_species: List[str],
    include_negatives: bool = False,
) -> pd.DataFrame:
    metadata_path = dataset_root / "metadata_filtered_filled.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"FNJV metadata file was not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    required_columns = {"Arquivo do registro", "Code"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"FNJV metadata is missing columns: {missing}")

    df = df[df["Code"].notna()].copy()
    df["Code"] = df["Code"].astype(str).str.strip()
    df = df[df["Code"] != "IGNORE"]
    df["audio_path"] = df["Arquivo do registro"].apply(lambda x: str(dataset_root / str(x)))
    df = df[df["audio_path"].apply(lambda x: Path(x).exists())]

    df_target = df[df["Code"].isin(target_species)].copy()
    df_target["label_code"] = df_target["Code"]
    df_target["is_negative"] = False

    df_non_target = df[~df["Code"].isin(target_species)].copy()
    if include_negatives:
        df_neg = _sample_negative_rows(df_non_target)
        if not df_neg.empty:
            df_neg["label_code"] = ""
            df_neg["is_negative"] = True
    else:
        df_neg = pd.DataFrame(columns=df.columns.tolist() + ["label_code", "is_negative"])

    out = pd.concat([df_target, df_neg], ignore_index=True)
    if out.empty:
        raise ValueError("FNJV dataframe is empty after filtering.")

    out["source_dataset"] = "FNJV"
    out["strong_events"] = [[] for _ in range(len(out))]
    return out.reset_index(drop=True)


def resolve_strong_label_path(file_path: str, anuraset_root: str, anuraset_strong_root: str) -> str:
    if str(anuraset_root).startswith("/ds-iml/"):
        resolved = str(file_path)
    else:
        marker = "/strong_labels/"
        normalized = str(file_path).replace("\\", "/")
        if marker in normalized:
            suffix = normalized.split(marker, 1)[1]
            resolved = str(Path(anuraset_strong_root) / Path(suffix))
        else:
            resolved = str(Path(anuraset_strong_root) / Path(normalized).name)
    print(f"[anuraset] resolve_strong_label_path: {file_path} -> {resolved}")
    return resolved


def resolve_wav_path_from_file_name(file_name: str, anuraset_root: str) -> str:
    stem = Path(str(file_name)).stem
    site = stem.split("_", 1)[0]
    wav_name = stem + ".wav"
    return str(Path(anuraset_root) / site / wav_name)


def _prepare_anuraset_dataframe(anuraset_root: Path, strong_root: Path, strong_csv_path: Path, target_species: List[str]) -> pd.DataFrame:
    if not strong_csv_path.exists():
        raise FileNotFoundError(f"AnuraSet strong-label CSV not found: {strong_csv_path}")

    meta = pd.read_csv(strong_csv_path)
    required = {"file_name", "file_path", "start_second", "end_second", "label", "subset"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"AnuraSet strong-label CSV missing columns: {missing}")

    rows = []
    for file_name, grp in meta.groupby("file_name"):
        first_file_path = str(grp.iloc[0]["file_path"])
        resolved_txt = resolve_strong_label_path(first_file_path, str(anuraset_root), str(strong_root))
        wav_path = resolve_wav_path_from_file_name(str(file_name), str(anuraset_root))
        if not Path(wav_path).exists():
            continue

        subset_vals = set(str(x).strip().lower() for x in grp["subset"].dropna().tolist())
        subset = "train" if "train" in subset_vals else ("test" if "test" in subset_vals else "train")

        target_grp = grp[grp["label"].isin(target_species)].copy()
        events = [(float(r.start_second), float(r.end_second), str(r.label)) for r in target_grp.itertuples(index=False)]

        if events:
            label_code = str(events[0][2])
            is_negative = False
        else:
            if not config.USE_NEGATIVE_SAMPLES:
                continue
            label_code = ""
            is_negative = True

        rows.append(
            {
                "file_name": str(file_name),
                "resolved_strong_path": resolved_txt,
                "audio_path": wav_path,
                "Code": label_code,
                "label_code": label_code,
                "source_dataset": "ANURASET",
                "strong_events": events,
                "subset": subset,
                "is_negative": is_negative,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("AnuraSet dataframe is empty after resolving strong labels and wav paths.")

    first20 = out["file_name"].drop_duplicates().head(20)
    check = out[out["file_name"].isin(first20)]
    txt_ok = check["resolved_strong_path"].apply(lambda x: Path(str(x)).exists()).all()
    wav_ok = check["audio_path"].apply(lambda x: Path(str(x)).exists()).all()
    print(f"[anuraset] first20 strong path exists={txt_ok} wav path exists={wav_ok}")
    if not txt_ok or not wav_ok:
        raise RuntimeError("FAIL: AnuraSet path resolution check failed for first 20 files")

    if config.USE_NEGATIVE_SAMPLES:
        pos = out[out["is_negative"] == False]
        neg = out[out["is_negative"] == True]

        # If the provided strong-label CSV is already species-filtered, explicit negative rows can be empty.
        # In that case, mine pseudo-negative wavs from files not present in the CSV.
        if neg.empty:
            known_stems = set(Path(str(x)).stem for x in out["file_name"].tolist())
            extra_rows = []
            for wav_path in sorted(anuraset_root.rglob("*.wav")):
                stem = wav_path.stem
                if stem in known_stems:
                    continue
                subset = "test" if (abs(hash(stem)) % 5 == 0) else "train"
                extra_rows.append(
                    {
                        "file_name": stem,
                        "resolved_strong_path": "",
                        "audio_path": str(wav_path),
                        "Code": "",
                        "label_code": "",
                        "source_dataset": "ANURASET",
                        "strong_events": [],
                        "subset": subset,
                        "is_negative": True,
                    }
                )
            if extra_rows:
                neg = pd.DataFrame(extra_rows)
                print(f"[anuraset] mined pseudo-negatives from raw wavs: {len(neg)}")
            else:
                print("[anuraset] WARNING: no explicit or mined negatives found")

        # sample negatives per subset so both validation/test pools can retain negatives
        neg_parts = []
        for _subset_name, grp in neg.groupby("subset"):
            neg_parts.append(_sample_negative_rows(grp))
        neg_sampled = pd.concat(neg_parts, ignore_index=True) if neg_parts else neg.head(0).copy()

        out = pd.concat([pos, neg_sampled], ignore_index=True)
    else:
        out = out[out["is_negative"] == False]

    neg_total = int((out["is_negative"] == True).sum()) if "is_negative" in out.columns else 0
    print(f"[anuraset] negatives after sampling: {neg_total}/{len(out)}")
    return out.reset_index(drop=True)


def _split_anuraset_by_subset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_pool = df[df["subset"].str.lower() == "train"].reset_index(drop=True)
    test_pool = df[df["subset"].str.lower() == "test"].reset_index(drop=True)
    if test_pool.empty:
        train_pool, test_pool = train_test_split(train_pool, test_size=config.TEST_SPLIT, random_state=config.SEED, shuffle=True)

    print(f"[anuraset] subset counts: train_pool={len(train_pool)} test_pool={len(test_pool)}")
    _log_split_counts("anuraset_train_pool", train_pool)
    _log_split_counts("anuraset_test_pool", test_pool)
    return train_pool.reset_index(drop=True), test_pool.reset_index(drop=True)


def _split_validation_from_train_pool(train_pool: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(train_pool) < 2:
        return train_pool.copy(), pd.DataFrame(columns=train_pool.columns)

    train_part, val_part = train_test_split(
        train_pool,
        test_size=config.VALIDATION_SPLIT,
        random_state=config.SEED,
        shuffle=True,
    )
    inter = set(train_part["file_name"]).intersection(set(val_part["file_name"]))
    if inter:
        raise RuntimeError("FAIL: train/val split overlap detected")

    print(f"[anuraset] validation split from train pool only: train_part={len(train_part)} val_part={len(val_part)} overlap={len(inter)}")
    _log_split_counts("anuraset_train_part", train_part)
    _log_split_counts("anuraset_val_part", val_part)
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def _determinism_check(train_pool: pd.DataFrame) -> None:
    if len(train_pool) < 2:
        return
    _, a_val = train_test_split(train_pool, test_size=config.VALIDATION_SPLIT, random_state=config.SEED, shuffle=True)
    _, b_val = train_test_split(train_pool, test_size=config.VALIDATION_SPLIT, random_state=config.SEED, shuffle=True)
    a_top = a_val["file_name"].head(10).tolist()
    b_top = b_val["file_name"].head(10).tolist()
    print(f"[anuraset] determinism top10 runA={a_top}")
    print(f"[anuraset] determinism top10 runB={b_top}")
    if a_top != b_top:
        raise RuntimeError("FAIL: deterministic validation split check failed")


def load_datasets() -> DatasetBundle:
    train_name = str(config.DATASET_TRAIN).upper()
    val_name = str(config.DATASET_VAL).upper()
    test_name = str(config.DATASET_TEST).upper()

    anura_root: Path | None = None
    strong_root: Path | None = None
    anura_train_pool: pd.DataFrame | None = None
    anura_test_pool: pd.DataFrame | None = None

    if train_name.startswith("FNJV"):
        train_root, train_species = _resolve_fnjv_root(config.DATASET_TRAIN)
        # Keep FNJV/source training strictly positive-target only; negatives are for val/test (AnuraSet) usage.
        train_df = _prepare_fnjv_dataframe(train_root, train_species, include_negatives=False)
    elif train_name == "ANURASET":
        train_species, strong_csv = _resolve_anuraset_targets_from_config()
        anura_root = Path(config.ANURASET_ROOT)
        strong_root = Path(config.ANURASET_STRONG_ROOT)
        anura_df = _prepare_anuraset_dataframe(anura_root, strong_root, strong_csv, train_species)
        anura_train_pool, anura_test_pool = _split_anuraset_by_subset(anura_df)
        train_df = anura_train_pool.copy().reset_index(drop=True)
        train_root = anura_root
    else:
        raise ValueError(f"Unsupported training dataset name: {config.DATASET_TRAIN}")

    if config.APPLY_TRAIN_SPLIT:
        train_df = _class_stratified_fraction(train_df, config.TRAIN_SPLIT, "train")
    else:
        _log_split_counts("train", train_df)

    val_df = pd.DataFrame(columns=train_df.columns)
    test_df = pd.DataFrame(columns=train_df.columns)
    val_root: Path | None = None
    test_root: Path | None = None

    if val_name == "ANURASET" or test_name == "ANURASET":
        if anura_root is None or strong_root is None or anura_train_pool is None or anura_test_pool is None:
            anura_root = Path(config.ANURASET_ROOT)
            strong_root = Path(config.ANURASET_STRONG_ROOT)
            train_species_for_anura, strong_csv = _resolve_anuraset_targets_from_config()
            anura_df = _prepare_anuraset_dataframe(anura_root, strong_root, strong_csv, train_species_for_anura)
            anura_train_pool, anura_test_pool = _split_anuraset_by_subset(anura_df)

        train_pool = anura_train_pool.copy()
        test_pool = anura_test_pool.copy()

        if config.APPLY_TEST_SPLIT:
            test_pool = _class_stratified_fraction(test_pool, config.TEST_SPLIT, "test")
        else:
            _log_split_counts("test", test_pool)

        if config.APPLY_VALIDATION_SPLIT:
            _, val_pool = _split_validation_from_train_pool(train_pool)
            val_pool = _class_stratified_fraction(val_pool, 1.0, "val")
        else:
            _, val_pool = _split_validation_from_train_pool(train_pool)

        _determinism_check(train_pool)

        if val_name == "ANURASET":
            val_df = val_pool.reset_index(drop=True)
            val_root = anura_root
        if test_name == "ANURASET":
            test_df = test_pool.reset_index(drop=True)
            test_root = anura_root

        val_neg = int((val_df["is_negative"] == True).sum()) if "is_negative" in val_df.columns else 0
        test_neg = int((test_df["is_negative"] == True).sum()) if "is_negative" in test_df.columns else 0
        print(f"[anuraset] negatives in val/test: val={val_neg} test={test_neg}")

        overlap = set(val_df.get("file_name", [])).intersection(set(test_df.get("file_name", [])))
        print(f"[anuraset] val/test overlap={len(overlap)}")
        if overlap:
            raise RuntimeError("FAIL: validation and test overlap detected")

    if val_name.startswith("FNJV"):
        val_root_resolved, _ = _resolve_fnjv_root(config.DATASET_VAL)
        fnjv_val_df = _prepare_fnjv_dataframe(val_root_resolved, train_species, include_negatives=False)
        if config.APPLY_VALIDATION_SPLIT:
            _, val_tmp = train_test_split(fnjv_val_df, test_size=config.VALIDATION_SPLIT, random_state=config.SEED)
            val_df = val_tmp.reset_index(drop=True)
        else:
            val_df = fnjv_val_df.copy()
        _log_split_counts("val", val_df)
        val_root = val_root_resolved

    if test_name.startswith("FNJV"):
        test_root_resolved, _ = _resolve_fnjv_root(config.DATASET_TEST)
        fnjv_test_df = _prepare_fnjv_dataframe(test_root_resolved, train_species, include_negatives=False)
        if config.APPLY_TEST_SPLIT:
            _, test_tmp = train_test_split(fnjv_test_df, test_size=config.TEST_SPLIT, random_state=config.SEED)
            test_df = test_tmp.reset_index(drop=True)
        else:
            test_df = fnjv_test_df.copy()
        _log_split_counts("test", test_df)
        test_root = test_root_resolved

    return DatasetBundle(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        species_codes=train_species,
        train_root=train_root,
        val_root=val_root,
        test_root=test_root,
    )


def summarize_bundle(bundle: DatasetBundle) -> Dict[str, int]:
    return {
        "train_files": len(bundle.train_df),
        "val_files": len(bundle.val_df),
        "test_files": len(bundle.test_df),
        "num_species": len(bundle.species_codes),
    }
