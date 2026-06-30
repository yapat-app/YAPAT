# Unified configuration for WSSED pipelines (classic + BirdNET).

# Random seed
SEED = 42

# Dataset roots
ANURASET_ROOT = "/srv/demos/shared/datasets/AnuraSet/raw_data"
# FNJV_ROOT = "/srv/demos/shared/datasets/FNJV/458"
FNJV_ROOT = "/srv/demos/shared/datasets/FNJV/578"

# Dataset selection
DATASET_TRAIN = "FNJV"
DATASET_VAL = "AnuraSet"
DATASET_TEST = "AnuraSet"

# POOLING options: max, mean, linear, exp, att, auto, power, hi, hi_plus, hi_fixed
POOLING = "mean"
# BAG_SECONDS can be an integer (e.g., 3, 10, 30) to cut fixed-length bags,
# or "full" to use the entire recording as a single bag.
BAG_SECONDS = "full"

# FULL_BAG_METHOD is used only when BAG_SECONDS == "full".
# Options: "batch" (batch_size=1, no padding) or "pad" (pad to max length).
FULL_BAG_METHOD = "batch"

# PAD_MODE is used only when BAG_SECONDS == "full" and FULL_BAG_METHOD == "pad".
# Options: "repeat" (repeat audio to max length) or "silence" (zero-pad + mask).
PAD_MODE = "repeat"

# MODEL_NAME options:
#   - Classic pipeline: Baseline, CNN-biGRU, CNN-Transformer, CDur, TALNet
#   - BirdNET pipeline: BirdNET
MODEL_NAME = "BirdNET"

# PIPELINE selects which training/evaluation entrypoint will run from main.py.
# Options:
#   - "auto": MODEL_NAME == "BirdNET" => BirdNET pipeline, otherwise classic pipeline
#   - "classic": always run the original MainClasses-based pipeline
#   - "birdnet": always run focal-data/BirdNET pipeline
PIPELINE = "birdnet"

EPOCHS = 10
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-3

VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
TRAIN_SPLIT = 0.1
APPLY_TRAIN_SPLIT = False
APPLY_VALIDATION_SPLIT = True
APPLY_TEST_SPLIT = True

# When True, add a proportion of non-target recordings to validation/test splits.
INCLUDE_NEGATIVE_SPLITS = True

# Toggle class-specific threshold tuning during evaluation/visualization only.
USE_CLASS_SPECIFIC_THRESHOLD_TUNING = True

# TARGET_SPECIES = ["DENMIN", "LEPLAT", "PHYCUV", "SPHSUR", "SCIPER", "BOABIS", "BOAFAB", "LEPPOD", "PHYALB"]
TARGET_SPECIES = ["DENMIN", "BOARAN", "DENNAN", "LEPFUS"]

sample_rate = 22000
n_mels = 64
n_fft = 1100
hop_length = 550

# Upper-case aliases used by the BirdNET side of the unified main.py.
SAMPLE_RATE = sample_rate
N_MELS = n_mels
N_FFT = n_fft

threshold = 0.5

OVERLAP_BAGS = False
HOP_SECONDS = 1

# BirdNET window settings
WINDOW_SECONDS = 3

# BirdNET/Focal-data dataset root aliases
FNJV_458_ROOT = "/srv/demos/shared/datasets/FNJV/458"
FNJV_578_ROOT = FNJV_ROOT
ANURASET_STRONG_ROOT = "/srv/demos/shared/datasets/AnuraSet/strong_labels"

STRONG_LABELS_458 = "/srv/demos/shared/datasets/AnuraSet/strong_labels/FNJV_458_species_labels.csv"
STRONG_LABELS_578 = "/srv/demos/shared/datasets/AnuraSet/strong_labels/FNJV_578_species_labels.csv"
# BirdNET pipeline needs subset-aware strong labels.
STRONG_LABELS_458_WITH_SUBSET = "/srv/demos/shared/datasets/AnuraSet/strong_labels/FNJV_458_species_labels_with_subset.csv"
STRONG_LABELS_578_WITH_SUBSET = "/srv/demos/shared/datasets/AnuraSet/strong_labels/FNJV_578_species_labels_with_subset.csv"

# STRONG_LABEL_LEVELS can include any of: "High", "Medium", "Low".
STRONG_LABEL_LEVELS = ["High", "Medium", "Low"]

# AnuraSet eval thresholds for visualization/localization.
ANURASET_EVAL = {
    "tag_threshold": 0.5,
    "loc_threshold_high": 0.3,
    "loc_threshold_low": 0.1,
    "smooth": 10,
}

# LOCALIZATION_MODE can be "frame" (current behavior) or "block" (aggregate frames).
LOCALIZATION_MODE = "frame"

# BLOCK_SECONDS is used only when LOCALIZATION_MODE == "block".
BLOCK_SECONDS = 1.0

# ---------------------- BirdNET pipeline options ----------------------
SEEDS = [SEED]
MIL_POOLING = "lin"  # lin | max | avg | exp
EMBEDDING_SOURCE = "birdnet_analyzer"  # birdnet_analyzer | precomputed_txt
ALLOW_AUTO_EXTRACT = True
USE_FALLBACK_FEATURES = False
BIRDNET_CACHE_DIR = "birdnet_cache"
PRECOMPUTED_TXT_DIR = "birdnet_txt_embeddings"
BIRDNET_MODEL_PATH = None
BIRDNET_SR = 48000
EXPECTED_EMBED_DIM = 1024

LOSS_NAME = "bce"  # bce | weighted_bce | focal | asl
WEIGHTED_BCE_POS_WEIGHT_CLAMP = 20.0
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
ASL_GAMMA_POS = 0.0
ASL_GAMMA_NEG = 4.0
ASL_CLIP = 0.05

ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_EPOCHS = 30
SMOKE_MODE = False
SMOKE_MAX_BAGS = 2
SMOKE_EPOCHS = 1

TARGET_SPECIES_458 = ["DENMIN", "LEPLAT", "PHYCUV", "SPHSUR", "SCIPER", "BOABIS", "BOAFAB", "LEPPOD", "PHYALB"]
TARGET_SPECIES_578 = TARGET_SPECIES
USE_NEGATIVE_SAMPLES = False
NEGATIVE_SAMPLE_RATIO = 0.05

UNCERTAINTY_CENTER = 0.5
UNCERTAINTY_TOP_K = 100
ENABLE_FALLBACK_EXTRACTOR = True
ANNOTATION_STORE = "active_learning_annotations.csv"
GUI = False
MULTI_LABEL = False
RETRAIN_EVERY_N = 5
PRINT_EVERY_EPOCHS = 10
COLORED_GT = False
