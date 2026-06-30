"""Central configuration for the WSSED BirdNET pipeline."""

# Reproducibility
SEED = 42
SEEDS = [0, 1, 2, 3, 4, 5, 42]

# Dataset roots
# ANURASET_ROOT = "/ds-iml/Bioacoustics/AnuraSet/raw_data"
# ANURASET_STRONG_ROOT = "/ds-iml/Bioacoustics/AnuraSet/strong_labels"
# FNJV_458_ROOT = "/ds-iml/Bioacoustics/FNJV/458"
# FNJV_578_ROOT = "/ds-iml/Bioacoustics/FNJV/578"
ANURASET_ROOT = r"C:\Users\noma01\PycharmProjects\WSSED\PAM datasets\AnuraSet\raw_data"
ANURASET_STRONG_ROOT = r"C:\Users\noma01\PycharmProjects\WSSED\PAM datasets\AnuraSet\strong_labels"
FNJV_458_ROOT = r"C:\Users\noma01\PycharmProjects\WSSED\PAM datasets\FNJV\458"
FNJV_578_ROOT = r"C:\Users\noma01\PycharmProjects\WSSED\PAM datasets\FNJV\578"

# Dataset selection
DATASET_TRAIN = "FNJV_578"
DATASET_VAL = "ANURASET"
DATASET_TEST = "ANURASET"

# Model setup
MODEL_NAME = "BirdNet"
NUM_EPOCHS = 50
EPOCHS = NUM_EPOCHS
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
threshold = 0.5
HOP_SECONDS = 1
WINDOW_SECONDS = 3
MIL_POOLING = "lin"
GUI = False
MULTI_LABEL = False
RETRAIN_EVERY_N = 5
PRINT_EVERY_EPOCHS = 10
COLORED_GT = False

# Embedding source settings
EMBEDDING_SOURCE = "birdnet_analyzer"  # birdnet_analyzer | precomputed_txt
ALLOW_AUTO_EXTRACT = True
USE_FALLBACK_FEATURES = False
BIRDNET_CACHE_DIR = "birdnet_cache"
PRECOMPUTED_TXT_DIR = "birdnet_txt_embeddings"
# Optional explicit model path for birdnet-analyzer package.
# Leave as None to use the package default checkpoint.
BIRDNET_MODEL_PATH = None
BIRDNET_SR = 48000
EXPECTED_EMBED_DIM = 1024

# Loss selection
LOSS_NAME = "bce"  # bce | weighted_bce | focal | asl
WEIGHTED_BCE_POS_WEIGHT_CLAMP = 20.0
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25
ASL_GAMMA_POS = 0.0
ASL_GAMMA_NEG = 4.0
ASL_CLIP = 0.05

# Training control
ENABLE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_EPOCHS = 30

# Smoke-test settings
SMOKE_MODE = False
SMOKE_MAX_BAGS = 2
SMOKE_EPOCHS = 1

# Audio and spectral settings
SAMPLE_RATE = 22000
N_MELS = 64
N_FFT = 1100

# Target species
TARGET_SPECIES_458 = ["DENMIN", "LEPLAT", "PHYCUV", "SPHSUR", "SCIPER", "BOABIS", "BOAFAB", "LEPPOD", "PHYALB"]
TARGET_SPECIES_578 = ["DENMIN", "BOARAN", "DENNAN", "LEPFUS"]

# Dataset split settings
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
TRAIN_SPLIT = 0.1
APPLY_TRAIN_SPLIT = False
APPLY_VALIDATION_SPLIT = False
APPLY_TEST_SPLIT = False

# Negative sample controls
USE_NEGATIVE_SAMPLES = False
NEGATIVE_SAMPLE_RATIO = 0.05

# Strong-label CSV paths for AnuraSet overlap with FNJV species
# STRONG_LABELS_578 = "/ds-iml/Bioacoustics/AnuraSet/strong_labels/FNJV_578_species_labels_with_subset.csv"
# STRONG_LABELS_458 = "/ds-iml/Bioacoustics/AnuraSet/strong_labels/FNJV_458_species_labels_with_subset.csv"
STRONG_LABELS_578 = r"C:\Users\noma01\PycharmProjects\WSSED\PAM datasets\AnuraSet\strong_labels\FNJV_578_species_labels_with_subset.csv"
STRONG_LABELS_458 = r"C:\Users\noma01\PycharmProjects\WSSED\PAM datasets\AnuraSet\strong_labels\FNJV_458_species_labels_with_subset.csv"

# Active learning settings
UNCERTAINTY_CENTER = 0.5
UNCERTAINTY_TOP_K = 100

# Runtime behavior
ENABLE_FALLBACK_EXTRACTOR = True
ANNOTATION_STORE = "active_learning_annotations.csv"
