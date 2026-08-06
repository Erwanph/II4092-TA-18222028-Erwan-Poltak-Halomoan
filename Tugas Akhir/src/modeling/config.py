from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "processed" / "dataset_final.csv"
DATASET_RAW_PATH = DATA_DIR / "processed" / "dataset_raw_integrated.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset Configuration
TARGET_COLUMN = "BI_Rate_Pct"
FEATURE_COLUMNS = [
    "Inflation_YoY_Pct",
    "Inflation_Gap_Pct",
    "GDP_Growth_YoY_Pct",
    "USD_IDR_Monthly_Avg",
    "M2_Triliun_Rp",
    "Credit_Growth_YoY_Pct",
    "IHSG_End_of_Month",
    "Foreign_Reserves_Miliar_USD",
    "Federal_Funds_Rate_Pct",
    "Oil_Price_Brent_USD_per_Bbl",
    "Gold_Price_USD_per_Oz",
    "VIX_Volatility_Index",
]

# Split Configuration
TEST_SIZE = 0.2
VAL_SIZE = 0.2
RANDOM_SEED = 42

# panjang jendela awal LSTM/BiLSTM (ikut ditala pada fase tuning)
SEQUENCE_LENGTH = 8

# LLM Configuration
LLM_MODEL = "gpt-4o"
