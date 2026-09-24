from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# Paths
# =========================
DATA_DIR = Path("data/PlantVillage")
TRAIN_FILE = DATA_DIR / "color_train.txt"
TEST_FILE = DATA_DIR / "color_test.txt"

# =========================
# Read official split files
# =========================
train_paths = TRAIN_FILE.read_text(encoding="utf-8").splitlines()
test_paths = TEST_FILE.read_text(encoding="utf-8").splitlines()

print("Official training images:", len(train_paths))
print("Official test images:", len(test_paths))

# =========================
# Create training dataframe
# =========================
train_df = pd.DataFrame({"path": train_paths})

# Class name is the parent directory
train_df["class"] = train_df["path"].apply(
    lambda x: Path(x).parent.name
)

# =========================
# Create validation split
# =========================
train_df, val_df = train_test_split(
    train_df,
    test_size=0.10,
    random_state=42,
    stratify=train_df["class"]
)

# =========================
# Create test dataframe
# =========================
test_df = pd.DataFrame({"path": test_paths})

test_df["class"] = test_df["path"].apply(
    lambda x: Path(x).parent.name
)

# =========================
# Save CSV files
# =========================
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# =========================
# Display information
# =========================
print("\nDataset prepared successfully!")

print("\nFinal split:")
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))

print("\nNumber of classes:", train_df["class"].nunique())

print("\nExample:")
print(train_df.head())

print("\nCSV files saved to:")
print(OUTPUT_DIR)