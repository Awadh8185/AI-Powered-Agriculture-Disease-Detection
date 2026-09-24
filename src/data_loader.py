from pathlib import Path
import pandas as pd
import tensorflow as tf

# =========================
# Configuration
# =========================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

DATA_DIR = Path("data/PlantVillage")
PROCESSED_DIR = Path("data/processed")

# =========================
# Load CSV files
# =========================

train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

# =========================
# Create class names
# =========================

class_names = sorted(train_df["class"].unique())

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

print("Number of classes:", len(class_names))

# =========================
# Convert paths to full paths
# =========================

def get_full_path(relative_path):
    return str(DATA_DIR.parent / "PlantVillage" / relative_path)


train_paths = train_df["path"].apply(get_full_path).values
val_paths = val_df["path"].apply(get_full_path).values
test_paths = test_df["path"].apply(get_full_path).values

train_labels = train_df["class"].map(class_to_index).values
val_labels = val_df["class"].map(class_to_index).values
test_labels = test_df["class"].map(class_to_index).values

# =========================
# Image loading function
# =========================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32)

    return image, label

# =========================
# Create TensorFlow datasets
# =========================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_paths, test_labels)
)

# =========================
# Apply preprocessing
# =========================

train_dataset = (
    train_dataset
    .shuffle(2000)
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = (
    val_dataset
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_dataset = (
    test_dataset
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# =========================
# Test the pipeline
# =========================

images, labels = next(iter(train_dataset))

print("\nData loader working successfully!")

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("\nClass names:")
for i, name in enumerate(class_names):
    print(i, ":", name)