import os
import cv2
import math
import numpy as np
import kagglehub
from glob import glob
from tensorflow import keras  # type: ignore
from keras import optimizers, models, layers
from keras.callbacks import EarlyStopping
from modules.densenet import DenseNet121_CBAM
from sklearn.model_selection import train_test_split
from modules.triplet_loss import TripletSemiHardLoss


def dataset_dl():
    dataset_path = kagglehub.dataset_download("cristianjaycosep/cedar-dataset")
    dataset_path = os.path.join(dataset_path, "cedar_signatures")
    return dataset_path


def preprocess_image(img_path, img_size=(224, 224)):
    """
    Preprocessing Pipeline adapted from thesis notebook:
    1. Load as Grayscale
    2. Apply Otsu's Threshold (Background Removal)
    3. Invert (White signal on black background)
    4. Resize to target size
    5. Convert to 3-Channel RGB (for DenseNet compatibility)
    6. Normalize to [0, 1]
    """
    # 1. Load as Grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load image {img_path}")
        return None

    # 2. Apply Otsu's Threshold
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Invert (Signatures are usually dark on light, we want features to be high values)
    img = cv2.bitwise_not(thresh)

    # 4. Resize
    img = cv2.resize(img, img_size)

    # 5. Convert to 3-channel (Required for standard DenseNet input)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 6. Normalize
    img = img.astype("float32") / 255.0

    return img


def load_data_writer_independent(root_path, img_size=(224, 224)):
    """
    Loads dataset assuming folder structure:
    root_path/
       writer_01/
          original_01.png
          forgeries_01.png
       writer_02/ ...

    Returns:
        X: Array of images
        y: Array of labels (writer IDs)
        writer_ids: Array of underlying writer identities (for splitting)
    """
    images, labels, writer_ids = [], [], []

    writers = sorted(
        [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    )

    # Optional: Filter for CEDAR/ICDAR specific naming if needed
    # writers = [w for w in writers if "writer" in w]

    print(f"Loading data from {len(writers)} writers...")

    for i, writer in enumerate(writers):
        writer_path = os.path.join(root_path, writer)

        # Load Originals (Assign distinct label)
        # Assuming format like 'original_*.png' or similar
        orig_pattern = os.path.join(writer_path, "original*.*")
        for img_path in glob(orig_pattern):
            img = preprocess_image(img_path, img_size)
            if img is not None:
                images.append(img)
                labels.append(i)  # Genuine Label for this writer
                writer_ids.append(i)

        # Load Forgeries (Assign distinct label offset by num_writers)
        # Note: For verification, we often train only on genuines or treat forgeries
        # as a separate class. The notebook strategy separates them.
        forg_pattern = os.path.join(writer_path, "forger*.*")
        for img_path in glob(forg_pattern):
            img = preprocess_image(img_path, img_size)
            if img is not None:
                images.append(img)
                # Labeling strategy from notebook:
                # Original = ID, Forgery = ID + Num_Writers
                labels.append(i + len(writers))
                writer_ids.append(i)

    print(f"Successfully loaded {len(images)} images.")
    return np.array(images), np.array(labels), np.array(writer_ids)


class BalancedBatchGenerator(keras.utils.Sequence):
    """
    Generates balanced batches for Triplet Loss (PK Sampling).
    Ensures each batch contains P classes and K instances per class.

    This is CRITICAL for Triplet Semi-Hard Loss to work.
    """

    def __init__(self, X, y, batch_size=24, k_instances=2):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.k_instances = (
            k_instances  # How many images per class in one batch (usually 2 or 4)
        )
        self.classes = np.unique(y)
        self.indices = {c: np.where(y == c)[0] for c in self.classes}

        # Filter classes that don't have enough samples
        self.classes = [
            c for c in self.classes if len(self.indices[c]) >= self.k_instances
        ]

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        batch_x, batch_y = [], []

        # We need (Batch_Size / K) classes
        num_classes_per_batch = self.batch_size // self.k_instances

        # Randomly select P classes
        selected_classes = np.random.choice(
            self.classes, num_classes_per_batch, replace=True
        )

        for cls in selected_classes:
            idxs = self.indices[cls]

            # Randomly select K instances from this class
            # If replacement is needed (small classes), allow it
            replace = len(idxs) < self.k_instances
            chosen_indices = np.random.choice(idxs, self.k_instances, replace=replace)

            batch_x.extend(self.X[chosen_indices])
            batch_y.extend(self.y[chosen_indices])

        return np.array(batch_x), np.array(batch_y)


def prepare_data_pipeline(dataset_path, img_size, batch_size, k_instances):
    """
    Orchestrates data loading, splitting, and generator creation.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path '{dataset_path}' does not exist.")

    X, y, writer_ids = load_data_writer_independent(dataset_path, img_size)

    if len(X) == 0:
        raise ValueError("No images loaded.")

    unique_writers = np.unique(writer_ids)
    train_writers, val_writers = train_test_split(
        unique_writers, test_size=0.2, random_state=42
    )

    train_mask = np.isin(writer_ids, train_writers)
    val_mask = np.isin(writer_ids, val_writers)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print("-" * 30)
    print(f"Train Writers: {len(train_writers)} | Samples: {len(X_train)}")
    print(f"Val Writers:   {len(val_writers)} | Samples: {len(X_val)}")
    print("-" * 30)

    train_gen = BalancedBatchGenerator(
        X_train, y_train, batch_size=batch_size, k_instances=k_instances
    )
    val_gen = BalancedBatchGenerator(
        X_val, y_val, batch_size=batch_size, k_instances=k_instances
    )

    return train_gen, val_gen


def build_embedding_model(input_shape=(224, 224, 3), embedding_dim=128):
    """
    Builds the model for Online Triplet Mining.
    This model takes 1 input image and outputs 1 L2-normalized embedding vector.
    """
    # 1. Base Model (DenseNet121 + CBAM inside blocks)
    # We use include_top=False to get the feature map before classification
    base_model = DenseNet121_CBAM(input_shape=input_shape, include_top=False)

    # Optional: Freeze base model initially?
    # base_model.trainable = False

    # 2. Embedding Head
    # Check if output is 4D (H,W,C) or 2D (B,F).
    if len(base_model.output.shape) == 4:
        x = layers.GlobalAveragePooling2D()(base_model.output)
    else:
        x = base_model.output

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(embedding_dim, activation=None, name="dense_embedding")(x)

    # 3. L2 Normalization (CRITICAL for Triplet Distance Metrics)
    embeddings = layers.UnitNormalization(axis=1, name="l2_norm")(x)

    model = models.Model(
        inputs=base_model.input, outputs=embeddings, name="DenseNet_CBAM_Embedding"
    )
    return model


def compile_and_train(
    model, train_gen, val_gen, learning_rate, margin, epochs, patience, save_path
):
    """
    Compiles and trains the model.
    """
    print("Compiling with TripletSemiHardLoss...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=TripletSemiHardLoss(margin=margin),
    )

    print("Starting Training...")
    early_stop = EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1,
    )

    print(f"Saving Model to {save_path}...")
    model.save(save_path)
    return history
