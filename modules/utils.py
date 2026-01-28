import os
import cv2
import math
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
from glob import glob
from tensorflow import keras  # type: ignore
from keras import optimizers, models, layers
from keras.callbacks import EarlyStopping
from modules import densenet
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from modules.triplet_loss import TripletSemiHardLoss

# --- 1. DATASET UTILS ---


def dataset_dl():
    dataset_path = kagglehub.dataset_download("cristianjaycosep/cedar-dataset")
    dataset_path = os.path.join(dataset_path, "cedar_signatures")
    return dataset_path


def preprocess_image(img_path, img_size=(224, 224)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.bitwise_not(thresh)
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype("float32") / 255.0
    return img


def load_data_writer_independent(root_path, img_size=(224, 224)):
    images, labels, writer_ids = [], [], []
    writers = sorted(
        [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    )
    print(f"Loading data from {len(writers)} writers...")

    for i, writer in enumerate(writers):
        writer_path = os.path.join(root_path, writer)

        # Originals
        for img_path in glob(os.path.join(writer_path, "original*.*")):
            img = preprocess_image(img_path, img_size)
            if img is not None:
                images.append(img)
                labels.append(i)
                writer_ids.append(i)

        # Forgeries
        for img_path in glob(os.path.join(writer_path, "forger*.*")):
            img = preprocess_image(img_path, img_size)
            if img is not None:
                images.append(img)
                labels.append(i + len(writers))
                writer_ids.append(i)

    print(f"Successfully loaded {len(images)} images.")
    return np.array(images), np.array(labels), np.array(writer_ids)


# --- 2. GENERATORS ---


class BalancedBatchGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size=24, k_instances=2):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.k_instances = k_instances
        self.classes = np.unique(y)
        self.indices = {c: np.where(y == c)[0] for c in self.classes}
        self.classes = [
            c for c in self.classes if len(self.indices[c]) >= self.k_instances
        ]

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        num_classes_per_batch = max(1, self.batch_size // self.k_instances)
        selected_classes = np.random.choice(
            self.classes, num_classes_per_batch, replace=True
        )

        for cls in selected_classes:
            idxs = self.indices[cls]
            replace = len(idxs) < self.k_instances
            chosen = np.random.choice(idxs, self.k_instances, replace=replace)
            batch_x.extend(self.X[chosen])
            batch_y.extend(self.y[chosen])

        return np.array(batch_x), np.array(batch_y)


# --- 3. MODEL UTILS ---


def build_embedding_model(input_shape=(224, 224, 3), embedding_dim=128, use_cbam=True):
    attn = "cbam_block" if use_cbam else None
    base_model = densenet.DenseNetImageNet121(
        input_shape=input_shape, include_top=False, weights=None, attention_module=attn
    )
    x = base_model.output
    if len(x.shape) == 4:
        x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(embedding_dim, activation=None, name="dense_embedding")(x)
    output = layers.UnitNormalization(axis=1, name="l2_norm")(x)

    return models.Model(
        inputs=base_model.input,
        outputs=output,
        name=f"DenseNet_{'CBAM' if use_cbam else 'STD'}",
    )


# --- 4. K-FOLD TRAINING LOOP (NEW) ---


def run_kfold_training(X, y, writer_ids, config):
    """
    Executes K-Fold Cross-Validation.
    Returns: best_model, history_list
    """
    print(f"\n[INFO] Starting {config['FOLDS']}-Fold Cross-Validation...")
    print("=" * 70)

    kf = KFold(n_splits=config["FOLDS"], shuffle=True, random_state=42)
    unique_writers = np.unique(writer_ids)

    fold_losses = []
    fold_histories = []
    best_val_loss = np.inf
    final_model = None

    for fold, (tr_idx, val_idx) in enumerate(kf.split(unique_writers), start=1):
        print(f"\nFOLD {fold}/{config['FOLDS']}")
        print("-" * 70)

        tr_writers = unique_writers[tr_idx]
        val_writers = unique_writers[val_idx]

        tr_mask = np.isin(writer_ids, tr_writers)
        val_mask = np.isin(writer_ids, val_writers)

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        print(f"Training samples: {len(X_tr)} | Validation samples: {len(X_val)}")

        train_gen = BalancedBatchGenerator(
            X_tr, y_tr, config["BATCH_SIZE"], config["K_INSTANCES"]
        )
        val_gen = BalancedBatchGenerator(
            X_val, y_val, config["BATCH_SIZE"], config["K_INSTANCES"]
        )

        model = build_embedding_model(
            input_shape=(
                config["IMG_HEIGHT"],
                config["IMG_WIDTH"],
                config["IMG_CHANNELS"],
            ),
            embedding_dim=config["EMBEDDING_DIM"],
            use_cbam=config["USE_CBAM"],
        )

        model.compile(
            optimizer=optimizers.Adam(learning_rate=config["LEARNING_RATE"]),
            loss=TripletSemiHardLoss(margin=config["MARGIN"]),
        )

        early_stop = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )

        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config["EPOCHS"],
            callbacks=[early_stop],
            verbose=1,
        )
        fold_histories.append(history)

        val_loss = min(history.history["val_loss"])
        fold_losses.append(val_loss)
        print(f"Fold {fold} Best Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            final_model = model
            model.save(config["MODEL_SAVE_PATH"])
            print(f"New best model saved to {config['MODEL_SAVE_PATH']}")

    print("\n" + "=" * 70)
    print(
        f"Avg K-Fold Loss: {np.mean(fold_losses):.4f} | Best Loss: {best_val_loss:.4f}"
    )
    print("=" * 70)

    return final_model, fold_histories


# --- 5. EVALUATION UTILS (NEW) ---


def evaluate_test_set(model, X, y, writer_ids, num_writers):
    print("\n[INFO] Evaluating on Test Set...")
    print("=" * 70)

    embeddings = model.predict(X, verbose=0)
    distances, labels = [], []

    present_writers = np.unique(writer_ids)
    for w in present_writers:
        genuine_idx = np.where(y == w)[0]
        forged_idx = np.where(y == (w + num_writers))[0]

        # Genuine Pairs
        for i in range(len(genuine_idx)):
            for j in range(i + 1, len(genuine_idx)):
                dist = np.linalg.norm(
                    embeddings[genuine_idx[i]] - embeddings[genuine_idx[j]]
                )
                distances.append(dist)
                labels.append(1)

        # Forgery Pairs
        for i in genuine_idx:
            for j in forged_idx:
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances.append(dist)
                labels.append(0)

    distances = np.array(distances)
    labels = np.array(labels)

    # Find Threshold
    best_acc, best_thresh = 0, 0
    t_min, t_max = np.min(distances), np.max(distances)
    for t in np.linspace(t_min, t_max, 1000):
        preds = (distances < t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t

    preds = (distances < best_thresh).astype(int)

    print(f"Optimal Threshold: {best_thresh:.4f}")
    print(f"Accuracy:  {accuracy_score(labels, preds)*100:6.2f}%")
    print(f"Precision: {precision_score(labels, preds)*100:6.2f}%")
    print(f"Recall:    {recall_score(labels, preds)*100:6.2f}%")
    print(f"F1-Score:  {f1_score(labels, preds)*100:6.2f}%")
    print(f"ROC-AUC:   {roc_auc_score(labels, -distances)*100:6.2f}%")
    print("=" * 70)


def plot_training_history(histories):
    if not histories:
        return
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for i, h in enumerate(histories):
        plt.plot(h.history["loss"], label=f"Fold {i+1}", alpha=0.7)
        plt.plot(h.history["val_loss"], "--", alpha=0.7)
    plt.title("Triplet Loss per Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    min_losses = [min(h.history["val_loss"]) for h in histories]
    plt.bar(range(1, len(histories) + 1), min_losses, color="steelblue")
    plt.title("Best Validation Loss per Fold")
    plt.xlabel("Fold")

    plt.tight_layout()
    plt.show()
