import os
import numpy as np
from sklearn.model_selection import train_test_split
from modules.utils import (
    dataset_dl,
    load_data_writer_independent,
    run_kfold_training,
    evaluate_test_set,
    plot_training_history,
)

# === CONFIGURATION ===
CONFIG = {
    "USE_CBAM": True,
    "DEFAULT_DATASET_PATH": "signatures",
    "MODEL_SAVE_PATH": "densenet_cbam_best.keras",
    # Image & Hyperparameters
    "IMG_HEIGHT": 224,
    "IMG_WIDTH": 224,
    "IMG_CHANNELS": 3,
    "FOLDS": 5,
    "BATCH_SIZE": 4,
    "K_INSTANCES": 2,
    "EMBEDDING_DIM": 128,
    "LEARNING_RATE": 0.001,
    "MARGIN": 1.0,
    "EPOCHS": 100,
}


def main():
    print("=== Signature Verification Pipeline (K-Fold) ===\n")

    # 1. Dataset Resolution
    if os.path.exists(CONFIG["DEFAULT_DATASET_PATH"]):
        dataset_path = CONFIG["DEFAULT_DATASET_PATH"]
    else:
        try:
            dataset_path = dataset_dl()
        except Exception as e:
            print(f"[ERROR] {e}")
            return

    # 2. Load & Split Data
    print("\n[INFO] Loading Data...")
    try:
        X, y, writer_ids = load_data_writer_independent(
            dataset_path, img_size=(CONFIG["IMG_HEIGHT"], CONFIG["IMG_WIDTH"])
        )
        NUM_WRITERS = len(np.unique(writer_ids))

        # 80/20 Writer-Independent Split
        unique_writers = np.unique(writer_ids)
        writers_tv, writers_test = train_test_split(
            unique_writers, test_size=0.2, random_state=42
        )

        # Create Masks
        tv_mask = np.isin(writer_ids, writers_tv)
        test_mask = np.isin(writer_ids, writers_test)

        X_tv, y_tv, ids_tv = X[tv_mask], y[tv_mask], writer_ids[tv_mask]
        X_test, y_test, ids_test = X[test_mask], y[test_mask], writer_ids[test_mask]

        print(f"Train/Val: {len(X_tv)} samples ({len(writers_tv)} writers)")
        print(f"Test:      {len(X_test)} samples ({len(writers_test)} writers)")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # 3. Run K-Fold Training
    best_model, histories = run_kfold_training(X_tv, y_tv, ids_tv, CONFIG)

    # 4. Evaluate on Test Set
    if best_model:
        evaluate_test_set(best_model, X_test, y_test, ids_test, NUM_WRITERS)
        plot_training_history(histories)


if __name__ == "__main__":
    main()
