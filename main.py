import os
from modules.utils import dataset_dl, prepare_data_pipeline, build_embedding_model

DEFAULT_DATASET_PATH = "signatures"

# Image Paramters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Training Hyperparameters
BATCH_SIZE = 24
K_INSTANCES = 2
EMBEDDING_DIM = 128
LEARNING_RATE = 0.0005
MARGIN = 1.0
EPOCHS = 75
PATIENCE = 10

MODEL_SAVE_PATH = "densenet_cbam_triplet_model.h5"


def main():
    print("=== Starting Signature Verification Training Pipeline ===\n")

    # 1. Dataset Resolution
    # Check if dataset exists, if not, try to download it
    if os.path.exists(DEFAULT_DATASET_PATH):
        dataset_path = DEFAULT_DATASET_PATH
        print(f"[1/4] Found existing dataset at: {dataset_path}")
    else:
        print(
            f"[1/4] Dataset not found at '{DEFAULT_DATASET_PATH}'. Attempting download..."
        )
        try:
            dataset_path = dataset_dl()
            print(f"      Dataset downloaded to: {dataset_path}")
        except Exception as e:
            print(f"      CRITICAL ERROR: Failed to download dataset. {e}")
            return

    # 2. Prepare Data Generators
    # This handles loading, preprocessing, splitting, and creating the balanced batch generators
    print("\n[2/4] Preparing Data Generators...")
    try:
        train_gen, val_gen = prepare_data_pipeline(
            dataset_path=dataset_path,
            img_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            k_instances=K_INSTANCES,
        )
    except Exception as e:
        print(f"      CRITICAL ERROR: {e}")
        return

    # 3. Build Model Architecture
    print("\n[3/4] Building Model Architecture...")
    model = build_embedding_model(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), embedding_dim=EMBEDDING_DIM
    )
    model.summary()  # Uncomment if you want to see the model summary

    # 4. Compile, Train, and Save
    # This handles the training loop, loss compilation, callbacks, and saving
    """
    print(f"\n[4/4] Starting Training (Epochs: {EPOCHS}, Patience: {PATIENCE})...")
    history = compile_and_train(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        learning_rate=LEARNING_RATE,
        margin=MARGIN,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_path=MODEL_SAVE_PATH
    )
    """
    print("\n=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    main()
