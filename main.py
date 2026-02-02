import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(dataset_name: str, download_path: str):
    """
    Download a dataset from Kaggle.

    Args:
        dataset_name: Kaggle dataset handle (e.g., "owner/dataset")
        download_path: Directory path where the dataset will be downloaded
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_path, exist_ok=True)
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading dataset: {dataset_name} to {download_path}...")
    api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    print(f"Dataset downloaded successfully to {download_path}")


def main():
    print("Hello from deep-learning-based-signature-forgery-detection-for-personal-identity-authentication!")
    
    # Download the BHSIG260 dataset
    print("\n=== Downloading BHSig260 Dataset ===")
    download_kaggle_dataset("nth2165/bhsig260-hindi-bengali", "data")
    
    # Download the CEDAR dataset
    print("\n=== Downloading CEDAR Dataset ===")
    download_kaggle_dataset("shreelakshmigp/cedardataset", "data/cedardataset")
    
    print("\n✅ All datasets downloaded successfully!")


if __name__ == "__main__":
    main()
