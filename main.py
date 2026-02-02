import os
from kaggle.api.kaggle_api_extended import KaggleApi


def download_bhsig_dataset(dataset_name: str = "nth2165/bhsig260-hindi-bengali", download_path: str = "data"):
    """
    Download the BHSIG260 dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset handle (default: nth2165/bhsig260-hindi-bengali)
        download_path: Directory path where the dataset will be downloaded (default: data)
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
    download_bhsig_dataset()


if __name__ == "__main__":
    main()
