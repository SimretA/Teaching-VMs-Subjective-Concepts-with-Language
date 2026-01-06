import os
import requests
import zipfile
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Make sure to have your KAGGLE_API_TOKEN variable set in your .env file
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")
if not KAGGLE_API_TOKEN:
    raise ValueError("KAGGLE_API_TOKEN not found in environment variables.")

# Get the script's directory and construct the data path relative to it
SCRIPT_DIR = Path(__file__).parent
data_dir = SCRIPT_DIR.parent / "data"


def download_celeba():
    """Download CelebA sample dataset"""
    print("Downloading CelebA sample dataset...")
    celeba_url = "https://bit.ly/celeba-sample"
    zip_path = data_dir / "celeba-sample.zip"

    resp = requests.get(celeba_url, allow_redirects=True, timeout=120)
    resp.raise_for_status()

    zip_path.write_bytes(resp.content)
    print(f"Downloaded to {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=data_dir)
    print(f"Extracted CelebA to {data_dir}")


def download_flickr_faces():
    """Download Flickr8k dataset from Kaggle"""
    print("Downloading Flickr8k dataset from Kaggle...")
    # Kaggle API endpoint for downloading datasets
    # Dataset: adityajn105/flickr8k
    kaggle_dataset = "adityajn105/flickr8k"
    zip_path = data_dir / "flickr8k.zip"

    # Use Kaggle API to download the dataset
    # Format: https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset-slug}
    url = f"https://www.kaggle.com/api/v1/datasets/download/{kaggle_dataset}"

    headers = {"Authorization": f"Bearer {KAGGLE_API_TOKEN}"}

    resp = requests.get(url, headers=headers, allow_redirects=True, timeout=300)
    resp.raise_for_status()

    zip_path.write_bytes(resp.content)
    print(f"Downloaded to {zip_path}")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(path=data_dir / "flickr8k")
    print(f"Extracted Flickr8k to {data_dir / 'flickr8k'}")


def main():
    """Download datasets based on command-line arguments"""
    parser = argparse.ArgumentParser(description="Download datasets for CLIP experiments")
    parser.add_argument(
        "--dataset",
        choices=["celeba", "flickr", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    args = parser.parse_args()

    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir.absolute()}\n")

    try:
        if args.dataset in ["celeba", "all"]:
            download_celeba()
            if args.dataset == "all":
                print("\n" + "=" * 50 + "\n")

        if args.dataset in ["flickr", "all"]:
            download_flickr_faces()

        print("\nDataset(s) downloaded successfully!")
    except Exception as e:
        print(f"\nError downloading datasets: {e}")
        raise


if __name__ == "__main__":
    main()
