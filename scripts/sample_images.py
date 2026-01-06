"""
Image sampling utilities for CelebA and Flickr8k datasets.
"""
import random
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
from PIL import Image


# Get the script's directory and construct the data path relative to it
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"

CELEBA_DIR = DATA_DIR / "celeba-sample"
FLICKR_DIR = DATA_DIR / "flickr8k"
FLICKR_IMAGES_DIR = FLICKR_DIR / "Images"
FLICKR_CAPTIONS_FILE = FLICKR_DIR / "captions.txt"


def sample_celeba_images(
    n: int = 100, seed: Optional[int] = 42, return_images: bool = False
) -> List[Path] | List[Tuple[Path, Image.Image]]:
    """
    Sample n random images from the CelebA dataset.

    Args:
        n: Number of images to sample (default: 100)
        seed: Random seed for reproducibility (default: 42)
        return_images: If True, return (path, PIL.Image) tuples; if False, return paths only

    Returns:
        List of image paths or list of (path, image) tuples
    """
    if not CELEBA_DIR.exists():
        raise FileNotFoundError(
            f"CelebA directory not found at {CELEBA_DIR}. "
            "Run scripts/load_datasets.py --dataset celeba first."
        )

    # Get all jpg files
    all_images = sorted(CELEBA_DIR.glob("*.jpg"))

    if len(all_images) == 0:
        raise ValueError(f"No images found in {CELEBA_DIR}")

    # Sample n images
    if seed is not None:
        random.seed(seed)

    n_samples = min(n, len(all_images))
    sampled_paths = random.sample(all_images, n_samples)

    if return_images:
        return [(path, Image.open(path)) for path in sampled_paths]
    else:
        return sampled_paths


def sample_flickr_images(
    n: int = 100, seed: Optional[int] = 42, return_images: bool = False
) -> List[Path] | List[Tuple[Path, Image.Image]]:
    """
    Sample n random images from the Flickr8k dataset.

    Args:
        n: Number of images to sample (default: 100)
        seed: Random seed for reproducibility (default: 42)
        return_images: If True, return (path, PIL.Image) tuples; if False, return paths only

    Returns:
        List of image paths or list of (path, image) tuples
    """
    if not FLICKR_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Flickr8k images directory not found at {FLICKR_IMAGES_DIR}. "
            "Run scripts/load_datasets.py --dataset flickr first."
        )

    # Get all jpg files
    all_images = sorted(FLICKR_IMAGES_DIR.glob("*.jpg"))

    if len(all_images) == 0:
        raise ValueError(f"No images found in {FLICKR_IMAGES_DIR}")

    # Sample n images
    if seed is not None:
        random.seed(seed)

    n_samples = min(n, len(all_images))
    sampled_paths = random.sample(all_images, n_samples)

    if return_images:
        return [(path, Image.open(path)) for path in sampled_paths]
    else:
        return sampled_paths


def get_flickr_captions(image_path: Path | str) -> List[str]:
    """
    Get all captions for a specific Flickr8k image.

    Args:
        image_path: Path to the image or just the image filename

    Returns:
        List of captions for the image (typically 5 captions per image)
    """
    if not FLICKR_CAPTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Flickr8k captions file not found at {FLICKR_CAPTIONS_FILE}. "
            "Run scripts/load_datasets.py --dataset flickr first."
        )

    # Get just the filename if a full path was provided
    if isinstance(image_path, Path):
        image_name = image_path.name
    else:
        image_name = Path(image_path).name

    # Load captions
    df = pd.read_csv(FLICKR_CAPTIONS_FILE)

    # Filter for this specific image
    captions = df[df["image"] == image_name]["caption"].tolist()

    return captions


def get_all_flickr_captions() -> pd.DataFrame:
    """
    Load all Flickr8k captions as a DataFrame.

    Returns:
        DataFrame with columns: ['image', 'caption']
    """
    if not FLICKR_CAPTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Flickr8k captions file not found at {FLICKR_CAPTIONS_FILE}. "
            "Run scripts/load_datasets.py --dataset flickr first."
        )

    return pd.read_csv(FLICKR_CAPTIONS_FILE)


def load_images(paths: List[Path]) -> List[Image.Image]:
    """
    Load PIL Images from a list of paths.

    Args:
        paths: List of image file paths

    Returns:
        List of PIL Image objects
    """
    return [Image.open(path) for path in paths]


def sample_dataset(
    dataset: str = "flickr",
    n: int = 100,
    seed: Optional[int] = 42,
    return_images: bool = False,
) -> List[Path] | List[Tuple[Path, Image.Image]]:
    """
    Generic function to sample images from either dataset.

    Args:
        dataset: Either "celeba" or "flickr"
        n: Number of images to sample (default: 100)
        seed: Random seed for reproducibility (default: 42)
        return_images: If True, return (path, PIL.Image) tuples; if False, return paths only

    Returns:
        List of image paths or list of (path, image) tuples
    """
    if dataset.lower() in ["celeba", "celeb"]:
        return sample_celeba_images(n=n, seed=seed, return_images=return_images)
    elif dataset.lower() in ["flickr", "flickr8k"]:
        return sample_flickr_images(n=n, seed=seed, return_images=return_images)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}. Choose 'celeba' or 'flickr'."
        )


# Example usage
if __name__ == "__main__":
    print("Sampling 10 CelebA images...")
    celeba_paths = sample_celeba_images(n=10)
    print(f"Sampled {len(celeba_paths)} CelebA images")
    for path in celeba_paths[:3]:
        print(f"  - {path.name}")

    print("\nSampling 10 Flickr8k images...")
    flickr_paths = sample_flickr_images(n=10)
    print(f"Sampled {len(flickr_paths)} Flickr8k images")
    for path in flickr_paths[:3]:
        print(f"  - {path.name}")
        captions = get_flickr_captions(path)
        print(f"    Captions: {len(captions)}")
        if captions:
            print(f"    Example: {captions[0]}")
