"""
Script to create CelebA subsets in the same format as Flickr8k
Based on step-01.ipynb structure
"""

import random
import shutil
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import time

# Configuration
CELEBA_SOURCE_DIR = Path("./data/celeba-sample")
PROJECT_DIR = Path("./data")
SUBSET_SIZE = 500  # Number of images to select (adjust as needed)
SEED = 42  # For reproducibility
BATCH_SIZE = 16  # For embedding computation

# Output paths
CELEBA_SUBSET_DIR = PROJECT_DIR / "celeba_subset"
CELEBA_EMBEDDINGS_FILE = PROJECT_DIR / "celeba_embeddings.npz"
CELEBA_METADATA_FILE = PROJECT_DIR / "celeba_subset_metadata.json"

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def select_random_subset(source_dir, n=500, seed=42):
    """Select a random subset of CelebA images."""
    random.seed(seed)

    all_images = sorted(list(Path(source_dir).glob("*.jpg")))
    print(f"Found {len(all_images)} total CelebA images in {source_dir}")

    if len(all_images) == 0:
        raise ValueError(f"No images found in {source_dir}")

    # Random sample
    selected = random.sample(all_images, min(n, len(all_images)))

    return selected


def copy_subset_images(selected_paths, dst_dir):
    """Copy selected CelebA images to subset folder."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nCopying {len(selected_paths)} images to {dst_dir}...")

    copied = 0
    for src_path in tqdm(selected_paths, desc="Copying images"):
        dst_path = dst_dir / src_path.name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)
        copied += 1

    return copied


def compute_clip_embeddings(image_dir, model, processor, device, batch_size=16):
    """
    Compute CLIP embeddings for all images in a directory.

    Args:
        image_dir: Path to directory containing images
        model: CLIP model
        processor: CLIP processor
        device: Device to run on
        batch_size: Number of images to process at once

    Returns:
        image_names: List of image filenames
        embeddings: numpy array of shape (num_images, 512)
    """
    image_dir = Path(image_dir)
    image_paths = sorted(list(image_dir.glob("*.jpg")))

    print(f"\nComputing CLIP embeddings for {len(image_paths)} images...")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")

    image_names = []
    all_embeddings = []

    start_time = time.time()

    # Process in batches for efficiency
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Computing embeddings"):
        batch_paths = image_paths[i : i + batch_size]
        batch_images = []
        batch_names = []

        # Load images in batch
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_images.append(img)
                batch_names.append(img_path.name)
            except Exception as e:
                print(f"  Warning: Skipping {img_path.name}: {e}")

        if not batch_images:
            continue

        # Process batch through CLIP
        inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings (important for cosine similarity)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Store results
        all_embeddings.append(image_features.cpu().numpy())
        image_names.extend(batch_names)

    # Concatenate all batches
    embeddings = np.vstack(all_embeddings)

    elapsed = time.time() - start_time
    print(f"\nComputed {len(image_names)} embeddings in {elapsed:.1f} seconds")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Speed: {len(image_names) / elapsed:.1f} images/second")

    return image_names, embeddings


def save_embeddings(image_names, embeddings, output_file):
    """Save embeddings and image names to a compressed numpy file."""
    np.savez_compressed(output_file, image_names=np.array(image_names), embeddings=embeddings)
    print(f"\nSaved embeddings to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def save_metadata(selected_images, output_file, subset_size, seed):
    """Save subset metadata to JSON for reproducibility."""
    metadata = {
        "dataset": "CelebA",
        "num_images": len(selected_images),
        "subset_size": subset_size,
        "seed": seed,
        "images": [img.name for img in selected_images],
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to: {output_file}")
    print(f"  Contains {len(selected_images)} image names")


def main():
    print("=" * 60)
    print("Creating CelebA Subset")
    print("=" * 60)

    # Check source directory exists
    if not CELEBA_SOURCE_DIR.exists():
        raise FileNotFoundError(f"CelebA source directory not found: {CELEBA_SOURCE_DIR}")

    # Step 1: Select random subset
    print(f"\n1. Selecting {SUBSET_SIZE} random images...")
    selected_images = select_random_subset(CELEBA_SOURCE_DIR, n=SUBSET_SIZE, seed=SEED)
    print(f"  Selected {len(selected_images)} images")

    # Step 2: Copy to subset folder
    print(f"\n2. Copying images to subset folder...")
    copied = copy_subset_images(selected_images, CELEBA_SUBSET_DIR)
    print(f"  Copied {copied} images to {CELEBA_SUBSET_DIR}")

    # Step 3: Load CLIP model
    print(f"\n3. Loading CLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"  Loaded {model_name}")

    # Step 4: Compute embeddings
    print(f"\n4. Computing CLIP embeddings...")
    image_names, embeddings = compute_clip_embeddings(CELEBA_SUBSET_DIR, model, processor, device, batch_size=BATCH_SIZE)

    # Step 5: Save embeddings
    print(f"\n5. Saving embeddings...")
    save_embeddings(image_names, embeddings, CELEBA_EMBEDDINGS_FILE)

    # Step 6: Save metadata
    print("\n6. Saving metadata...")
    save_metadata(selected_images, CELEBA_METADATA_FILE, SUBSET_SIZE, SEED)

    # Summary
    print("\n" + "=" * 60)
    print("CelebA Subset Creation Complete!")
    print("=" * 60)
    print("\nCreated files:")
    print(f"  1. {CELEBA_SUBSET_DIR}/ ({len(image_names)} images)")
    print(f"  2. {CELEBA_EMBEDDINGS_FILE} (CLIP embeddings)")
    print(f"  3. {CELEBA_METADATA_FILE} (metadata)")
    print("\nYou can now use these files with the same format as Flickr8k!")


if __name__ == "__main__":
    main()
