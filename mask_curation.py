import os
import argparse
import numpy as np
from tifffile import imread, imwrite
from cellpose.models import CellposeModel
import torch
from pathlib import Path

# Local import
from imutils.curation_controller import CurationController

def center_crop(image: np.ndarray, crop_size: int = 200) -> np.ndarray:
    """
    Crops a square region from the center of an image.
    Handles (H, W, C), (C, H, W), and (H, W) formats.
    """
    if image.ndim == 3:
        # Decide channel layout by comparing dimensions
        if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            h, w = image.shape[1], image.shape[2]
            is_channels_first = True
        else:
            h, w = image.shape[0], image.shape[1]
            is_channels_first = False
    else:  # 2D image
        h, w = image.shape
        is_channels_first = None

    start_y = max(0, (h - crop_size) // 2)
    start_x = max(0, (w - crop_size) // 2)
    end_y = start_y + crop_size
    end_x = start_x + crop_size

    if is_channels_first is True:
        return image[:, start_y:end_y, start_x:end_x]
    elif is_channels_first is False:
        return image[start_y:end_y, start_x:end_x, :]
    else:
        return image[start_y:end_y, start_x:end_x]

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate and curate a dataset for Cellpose using imutils.")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the directory with raw images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the base directory for saving outputs.")
    parser.add_argument("--crop_size", type=int, default=300, help="Size of the square center crop for each image.")
    return parser.parse_args()

def main():
    """
    Main function to run the image processing and curation pipeline.
    
    This script performs the following steps:
    1.  Loads raw images from a specified directory.
    2.  Asks the user if they want to load pre-existing masks.
    3.  Crops the images and saves them.
    4.  Generates initial masks using Cellpose OR loads existing masks.
    5.  Launches an interactive curation tool to refine the masks.
    6.  Saves the final curated masks.
    """
    args = parse_args()

    # --- Setup Paths ---
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    
    # Define and create output subdirectories
    crops_dir = output_dir / "crops"
    cpose_masks_dir = output_dir / "cpose_masks"
    curated_masks_dir = output_dir / "curated_masks"

    for path in [crops_dir, cpose_masks_dir, curated_masks_dir]:
        path.mkdir(parents=True, exist_ok=True)

    # --- Check for Existing Masks ---
    load_existing = False
    if any(cpose_masks_dir.iterdir()) or any(curated_masks_dir.iterdir()):
        while True:
            resp = input("Found existing masks. Load them for curation? (y/n): ").lower()
            if resp in ('y', 'yes'):
                load_existing = True
                break
            elif resp in ('n', 'no'):
                break
            print("Invalid input. Please enter 'y' or 'n'.")
    
    # --- Initialize Model (if needed) ---
    model = None
    if not load_existing:
        print("\n--- Initializing Cellpose model ---")
        model = CellposeModel(gpu=torch.cuda.is_available())

    # --- Phase 1: Batch Processing ---
    print("\n--- Batch processing started ---")
    data_for_curation = []
    
    # Filter for common image file types
    valid_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    files = sorted([f for f in raw_data_dir.iterdir() if f.suffix.lower() in valid_extensions])

    for idx, raw_path in enumerate(files, start=1):
        key = raw_path.stem
        print(f"Processing {idx}/{len(files)}: {raw_path.name}")

        # Define paths for derived files
        crop_path = crops_dir / f"{key}_crop.tif"
        cpose_path = cpose_masks_dir / f"{key}_cpose_mask.tif"
        curated_path = curated_masks_dir / f"{key}_curated_mask.tif"

        # Load and crop the raw image
        raw_image = imread(raw_path)
        crop = center_crop(raw_image, args.crop_size)
        
        # Save the cropped image for reference
        imwrite(crop_path, crop)
        print(f"  → Saved cropped image to '{crop_path.name}'")
        
        initial_mask = None
        mask_source = "newly generated"

        if load_existing:
            if curated_path.exists():
                initial_mask = imread(curated_path)
                mask_source = "existing curated mask"
            elif cpose_path.exists():
                initial_mask = imread(cpose_path)
                mask_source = "existing cellpose mask"

        # If we aren't loading existing masks or none were found for this image, run Cellpose
        if initial_mask is None:
            if model is None: # Lazy initialization if starting with 'load_existing' but some images need processing
                print("\n--- Initializing Cellpose model for remaining images ---")
                model = CellposeModel(gpu=torch.cuda.is_available())
            
            print("  → Running Cellpose to generate initial mask...")
            masks, _, _ = model.eval(crop, diameter=None)
            imwrite(cpose_path, masks)
            print(f"  → Saved initial Cellpose mask to '{cpose_path.name}'")
            initial_mask = masks
        else:
             print(f"  → Loaded {mask_source}.")


        # Queue image and its mask for the curation step
        data_for_curation.append({
            "image": crop,
            "masks": initial_mask,
            "title": key,
            "path": curated_path
        })

    # --- Phase 2: Interactive Curation ---
    if not data_for_curation:
        print("\nNo images to process or curate. All done! ✅")
        return

    print(f"\n--- Launching curation for {len(data_for_curation)} images ---")
    controller = CurationController(
        images=[d["image"] for d in data_for_curation],
        initial_masks=[d["masks"] for d in data_for_curation],
        titles=[d["title"] for d in data_for_curation],
        mask_save_paths=[d["path"] for d in data_for_curation]
    )
    controller.start()
    print("\n🎉 Curation session finished!")

if __name__ == "__main__":
    main()

