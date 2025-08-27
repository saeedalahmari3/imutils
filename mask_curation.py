import os
import argparse
import numpy as np
from tifffile import imread, imwrite
from cellpose.models import CellposeModel
import torch
from pathlib import Path
import platform

# Local import
from imutils.curation_controller import CurationController

def setup_device():
    """Configure the device for Mac M-series (Apple Silicon) or other platforms"""
    if platform.system() == 'Darwin' and platform.processor() == 'arm':
        # Check if MPS (Metal Performance Shaders) is available
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal GPU acceleration")
        else:
            device = torch.device("cpu")
            print("MPS not available, falling back to CPU")
    else:
        # For other platforms, use CUDA if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    return device

def center_crop(image: np.ndarray, crop_size: int = 200) -> np.ndarray:
    """
    Crops four square regions from the center of an image.
    Handles (H, W, C), (C, H, W), and (H, W) formats.
    Returns a list of 4 cropped images.
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

    # Calculate the center
    center_y = h // 2
    center_x = w // 2
    half_crop = crop_size // 2

    # Offsets for 4 crops: top-left, top-right, bottom-left, bottom-right
    offsets = [
        (-half_crop, -half_crop),  # top-left
        (-half_crop, half_crop),   # top-right
        (half_crop, -half_crop),   # bottom-left
        (half_crop, half_crop)     # bottom-right
    ]
    crops = []
    for dy, dx in offsets:
        start_y = max(0, center_y + dy)
        start_x = max(0, center_x + dx)
        end_y = min(h, start_y + crop_size)
        end_x = min(w, start_x + crop_size)
        if is_channels_first is True:
            crop = image[:, start_y:end_y, start_x:end_x]
        elif is_channels_first is False:
            crop = image[start_y:end_y, start_x:end_x, :]
        else:
            crop = image[start_y:end_y, start_x:end_x]
        crops.append(crop)
    return crops

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate and curate a dataset for Cellpose using imutils.")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to the directory with raw images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the base directory for saving outputs.")
    parser.add_argument("--crop_size", type=int, default=256, help="Size of the square center crop for each image.")
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
    
    # --- Initialize Model and Device (if needed) ---
    model = None
    device = setup_device()
    
    if not load_existing:
        print("\n--- Initializing Cellpose model ---")
        # For M-series Macs, we'll use MPS (Metal) backend if available
        use_gpu = device.type in ['cuda', 'mps']
        model = CellposeModel(gpu=use_gpu)

    # --- Phase 1: Batch Processing ---
    print("\n--- Batch processing started ---")
    data_for_curation = []
    # Filter for common image file types
    valid_extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    files = sorted([
        f for f in raw_data_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in valid_extensions and f.name.startswith("nucleus.p")
    ])
    print(files)

    for idx, raw_path in enumerate(files, start=1):
        # Get the parent folder name as the key instead of the file name
        folder_key = raw_path.parent.name
        key = folder_key  # Use folder name instead of file name
        print(f"Processing {idx}/{len(files)}: {raw_path.name} (Folder: {folder_key})")

        # Define paths for derived files
        crop_path = crops_dir / f"{folder_key}_crop.tif"
        cpose_path = cpose_masks_dir / f"{folder_key}_cpose_mask.tif"
        curated_path = curated_masks_dir / f"{folder_key}_curated_mask.tif"

        # Load the raw image (z-stack)
        raw_image = imread(raw_path)
        
        # Check if the image is a z-stack (3D)
        if raw_image.ndim == 3:
            num_slices = raw_image.shape[0]
            print(f"  → Processing z-stack with {num_slices} slices")
            
            z = 0
            step = 5
            while z < num_slices:
                # Get the current slice
                slice_image = raw_image[z]
                
                # Create slice-specific paths
                slice_key = f"{folder_key}-z{z:03d}"  # Format: foldername-z001, foldername-z002, etc.
                
                # Crop the slice (returns 4 crops)
                cropped_slices = center_crop(slice_image, args.crop_size)
                for crop_idx, cropped_slice in enumerate(cropped_slices):
                    crop_suffix = f"-c{crop_idx+1}"
                    crop_key = f"{slice_key}{crop_suffix}"
                    slice_path = crops_dir / f"{crop_key}_crop.tif"
                    slice_cpose_path = cpose_masks_dir / f"{crop_key}_mask.tif"
                    slice_curated_path = curated_masks_dir / f"{crop_key}_curated.tif"

                    # Save the cropped slice
                    imwrite(slice_path, cropped_slice)
                    print(f"  → Saved cropped slice {z+1}/{num_slices} crop {crop_idx+1}/4 to '{slice_path.name}'")

                    initial_mask = None
                    mask_source = "newly generated"

                    if load_existing:
                        if slice_curated_path.exists():
                            initial_mask = imread(slice_curated_path)
                            mask_source = "existing curated mask"
                        elif slice_cpose_path.exists():
                            initial_mask = imread(slice_cpose_path)
                            mask_source = "existing cellpose mask"

                    # Generate mask for the slice if needed
                    if initial_mask is None:
                        if model is None:
                            print("\n--- Initializing Cellpose model for remaining images ---")
                            model = CellposeModel(gpu=torch.cuda.is_available())

                        print(f"  → Running Cellpose on slice {z+1}/{num_slices} crop {crop_idx+1}/4 ...")
                        masks, _, _ = model.eval(cropped_slice, diameter=None)
                        imwrite(slice_cpose_path, masks)
                        print(f"  → Saved Cellpose mask to '{slice_cpose_path.name}'")
                        initial_mask = masks
                    else:
                        print(f"  → Loaded {mask_source} for slice {z+1}/{num_slices} crop {crop_idx+1}/4")

                    # Convert to RGB if needed and ensure correct format
                    if cropped_slice.ndim == 2:
                        display_image = np.stack([cropped_slice] * 3, axis=-1)
                    else:
                        display_image = cropped_slice

                    # Queue slice and its mask for curation
                    data_for_curation.append({
                        "image": display_image,
                        "masks": initial_mask,
                        "title": crop_key,
                        "path": slice_curated_path
                    })
                
                z += step
                #step += 1
        else:
            # Handle 2D image
            print("  → Processing 2D image")

            # Crop the image (returns 4 crops)
            cropped_images = center_crop(raw_image, args.crop_size)
            for crop_idx, cropped_image in enumerate(cropped_images):
                crop_suffix = f"-c{crop_idx+1}"
                crop_key = f"{folder_key}{crop_suffix}"
                slice_path = crops_dir / f"{crop_key}_crop.tif"
                slice_cpose_path = cpose_masks_dir / f"{crop_key}_mask.tif"
                slice_curated_path = curated_masks_dir / f"{crop_key}_curated.tif"

                # Save the cropped image
                imwrite(slice_path, cropped_image)
                print(f"  → Saved cropped image crop {crop_idx+1}/4 to '{slice_path.name}'")

                initial_mask = None
                mask_source = "newly generated"

                if load_existing:
                    if slice_curated_path.exists():
                        initial_mask = imread(slice_curated_path)
                        mask_source = "existing curated mask"
                    elif slice_cpose_path.exists():
                        initial_mask = imread(slice_cpose_path)
                        mask_source = "existing cellpose mask"

                # Generate mask if needed
                if initial_mask is None:
                    if model is None:
                        print("\n--- Initializing Cellpose model for remaining images ---")
                        model = CellposeModel(gpu=torch.cuda.is_available())

                    print(f"  → Running Cellpose to generate initial mask crop {crop_idx+1}/4 ...")
                    masks, _, _ = model.eval(cropped_image, diameter=None)
                    imwrite(slice_cpose_path, masks)
                    print(f"  → Saved Cellpose mask to '{slice_cpose_path.name}'")
                    initial_mask = masks
                else:
                    print(f"  → Loaded {mask_source} for crop {crop_idx+1}/4.")

                # Convert to RGB if needed and ensure correct format
                if cropped_image.ndim == 2:
                    display_image = np.stack([cropped_image] * 3, axis=-1)
                else:
                    display_image = cropped_image

                # Queue image and its mask for curation
                data_for_curation.append({
                    "image": display_image,
                    "masks": initial_mask,
                    "title": crop_key,
                    "path": slice_curated_path
                })

    # --- Phase 2: Interactive Curation ---
    if not data_for_curation:
        print("\nNo images to process or curate. All done! ✅")
        return

    # Sort data for curation: c1, then c2, etc. to view all crops of a certain number together.
    data_for_curation.sort(key=lambda item: (int(item['title'].split('-c')[-1]), item['title']))

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

