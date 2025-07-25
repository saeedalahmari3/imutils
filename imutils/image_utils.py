# image_utils.py

import os
import numpy as np
from tifffile import imread

def build_raw_group_map(raw_dir):
    """
    Scans a directory for .tif files and groups them by a unique key
    derived from their filenames.
    """
    print("Building image group map from raw files...")
    tiff_paths = [os.path.join(raw_dir, f)
                  for f in os.listdir(raw_dir)
                  if f.lower().endswith('.tif')]
    raw_groups = {}
    for path in tiff_paths:
        parts = os.path.basename(path).split('_')
        # Construct key from parts, excluding the Z-stack index
        key = '_'.join(parts[:len(parts)-4] + parts[len(parts)-3:])
        root, _ = os.path.splitext(key)
        raw_groups.setdefault(root, []).append(path)
    print(f"Found {len(raw_groups)} unique image groups.")
    return raw_groups

def get_raw_group_from_key(key: str, raw_groups: dict) -> list:
    """
    Retrieves the list of file paths for a given unique key from the raw_groups map.

    Args:
        key (str): The unique identifier for an image group.
                   e.g., "MCF10A_A00-IncucyteRawDataLiveDead-varyGlucose-241015_2N-Ctrl_B2_4_00d00h00m"
        raw_groups (dict): The dictionary created by build_raw_group_map.

    Returns:
        list: A list of file paths corresponding to the key. Returns an empty list if the key is not found.
    """
    return raw_groups.get(key, [])

def robust_normalize(arr):
    """Normalizes an array using 1st and 99th percentiles to resist outliers."""
    arr = arr.astype(np.float32)
    p1, p99 = np.percentile(arr, [1, 99])
    clipped = np.clip(arr, p1, p99)
    if p99 > p1:
        normalized = (clipped - p1) / (p99 - p1) * 255.0
    else:
        normalized = np.zeros_like(arr)
    return normalized.astype(np.uint8)

def make_composite(root, raw_groups):
    """
    Creates a 3-channel RGB composite image from raw TIFF channels for a given root.
    - Red channel: 'dead' stain
    - Green channel: 'alive' stain
    - Blue channel: 'phase'
    """
    raw_imgs = {'phase': [], 'alive': [], 'dead': []}
    for p in raw_groups.get(root, []):
        name = os.path.basename(p).split('_')[-4].lower()
        img = imread(p)
        if name == 'phase':
            raw_imgs['phase'].append(img)
        elif 'alive' in name:
            raw_imgs['alive'].append(img)
        elif 'dead' in name:
            raw_imgs['dead'].append(img)

    imgs = {}
    for k in raw_imgs:
        if raw_imgs[k]:
            # Project the maximum intensity across the z-stack
            pmax_img = np.maximum.reduce(raw_imgs[k])
            imgs[k] = robust_normalize(pmax_img)

    # Ensure phase exists to define shape
    phase = imgs.get('phase', np.zeros_like(next(iter(imgs.values()))))
    alive = imgs.get('alive')
    dead = imgs.get('dead')

    comp = np.zeros((*phase.shape, 3), dtype=np.uint8)

    # Add dead signal to red channel (over phase)
    if dead is not None:
        comp[...,0] = np.add(phase, dead, where=(dead > 0), casting='unsafe')
    else:
        comp[...,0] = phase
    # Add alive signal to green channel (over phase)
    if alive is not None:
        comp[...,1] = np.add(phase, alive, where=(alive > 0), casting='unsafe')
    else:
        comp[...,1] = phase
    # Blue channel is just phase
    comp[...,2] = phase
    return comp

def make_cpose_input(root, raw_groups):
    """
    Creates a 2-channel image required for Cellpose segmentation.
    - Channel 1 (Cytoplasm): Phase contrast image
    - Channel 2 (Nuclei): Sum of all fluorescence images
    """
    phase_img = None
    other_imgs = []
    for p in raw_groups.get(root, []):
        name = os.path.basename(p).split('_')[-4].lower()
        img = imread(p)
        if name == 'phase':
            phase_img = img
        else:
            other_imgs.append(img)

    if phase_img is None: raise ValueError(f"No phase image found for {root}")
    if not other_imgs: raise ValueError(f"No non-phase images found for {root}")

    cytoplasm = robust_normalize(phase_img)
    # Sum all fluorescence channels to get a clear nuclei signal
    nuclei = robust_normalize(np.sum(other_imgs, axis=0))
    # Stack along the first axis for Cellpose [2, H, W]
    return np.stack([cytoplasm, nuclei], axis=0).astype(np.uint8)
