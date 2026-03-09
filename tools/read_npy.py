#!/usr/bin/env python3
"""Read and inspect NPY file contents."""

import os
import argparse
import numpy as np


def read_npy(npy_path: str, show_image: bool = False):
    """
    Read an NPY file and print summary stats.
    
    Args:
        npy_path: Path to NPY file.
        show_image: Whether to display as an image.
    """
    npy_path = os.path.expanduser(npy_path)
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File not found: {npy_path}")
    
    arr = np.load(npy_path)
    
    print(f"File: {npy_path}")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Min: {arr.min():.6f}")
    print(f"Max: {arr.max():.6f}")
    print(f"Mean: {arr.mean():.6f}")
    print(f"Std: {arr.std():.6f}")
    
    if show_image:
        import matplotlib.pyplot as plt
        
        if arr.ndim == 2:
            plt.imshow(arr, cmap='viridis')
            plt.colorbar(label='Depth')
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                plt.imshow(arr[:, :, 0], cmap='viridis')
                plt.colorbar(label='Depth')
            elif arr.shape[2] == 3:
                plt.imshow(arr)
            elif arr.shape[2] == 4:
                plt.imshow(arr)
            else:
                plt.imshow(arr[:, :, 0], cmap='viridis')
                plt.colorbar(label='Depth')
        else:
            print(f"Cannot display {arr.ndim}D array as an image")
            return arr
        
        plt.title(os.path.basename(npy_path))
        plt.tight_layout()
        plt.show()
    
    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and inspect an NPY file")
    parser.add_argument("input", help="Path to input NPY file")
    parser.add_argument("-s", "--show", action="store_true", help="Display as an image")
    
    args = parser.parse_args()
    
    read_npy(args.input, args.show)
