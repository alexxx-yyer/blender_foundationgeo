#!/usr/bin/env python3
"""读取并显示NPY文件内容"""

import os
import argparse
import numpy as np


def read_npy(npy_path: str, show_image: bool = False):
    """
    读取NPY文件并显示信息
    
    Args:
        npy_path: NPY文件路径
        show_image: 是否显示为图像
    """
    npy_path = os.path.expanduser(npy_path)
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"找不到文件: {npy_path}")
    
    arr = np.load(npy_path)
    
    print(f"文件: {npy_path}")
    print(f"形状: {arr.shape}")
    print(f"数据类型: {arr.dtype}")
    print(f"最小值: {arr.min():.6f}")
    print(f"最大值: {arr.max():.6f}")
    print(f"均值: {arr.mean():.6f}")
    print(f"标准差: {arr.std():.6f}")
    
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
            print(f"无法显示 {arr.ndim} 维数组为图像")
            return arr
        
        plt.title(os.path.basename(npy_path))
        plt.tight_layout()
        plt.show()
    
    return arr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="读取并显示NPY文件")
    parser.add_argument("input", help="输入的NPY文件路径")
    parser.add_argument("-s", "--show", action="store_true", help="显示为图像")
    
    args = parser.parse_args()
    
    read_npy(args.input, args.show)
