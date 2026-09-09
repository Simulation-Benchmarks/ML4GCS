# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 09:57:27 2026

@author: saeid
"""

import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd
import numpy as np

def load_h5dataset(h5_path = "spe11b_dataset_dt50y.h5"):
    with h5py.File(h5_path, "r") as f:
        # read images
        images = f["images"][:].astype(np.float32)

        # read image dataframe
        image_df = pd.DataFrame({
            "simulation" : [x.decode() for x in f["simulation"][:]],
            "time_year"  : f["time_year"][:]
        })

        # read image pair dataframe
        image_pair_df = pd.DataFrame({
            "img1"                 : f["img1"][:].astype(int),
            "img2"                 : f["img2"][:].astype(int),
            "wasserstein_distance" : f["wasserstein_distance"][:].astype(np.float32)
        })

    return images, image_df, image_pair_df

# def create_image_pairs(image_df, image_pair_df, img_height=840, img_width=120):
#     """
#     Create paired image arrays X1, X2 and target y from image_df and image_pair_df.
#     """

#     # image name -> image vector
#     image_array = np.stack(image_df["image"].values)

#     # img1 and img2 are indices
#     img1_idx = image_pair_df["img1"].values.astype(int)
#     img2_idx = image_pair_df["img2"].values.astype(int)

#     # paired images
#     x1 = image_array[img1_idx]
#     x2 = image_array[img2_idx]

#     # target
#     y = image_pair_df["wasserstein_distance"].values.astype(np.float32)

#     # reshape flat images to 2D images
#     x1 = x1.reshape(-1, img_height, img_width).astype(np.float32)
#     x2 = x2.reshape(-1, img_height, img_width).astype(np.float32)

#     return x1, x2, y


class ImagePairDataset(Dataset):
    def __init__(self, images, image_pair_df,
                 img_height=840, img_width=120,
                 x_mean=None, x_std=None,
                 y_mean=None, y_std=None):

        self.images = np.array(images, dtype=np.float32)
        self.image_pair_df = image_pair_df.reset_index(drop=True)

        self.img_height = img_height
        self.img_width = img_width

        self.img1_ids = self.image_pair_df["img1"].values.astype(int)
        self.img2_ids = self.image_pair_df["img2"].values.astype(int)

        self.targets = self.image_pair_df["wasserstein_distance"].values.astype("float32")

        self.x_mean = x_mean if x_mean is not None else self.images.mean()
        self.x_std  = x_std  if x_std  is not None else self.images.std()

        self.y_mean = y_mean if y_mean is not None else self.targets.mean()
        self.y_std  = y_std  if y_std  is not None else self.targets.std()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
    
        img1 = self.images[self.img1_ids[idx]]
        img2 = self.images[self.img2_ids[idx]]
    
        img1 = img1.reshape(self.img_height, self.img_width)
        img2 = img2.reshape(self.img_height, self.img_width)
    
        img1 = (img1 - self.x_mean) / (self.x_std + 1e-8)
        img2 = (img2 - self.x_mean) / (self.x_std + 1e-8)
    
        y = self.targets[idx]
        y = (y - self.y_mean) / (self.y_std + 1e-8)
    
        img1 = torch.tensor(img1, dtype=torch.float32)
        img2 = torch.tensor(img2, dtype=torch.float32)
    
        # combine two images into two channels
        x = torch.stack([img1, img2], dim=0)
    
        y = torch.tensor([y], dtype=torch.float32)
    
        return x, y
    
    