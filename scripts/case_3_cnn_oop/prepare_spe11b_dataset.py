# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:12:58 2026

@author: saeid
"""
import numpy as np
import pandas as pd
import json
import h5py

data_filepath           = 'spe11b_tmco2_dt50y.npz'
distance_filepath       = 'spe11b_distances_dt50y.npz'
metadata_filepath       = 'spe11b_metadata_dt50y.json'
output_h5               = 'spe11b_dataset_dt50y.h5'
img_size                = [840, 120]

# read image pairs and distance
distance_metadata = np.load(distance_filepath, allow_pickle=True)
pairs       = distance_metadata["pairs"]
distances   = distance_metadata["distances"]
image_pair_df = pd.DataFrame({
        "img1"                  : pairs[:, 0],
        "img2"                  : pairs[:, 1],
        "wasserstein_distance"  : distances
})
print(image_pair_df.head(5), '\n')
# check NaNs in wasserstein distance
print("NaNs in wasserstein_distance (will be replaced by zero):",
      image_pair_df["wasserstein_distance"].isna().sum(), '\n')
image_pair_df["wasserstein_distance"] = image_pair_df["wasserstein_distance"].fillna(0)

# read metadata
with open(metadata_filepath, "r") as f:
    metadata = json.load(f)
image_df = pd.DataFrame(
    metadata,
    columns=["simulation", "time_year"]
)
image_df["image_id"] = range(len(image_df))

# read images
with np.load(data_filepath, allow_pickle=True) as npz:
    images = npz["global_array"]
images = images.T.astype(np.float32)
print(type(images))
# [34x21=714, 840x120=100800], (34 results x 21 snapshots (every 50y) = 714)
print(images.shape) # expected: (714, 100800)
# check NaNs in all images
print("NaNs in images:", np.isnan(images).sum())
nan_image_ids = np.where(np.isnan(images).any(axis=1))[0]
print("Images containing NaNs (will be replaced by zero):", nan_image_ids, '\n')
# replace NaNs in images with 0
images = np.nan_to_num(images, nan=0.0)

assert images.shape[0] == len(image_df), \
    "Number of images and metadata rows do not match!"

# add images to dataframe
image_df["image"] = list(images)
print(image_df.head())
print(image_df.shape)

# save everything into one HDF5 file
with h5py.File(output_h5, "w") as f:
    f.create_dataset("images", data=images, compression="gzip")
    f.create_dataset("simulation", data=image_df["simulation"].astype("S").values)
    f.create_dataset("time_year", data=image_df["time_year"].values)
    f.create_dataset("img1", data=image_pair_df["img1"].astype("S").values)
    f.create_dataset("img2", data=image_pair_df["img2"].astype("S").values)
    f.create_dataset("wasserstein_distance",
                     data=image_pair_df["wasserstein_distance"].values)

print("HDF5 dataset saved successfully:", output_h5)