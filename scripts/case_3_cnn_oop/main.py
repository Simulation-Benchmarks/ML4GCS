# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:42:16 2026

@author: saeid
"""

import torch
from torch.utils.data import DataLoader
from data.dataset import ImagePairDataset, load_h5dataset
from sklearn.model_selection import train_test_split
from models.model_factory import ModelFactory
from training.trainer import Trainer
from training.tester import Tester
from utils.plotting import Plotter

dataset_filepath = "spe11b_dataset.h5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# read h5 file
images, image_df, image_pair_df = load_h5dataset()
print('Images = ', images.shape)
print('Image pairs = ', image_pair_df.shape)
print(image_pair_df.head(), "\n")

train_df, test_df = train_test_split(image_pair_df, test_size=0.1, 
                                     random_state=42, shuffle=True)
train_df, val_df  = train_test_split(train_df, test_size=0.22, # 20/(70+20)
                                     random_state=42, shuffle=True)

train_dataset = ImagePairDataset(images, train_df)
val_dataset   = ImagePairDataset(images, val_df,
                                x_mean=train_dataset.x_mean, x_std=train_dataset.x_std,
                                y_mean=train_dataset.y_mean, y_std=train_dataset.y_std)
test_dataset  = ImagePairDataset(images, test_df,
                                x_mean=train_dataset.x_mean, x_std=train_dataset.x_std,
                                y_mean=train_dataset.y_mean, y_std=train_dataset.y_std)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ModelFactory.create(model_name="simple_cnn", in_channels=2, dropout=0.2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_func =  torch.nn.MSELoss()
 
trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                  optimizer=optimizer, loss_func=loss_func, device=device)

trainer.fit(epochs=50)
Plotter.plot_losses(trainer.history)

tester = Tester(model, test_loader, device)
test_metrics, y_true, y_pred = tester.evaluate()
print(test_metrics)

Plotter.plot_predictions(y_true, y_pred)
Plotter.plot_residuals(y_true, y_pred)

