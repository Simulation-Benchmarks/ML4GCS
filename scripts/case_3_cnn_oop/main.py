# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:42:16 2026

@author: saeid
"""

import torch
from torch.utils.data import DataLoader
from spe11_wasserstein import data
from models.model_factory import ModelFactory
from training.trainer import Trainer
from training.tester import Tester
from utils.plotting import Plotter

device = "cuda" if torch.cuda.is_available() else "cpu"

# D_train, D_val, D_test of data/spe11b.h5, with the arrays on the device
train_dataset, val_dataset, test_dataset = data.make_datasets(
    asarray=lambda array: torch.as_tensor(array, device=device))
print(f"Samples: {len(train_dataset)} train / {len(val_dataset)} validation / "
      f"{len(test_dataset)} test\n")

# default num_workers=0: CUDA tensors do not work reliably with worker processes
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = ModelFactory.create(model_name="simple_cnn", in_channels=2, dropout=0.2)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_func =  torch.nn.MSELoss()
 
trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                  optimizer=optimizer, loss_func=loss_func, device=device)

trainer.fit(epochs=1)
Plotter.plot_losses(trainer.history)

tester = Tester(model, test_loader, device)
test_metrics, y_true, y_pred = tester.evaluate()
print(test_metrics)

Plotter.plot_predictions(y_true, y_pred)
Plotter.plot_residuals(y_true, y_pred)

