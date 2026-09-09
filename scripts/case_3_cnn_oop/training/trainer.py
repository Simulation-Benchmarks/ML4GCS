# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:25:39 2026

@author: saeid
"""

import torch
from training.metrics import regression_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_func,
                 device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.loss_func = loss_func

        self.history = {
            "train_loss": [],
            "train_mae":  [],
            "train_rmse":  [],
            "train_r2": [],
            "val_loss": [],
            "val_mae": [],
            "val_rmse": [],
            "val_r2": []
        }

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
    
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
    
            self.optimizer.zero_grad()
            pred = self.model(x)
    
            loss = self.loss_func(pred, y)
            loss.backward()
            self.optimizer.step()
    
            total_loss += loss.item()
            all_preds.append(pred.detach().cpu())
            all_targets.append(y.detach().cpu())
    
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
    
        metrics = regression_metrics(all_targets, all_preds)
    
        return total_loss / len(self.train_loader), metrics

    def _validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.loss_func(pred, y)

                total_loss += loss.item()
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()

        metrics = regression_metrics(all_targets, all_preds)

        return total_loss / len(self.val_loader), metrics

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss, train_metrics = self._train_one_epoch()
            val_loss, val_metrics = self._validate()
    
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
    
            self.history["train_mae"].append(train_metrics["mae"])
            self.history["val_mae"].append(val_metrics["mae"])
    
            self.history["train_rmse"].append(train_metrics["rmse"])
            self.history["val_rmse"].append(val_metrics["rmse"])
    
            self.history["train_r2"].append(train_metrics["r2"])
            self.history["val_r2"].append(val_metrics["r2"])
    
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train R2: {train_metrics['r2']:.4f} | "
                f"Val R2: {val_metrics['r2']:.4f}"
            )