# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:38:40 2026

@author: saeid
"""

import torch
from training.metrics import regression_metrics


class Tester:
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def _predict(self):
        self.model.eval()
        preds = []
        targets = []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)

                pred = self.model(x)

                preds.append(pred.cpu())
                targets.append(y)

        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()

        return targets, preds

    def evaluate(self):
        y_true, y_pred = self._predict()
        return regression_metrics(y_true, y_pred), y_true, y_pred