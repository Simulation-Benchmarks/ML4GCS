# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:40:34 2026

@author: saeid
"""

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class Plotter:
    @staticmethod
    def plot_losses(history):
        plt.figure()
        plt.plot(history["train_loss"], label="Train loss")
        plt.plot(history["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE loss")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_predictions(y_true, y_pred):       
        r2 = r2_score(y_true, y_pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.7)
    
        # x = y line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val],[min_val, max_val],'r-', linewidth=2)
    
        plt.xlabel("True values")
        plt.ylabel("Predicted values")
        plt.title("Prediction vs. True")
    
        # Display R² on plot
        plt.text(
            0.05, 0.95,
            f'$R^2$ = {r2:.4f}',
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
        #plt.legend()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred):
        residuals = y_true - y_pred

        plt.figure()
        plt.scatter(y_pred, residuals)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.title("Residual plot")
        plt.show()