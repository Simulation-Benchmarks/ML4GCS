# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:21:59 2026

@author: saeid
"""

from models.basic_cnn import simpleCNN


class ModelFactory:
    
    @staticmethod
    def create(model_name, **kwargs):
        if model_name == "simple_cnn":
            return simpleCNN(**kwargs)
        else:
            raise ValueError(f"The model name is unknowmn: {model_name}")