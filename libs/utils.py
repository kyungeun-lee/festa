import logging
from tqdm import tqdm
import time, pickle
import scipy.stats

from libs.data import *
from libs.transform import BinShuffling
from typing import Any, Dict
from collections import defaultdict, OrderedDict
from scipy.io import arff
import numpy as np
import pandas as pd
import os, torchvision, torch, openml, yaml
import sklearn.model_selection
import sklearn.datasets
import torch.nn.functional as F

def check_trained(savepath):
    train = True
    if os.path.exists(os.path.join(savepath, 'performance.npy')):
        train = False
    return train

from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, r2_score, log_loss
def evaluate(prediction, target, prediction_proba=None, tasktype="binclass", y_std=1., tabpfn=False):
    
    if not isinstance(target, np.ndarray):
        target = target.cpu().numpy()
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.cpu().numpy()
    if not isinstance(prediction_proba, np.ndarray):
        prediction_proba = prediction_proba.cpu().numpy()
    
    if tasktype == "multiclass":
        if (len(target.shape) == 1) & (len(prediction.shape) == 1):
            pass
        elif (len(target.shape) > 1) & (len(prediction.shape) == 1):
            target = np.argmax(target, axis=1)
        elif (len(target.shape) > 1) & (prediction.shape[1] == 1):
            target = np.argmax(target, axis=1)
    
    if tasktype in ['binclass', 'multiclass']:
        try:
            acc = accuracy_score(target, prediction)
        except ValueError:
            import IPython; IPython.embed()
        if not tabpfn:
            if (tasktype == "binclass") & (prediction_proba.shape[1] > 1):
                auc = roc_auc_score(target, prediction_proba[:, 1])
            elif tasktype == "binclass":
                auc = roc_auc_score(target, prediction_proba)
            else:
                try:
                    auc = roc_auc_score(target, prediction_proba, average='macro', multi_class='ovr')
                except ValueError:
                    if len(np.unique(target)) < prediction_proba.shape[1]:
                        prediction_proba = prediction_proba[:, np.unique(target)]
                        prediction_proba = prediction_proba / prediction_proba.sum(axis=1).reshape(-1, 1)
                        prediction_proba = np.nan_to_num(prediction_proba)
                        try:
                            auc = roc_auc_score(target, prediction_proba, average='macro', multi_class='ovr')
                        except ValueError:
                            auc = np.nan
                    else:
                        auc = np.nan
            try:
                logloss = log_loss(target, prediction_proba)
            except ValueError:
                logloss = np.nan
        else:
            auc, logloss = None, None
        return acc, auc, logloss
    else:
        assert tasktype == 'regression'
        rmse = mean_squared_error(target * y_std, prediction * y_std) ** 0.5
        r2 = r2_score(target * y_std, prediction * y_std)
        return rmse, r2

class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

class CosineAnnealingLR_Warmup(object):
    def __init__(self, optimizer, warmup_epochs, T_max, iter_per_epoch, base_lr, warmup_lr, eta_min, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.warmup_iter = self.iter_per_epoch * self.warmup_epochs
        self.cosine_iter = self.iter_per_epoch * (self.T_max - self.warmup_epochs)
        self.current_iter = (self.last_epoch + 1) * self.iter_per_epoch

        self.step()

    def get_current_lr(self):
        if self.current_iter < self.warmup_iter:
            current_lr = (self.base_lr - self.warmup_lr) / self.warmup_iter * self.current_iter + self.warmup_lr
        else:
            current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * (self.current_iter-self.warmup_iter) / self.cosine_iter)) / 2
        return current_lr

    def step(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        self.current_iter += 1


def CosineAnnealingParam(warmup_epochs, T_max, iter_per_epoch, current_iter, base_value, 
                         warmup_value=1e-8, eta_min=0):
    warmup_iter = iter_per_epoch * warmup_epochs
    cosine_iter = iter_per_epoch * (T_max - warmup_epochs)
    
    if current_iter < warmup_iter:
        return (base_value - warmup_value) / warmup_iter * current_iter + warmup_value
    else:
        return eta_min + (base_value - eta_min) * (1 + np.cos(np.pi * (current_iter - warmup_iter) / cosine_iter)) / 2

def saveresults(modelname, savepath, train_preds, test_preds, train_prob, test_prob, tasktype, train_score, test_score):
    if modelname.startswith("ssl"):
        with open(os.path.join(savepath, "train_preds.npy"), "wb") as f:
            pickle.dump(train_preds, f)
        with open(os.path.join(savepath, "test_preds.npy"), "wb") as f:
            pickle.dump(test_preds, f)
        if tasktype in ['binclass', 'multiclass']:
            with open(os.path.join(savepath, "train_probs.npy"), "wb") as f:
                pickle.dump(train_prob, f)
            with open(os.path.join(savepath, "test_probs.npy"), "wb") as f:
                pickle.dump(test_prob, f)
    elif modelname in ["stunt", "tabpfn"]:
        np.save(os.path.join(savepath, "train_preds.npy"), train_preds)
        np.save(os.path.join(savepath, "test_preds.npy"), test_preds)
    else:
        np.save(os.path.join(savepath, "train_preds.npy"), train_preds)
        np.save(os.path.join(savepath, "test_preds.npy"), test_preds)
        if tasktype in ['binclass', 'multiclass']:
            np.save(os.path.join(savepath, "train_probs.npy"), train_prob)
            np.save(os.path.join(savepath, "test_probs.npy"), test_prob)
    np.save(os.path.join(savepath, "performance.npy"), dict({"Train": train_score, "Test": test_score}))
    
def load_config(config_filename, shot=None):
    with open(f'/home/SemiTab/configs/{config_filename}', 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    modelname = configs["modelname"]
    if modelname == "knn":
        configs["params"] = dict({"k": shot})
    elif modelname.startswith("ssl") or (modelname == "subtab"):
        configs["params"]["k"] = shot
    return configs