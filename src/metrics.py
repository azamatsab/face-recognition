import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd


def find_best_threshold(far_sim, frr_sim):
    thresholds = np.arange(0, 1, 0.01)
    best_thr = 0
    diff = 1
    result = {"far": [], "frr": [], "thresh": []}
    for thr in thresholds:
        far = np.mean(far_sim >= thr)
        frr = np.mean(frr_sim < thr)
        if abs(far - frr) < diff:
            diff = abs(far - frr)
            best_thr = thr
        result["far"].append(far)
        result["frr"].append(frr)
        result["thresh"].append(thr)
    
    return best_thr, result


def calc_all_metrics(embeddings1, embeddings2, labels):
    far_emb1 = embeddings1[labels == 0]
    far_emb2 = embeddings2[labels == 0]
    far_sim = (far_emb1 @ far_emb2.T).diagonal()

    frr_emb1 = embeddings1[labels == 1]
    frr_emb2 = embeddings2[labels == 1]
    frr_sim = (frr_emb1 @ frr_emb2.T).diagonal()

    thr, result = find_best_threshold(far_sim, frr_sim)
    eer = round((np.mean(far_sim >= thr) + np.mean(frr_sim < thr)) / 2, 4)
    accuracy = round(np.mean(np.concatenate([far_sim < thr, frr_sim >= thr])) * 100, 2)

    result["accuracy"] = accuracy
    result["EER"] = eer
    return result


class AverageMeter:
    def __init__(self):
        self.metrics = None
        self.num_samples = 0

    def add(self, metrics, num_samples):
        self.num_samples += num_samples

        if isinstance(metrics, dict):
            if self.metrics is None:
                self.metrics = OrderedDict([(metric, 0) for metric in metrics])
            for metric in metrics:
                self.metrics[metric] += metrics[metric].item() * num_samples
        else:
            if self.metrics is None:
                self.metrics = 0
            self.metrics += metrics.item() * num_samples

    def get(self):
        if isinstance(self.metrics, dict):
            for metric in self.metrics:
                self.metrics[metric] = round(self.metrics[metric] / self.num_samples, 4)
        else:
            self.metrics = round(self.metrics / self.num_samples, 4)
        
        if not isinstance(self.metrics, dict):
            self.metrics = {"loss": self.metrics}
        return self.metrics


class MetricCalculator:
    def __init__(self):
        self.emb1 = []
        self.emb2 = []
        self.targets = []

    def add(self, outputs):
        embeddings1, embeddings2, labels = outputs
        self.emb1.append(embeddings1)
        self.emb2.append(embeddings2)
        self.targets.append(labels)

    def get(self):
        self.emb1 = np.concatenate(self.emb1, axis=0)
        self.emb2 = np.concatenate(self.emb2, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)
        result = calc_all_metrics(self.emb1, self.emb2, self.targets)
        return result
