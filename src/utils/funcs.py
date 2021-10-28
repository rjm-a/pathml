import torch
import pandas as pd
import numpy as np


def get_sample_weights(labels_path):
    sample = pd.read_csv(labels_path)
    target = sample['action_id'].to_numpy()
    class_sample_count = np.unique(target, return_counts=True)[1]

    # tot = np.sum(class_sample_count)
    # print(tot)
    # per = class_sample_count / tot
    # print(per)

    wts = 1. / class_sample_count
    sample_weights = wts[target]
    sample_weights = torch.from_numpy(sample_weights).double()

    assert len(sample_weights) == len(sample)

    return sample_weights, len(sample)
