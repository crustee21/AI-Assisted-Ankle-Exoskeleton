import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os

class FootDropDataset(Dataset):
    def __init__(self, data_dir, exclude_subj=None):

        inp = torch.from_numpy(np.load(os.path.join(data_dir, 'inputs.npy'))).float()
        lab = torch.from_numpy(np.load(os.path.join(data_dir, 'labels.npy'))).float()
        subject_ids = np.load(os.path.join(data_dir, 'subject_ids.npy'), allow_pickle=True)

        if (exclude_subj is None):
            self.inputs = inp
            self.labels = lab

        else:
            mask = np.array([s not in exclude_subj for s in subject_ids])
            self.inputs = inp[mask]
            self.labels = lab[mask]

    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.inputs[idx], self.labels[idx])
