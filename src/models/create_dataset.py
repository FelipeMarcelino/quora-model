import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class QuoraDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data.values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_1 = self.data[idx,3]
        question_2 = self.data[idx,4]
        label = self.data[idx,5]
        q1_len = self.data[idx,6]
        q2_len = self.data[idx,7]

        return question_1
