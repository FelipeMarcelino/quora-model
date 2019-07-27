import torch
import pandas as pd
import numpy as np
from ast import literal_eval
from torch.utils.data import Dataset

class QuoraDataset(Dataset):

    def __init__(self, data, transform=None):

        #(FIX) - Transfrom string to array
        data["question1"] = data["question1"].apply(lambda x: np.asarray(x.strip('[]').replace("\
                                                                                               ","").split(",")).astype(int))

        data["question2"] = data["question2"].apply(lambda x: np.asarray(x.strip('[]').replace("\
                                                                                               ","").split(",")).astype(int))
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
        id_row = self.data[idx,0]
        id_q1 = self.data[idx,1]
        id_q2 = self.data[idx,2]

        return question_1, q1_len, question_2, q2_len, label, id_row, id_q1, id_q2
