from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import torchaudio
import wavencoder
import random
import numpy as np

class SREDataset(Dataset):
    def __init__(self, hparams, data_type, is_train):
        self.df_full = pd.read_csv(hparams.speaker_csv_path)
        self.df = self.df_full[self.df_full['Use'] == data_type].reset_index(drop=True)
        self.gender_dict = {'m' : 0, 'f' : 1}
        self.is_train = is_train
        self.a_mean = self.df_full[self.df_full['Use'] == 'train']['age'].mean()
        self.a_std = self.df_full[self.df_full['Use'] == 'train']['age'].std()

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        utt_id = self.df.loc[idx, 'utt_id']
        gender = self.gender_dict[self.df.loc[idx, 'Sex']]
        age = self.df.loc[idx, 'age']
        age = (age - self.a_mean)/self.a_std
        wav_path = self.df.loc[idx, 'wav_path']

        wav, f = torchaudio.load(wav_path)
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)

        return utt_id, wav, torch.FloatTensor([age]), torch.FloatTensor([gender])
