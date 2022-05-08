from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import torchaudio
import wavencoder
import random
import numpy as np

class SREDataset(Dataset):
    def __init__(self, data_dir, data_type, is_train):
        self.csv_file = os.path.join(data_dir, 'data_info_age.csv')
        self.df_full = pd.read_csv(self.csv_file)
        self.df = self.df[self.df['Use'] == data_type]
        self.gender_dict = {'m' : 0, 'f' : 1}
        self.resampleUp = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
        self.is_train = is_train

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        gender = self.gender_dict[self.df.loc[idx, 'Sex']]
        age = self.df.loc[idx, 'age']
        wav_path = self.df.loc[idx, 'wav_path']

        wav, f = torchaudio.load(wav_path)
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
            wav = self.resampleUp(wav)

        a_mean = self.df_full[self.df_full['Use'] == 'train']['age'].mean()
        a_std = self.df_full[self.df_full['Use'] == 'train']['age'].std()
        age = (age - a_mean)/a_std


        probability = 0.5
        if self.is_train and random.random() <= probability:
            mixup_idx = random.randint(0, len(self.df.index)-1)
            mixup_wav_path = self.df.loc[mixup_idx, 'wav_path']
            mixup_gender = self.gender_dict[self.df.loc[mixup_idx, 'Sex']]
            mixup_age =  self.df.loc[mixup_idx, 'age']

            mixup_wav, _ = torchaudio.load(mixup_wav_path)

            if(mixup_wav.shape[0] != 1):
                mixup_wav = torch.mean(mixup_wav, dim=0) 
                mixup_wav = self.resampleUp(mixup_wav)

            mixup_age = (mixup_age - a_mean)/a_std
            
            if(mixup_wav.shape[1] < wav.shape[1]):
                cnt = (wav.shape[1]+mixup_wav.shape[1]-1)//mixup_wav.shape[1]
                mixup_wav = mixup_wav.repeat(1,cnt)[:,:wav.shape[1]]
            
            if(wav.shape[1] < mixup_wav.shape[1]):
                cnt = (mixup_wav.shape[1]+wav.shape[1]-1)//wav.shape[1]
                wav = wav.repeat(1,cnt)[:,:mixup_wav.shape[1]]
            
            alpha = 1
            lam = np.random.beta(alpha, alpha)
            
            wav = lam*wav + (1-lam)*mixup_wav
            age = lam*age + (1-lam)*mixup_age
            gender = lam*gender + (1-lam)*mixup_gender

        return wav, torch.FloatTensor([age]), torch.FloatTensor([gender])
