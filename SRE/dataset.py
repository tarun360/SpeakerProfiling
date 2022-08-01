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
        self.resampleUp = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)
        self.is_train = is_train
        self.a_mean = self.df_full[self.df_full['Use'] == 'train']['age'].mean()
        self.a_std = self.df_full[self.df_full['Use'] == 'train']['age'].std()
        # self.pad_crop_transform = wavencoder.transforms.Compose([
        #         wavencoder.transforms.PadCrop(pad_crop_length=3*16000, pad_position='center', crop_position='center'),    
        #     ])

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        utt_id = self.df.loc[idx, 'utt_id']
        gender = self.gender_dict[self.df.loc[idx, 'Sex']]
        age = self.df.loc[idx, 'age']
        wav_path = self.df.loc[idx, 'wav_path']

        wav, f = torchaudio.load(wav_path)
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
            # wav = self.pad_crop_transform(wav)

        wav = self.resampleUp(wav)
        age = (age - self.a_mean)/self.a_std


        probability = 0.5
        if self.is_train and random.random() <= probability:
            mixup_idx = random.randint(0, len(self.df.index)-1)
            mixup_wav_path = self.df.loc[mixup_idx, 'wav_path']
            mixup_gender = self.gender_dict[self.df.loc[mixup_idx, 'Sex']]
            mixup_age =  self.df.loc[mixup_idx, 'age']

            mixup_wav, _ = torchaudio.load(mixup_wav_path)

            if(mixup_wav.shape[0] != 1):
                mixup_wav = torch.mean(mixup_wav, dim=0) 
                # mixup_wav = self.pad_crop_transform(mixup_wav)

            mixup_wav = self.resampleUp(mixup_wav)

            mixup_age = (mixup_age - self.a_mean)/self.a_std
            
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

        return utt_id, wav, torch.FloatTensor([age]), torch.FloatTensor([gender])
