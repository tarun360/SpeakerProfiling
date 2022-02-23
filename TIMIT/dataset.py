from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
from torchaudio.transforms import MFCC
from IPython import embed

class TIMITDataset(Dataset):
    def __init__(self,
    wav_folder,
    hparams,
    is_train=True,
    ):
        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = hparams.speaker_csv_path
        self.df = pd.read_csv(self.csv_file)
        self.is_train = is_train
        self.noise_dataset_path = hparams.noise_dataset_path
        self.wav_len = hparams.wav_len
        self.data_type = hparams.data_type
        self.speed_change = hparams.speed_change

        self.speaker_list = self.df.loc[:, 'ID'].values.tolist()
        self.df.set_index('ID', inplace=True)
        self.gender_dict = {'M' : 0, 'F' : 1}

        if self.noise_dataset_path:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.AdditiveNoise(self.noise_dataset_path, p=0.5),
                wavencoder.transforms.Clipping(p=0.5),
                ])
        elif self.speed_change:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.SpeedChange(factor_range=(-0.1, 0.1), p=0.5),
                ])
        else:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='left', crop_position='random'),
                wavencoder.transforms.Clipping(p=0.5),
            ]) 

        self.test_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=hparams.wav_len, pad_position='left', crop_position='center')
        ])

    def __len__(self):
        return len(self.files)

    def get_age(self, idx):
        rec_date = self.df.loc[idx, 'RecDate'].split('/')
        birth_date = self.df.loc[idx, 'BirthDate'].split('/')
        m1, d1, y1 = [int(x) for x in birth_date]
        m2, d2, y2 = [int(x) for x in rec_date]
        return y2 - y1 - ((m2, d2) < (m1, d1))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file.split('_')[0][1:]
        g_id = file.split('_')[0]

        gender = self.gender_dict[self.df.loc[id, 'Sex']]
        height = self.df.loc[id, 'height']
        age =  self.df.loc[id, 'age']
        # self.get_age(id)

        wav, sig = torchaudio.load(os.path.join(self.wav_folder, file))
        mfcc_module = MFCC(sample_rate=sig, n_mfcc=128, melkwargs={"n_fft": 2048, "hop_length": 512, "power": 2})
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)

        if self.is_train and self.train_transform:
            wav = self.train_transform(wav)  
        else:
            wav = self.test_transform(wav)
        wav_mfcc = mfcc_module(wav).transpose(1, 2)
        h_mean = self.df[self.df['Use'] == 'TRN']['height'].mean()
        h_std = self.df[self.df['Use'] == 'TRN']['height'].std()
        a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        a_std = self.df[self.df['Use'] == 'TRN']['age'].std()
        
        height = (height - h_mean)/h_std
        age = (age - a_mean)/a_std

        return wav, wav_mfcc, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender])
