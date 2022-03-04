from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
from IPython import embed
import random

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
        self.data_type = hparams.data_type
        self.speed_change = hparams.speed_change

        self.speaker_list = self.df.loc[:, 'ID'].values.tolist()
        self.df.set_index('ID', inplace=True)
        self.gender_dict = {'M' : 0.0, 'F' : 1.0}

        if self.is_train:
            self.padCropTransform = wavencoder.transforms.Compose([
                    wavencoder.transforms.PadCrop(pad_crop_length=32000, pad_position='random', crop_position='random'),
            ])
        else:
            self.padCropTransform = wavencoder.transforms.Compose([
                    wavencoder.transforms.PadCrop(pad_crop_length=32000, pad_position='center', crop_position='center'),
            ])    
        
        self.train_transform = None

        self.test_transform = None

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
        
        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file))
        
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
            
#         wav = self.padCropTransform(wav)
        
        h_mean = self.df[self.df['Use'] == 'TRN']['height'].mean()
        h_std = self.df[self.df['Use'] == 'TRN']['height'].std()
        a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        a_std = self.df[self.df['Use'] == 'TRN']['age'].std()
        
        height = (height - h_mean)/h_std
        age = (age - a_mean)/a_std
        
        probability = 0.5
        if self.is_train and random.random() <= probability:
            # https://towardsdatascience.com/enhancing-neural-networks-with-mixup-in-pytorch-5129d261bc4a
            mixup_idx = random.randint(0, len(self.files)-1)
            mixup_file = self.files[mixup_idx]
            mixup_id = mixup_file.split('_')[0][1:]
                
            mixup_gender = self.gender_dict[self.df.loc[mixup_id, 'Sex']]
            mixup_height = self.df.loc[mixup_id, 'height']
            mixup_age =  self.df.loc[mixup_id, 'age']

            mixup_wav, _ = torchaudio.load(os.path.join(self.wav_folder, mixup_file))

            if(mixup_wav.shape[0] != 1):
                mixup_wav = torch.mean(mixup_wav, dim=0)

#             mixup_wav = self.padCropTransform(mixup_wav)

            mixup_height = (mixup_height - h_mean)/h_std
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
            height = lam*height + (1-lam)*mixup_height
            age = lam*age + (1-lam)*mixup_age
            gender = lam*gender + (1-lam)*mixup_gender
            
        return wav, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender])