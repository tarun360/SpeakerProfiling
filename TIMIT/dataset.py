from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder

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
        self.gender_type = hparams.gender_type
        self.speed_change = hparams.speed_change

        self.speaker_list = self.df.loc[:, 'ID'].values.tolist()
        self.df.set_index('ID', inplace=True)
        self.gender_dict = {'M' : 0, 'F' : 1}

        new_files = []
        if self.gender_type == None:
            self.list_gender = ['M', 'F'] # contain both male and female
        else:
            self.list_gender = [self.gender_type]
        for file in self.files:
            id = file.split('_')[0][1:]
            gender = self.df.loc[id, 'Sex']
            if gender in self.list_gender:
                new_files.append(file)
        self.files = new_files
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

        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file), normalize=True)
        
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)

        if self.is_train and self.train_transform:
            wav = self.train_transform(wav)  
        
        h_mean = self.df[(self.df['Use'] == 'TRN') & (self.df['Sex'].isin(self.list_gender))]['height'].mean()
        h_std = self.df[(self.df['Use'] == 'TRN') & (self.df['Sex'].isin(self.list_gender))]['height'].std()
        a_mean = self.df[(self.df['Use'] == 'TRN') & (self.df['Sex'].isin(self.list_gender))]['age'].mean()
        a_std = self.df[(self.df['Use'] == 'TRN') & (self.df['Sex'].isin(self.list_gender))]['age'].std()
                
        height = (height - h_mean)/h_std
        age = (age - a_mean)/a_std
    
        
        return wav, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender])
    
