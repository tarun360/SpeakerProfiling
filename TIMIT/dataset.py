from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
from IPython import embed
import random
from spafe.features.lpc import lpc, lpcc
from spafe.features.mfcc import mfcc
import librosa

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

        self.train_transform = None

        self.test_transform = None
        
        self.padCropTransform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=48000, pad_position='random', crop_position='random'),
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
        
        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file), normalize=False)
        
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)

        if self.is_train and self.train_transform:
            wav = self.train_transform(wav)  
        
        h_mean = self.df[self.df['Use'] == 'TRN']['height'].mean()
        h_std = self.df[self.df['Use'] == 'TRN']['height'].std()
        a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        a_std = self.df[self.df['Use'] == 'TRN']['age'].std()
        
        height = (height - h_mean)/h_std
        age = (age - a_mean)/a_std
        
        croppedPaddedWav = self.padCropTransform(wav).numpy()
        #LPC feature
        lpccFeature = torch.tensor(lpcc(sig=croppedPaddedWav, fs=16000, num_ceps=20, normalize=True)).float()
        lpccFeatureDelta1 = librosa.feature.delta(np.transpose(lpccFeature), order=1)
        lpccFeatureDelta1 = np.transpose(lpccFeatureDelta1)
        lpccFeatureCombined = np.concatenate((lpccFeature, lpccFeatureDelta1), axis=1)
        lpccFeatureCombined = torch.tensor(np.transpose(lpccFeatureCombined))
        lpccFeatureCombined = lpccFeatureCombined.unsqueeze(dim=0)
        
        #MFCC feature
#         mfccFeature = torch.tensor(mfcc(sig=croppedPaddedWav, fs=16000, num_ceps=20)).float()
#         mfccFeatureDelta1 = librosa.feature.delta(np.transpose(mfccFeature), order=1)
#         mfccFeatureDelta1 = np.transpose(mfccFeatureDelta1)
#         mfccFeatureCombined = np.concatenate((mfccFeature, mfccFeatureDelta1), axis=1)
#         mfccFeatureCombined = torch.tensor(np.transpose(mfccFeatureCombined))
#         mfccFeatureCombined = mfccFeatureCombined.unsqueeze(dim=0)
        
#         lpcMfccCombined = torch.cat((lpccFeatureCombined, mfccFeatureCombined), dim=0)
        
        return wav, lpccFeatureCombined, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender])