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
        self.gender_dict = {'M' : 0, 'F' : 1}

        new_files = []
        for file in self.files:
            id = file.split('_')[0][1:]
            gender = self.gender_dict[self.df.loc[id, 'Sex']]
            if(gender == 0):
                new_files.append(file)
        self.files = new_files

        self.train_transform = None

        self.test_transform = None
        
        self.padCropTransform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=64000, pad_position='center', crop_position='center'),
        ])
        
        self.mfccTransform = torchaudio.transforms.MFCC()
        self.cmvn = torchaudio.transforms.SlidingWindowCmn(norm_vars=True)
        self.deltaTransform = torchaudio.transforms.ComputeDeltas()

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
        assert (gender == 0)

        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file), normalize=True)
        
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
        
        croppedPaddedWav = self.padCropTransform(wav)
        #LPC feature
#         lpccFeature = lpcc(sig=croppedPaddedWav.numpy(), fs=16000, num_ceps=20, normalize=True)
#         lpccFeatureDelta1 = librosa.feature.delta(np.transpose(lpccFeature), order=1)
#         lpccFeatureDelta1 = np.transpose(lpccFeatureDelta1)
#         lpccFeatureCombined = np.concatenate((lpccFeature, lpccFeatureDelta1), axis=1)
#         lpccFeatureCombined = torch.tensor(np.transpose(lpccFeatureCombined))
#         lpccFeatureCombined = lpccFeatureCombined.unsqueeze(dim=0)
        
        #MFCC feature
        mfccFeature = self.mfccTransform(croppedPaddedWav)
        mfccDeltaFeature = self.deltaTransform(mfccFeature)
        mfccFeature = self.cmvn(mfccFeature)
        mfccDeltaFeature = self.cmvn(mfccDeltaFeature)
        mfccFeaturesCombined = torch.cat((mfccFeature, mfccDeltaFeature), dim=0)
        mfccFeaturesCombined = torch.nan_to_num(mfccFeaturesCombined)
        assert (torch.isnan(mfccFeaturesCombined).any() == False)
        
        return wav,  mfccFeaturesCombined, torch.FloatTensor([height]), torch.FloatTensor([age]), torch.FloatTensor([gender])
    