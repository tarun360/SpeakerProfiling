from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import torchaudio
import wavencoder


class SREDataset(Dataset):
    def __init__(self,
    data_dir,
    data_type,
    ):
        self.data_type = data_type
        self.data_dir = os.path.join(data_dir, data_type)
        self.csv_file = os.path.join(self.data_dir, '{}.csv'.format(data_type))
        self.df = pd.read_csv(self.csv_file)
        self.gender_dict = {'m' : 0, 'f' : 1}
        self.resampleUp = torchaudio.transforms.Resample(orig_freq=8000, new_freq=16000)

    def __len__(self):
        return len(self.df.index)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        gender = self.gender_dict[self.df.loc[idx, 'speaker_gender']]
        age =  self.df.loc[idx, 'speaker_age']
        wav_path = self.df.loc[idx, 'utt_wav_path']

        wav, f = torchaudio.load(wav_path)
        if(wav.shape[0] != 1):
            wav = torch.mean(wav, dim=0)
        wav = self.resampleUp(wav)
        # h_mean = self.df[self.df['Use'] == 'TRN']['height'].mean()
        # h_std = self.df[self.df['Use'] == 'TRN']['height'].std()
        # a_mean = self.df[self.df['Use'] == 'TRN']['age'].mean()
        # a_std = self.df[self.df['Use'] == 'TRN']['age'].std()
        # 
        # height = (height - h_mean)/h_std
        # age = (age - a_mean)/a_std

        return wav, torch.FloatTensor([age]), torch.FloatTensor([gender])
