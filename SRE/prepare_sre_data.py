import os
import shutil
import argparse
import scipy as sp
from sklearn.model_selection import train_test_split
import pandas as pd

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--path',
                       metavar='path',
                       type=str,
                       default='/home/tuantd/Documents/ISCAP_Age_Estimation/data/',
                       help='the path to dataset folder')
args = my_parser.parse_args()


data_dir = args.path

speaker_gender_dict = {}    
with open(os.path.join(data_dir, 'all', 'spk2gender'), 'r') as spk2gender:
    for line in spk2gender.readlines():
        line = line.strip('\n')
        speaker_id, gender = line.split(' ')
        if speaker_id not in speaker_gender_dict.keys():
            speaker_gender_dict[speaker_id] = gender

list_data_type = ['train', 'valid', 'test']

for data_type in list_data_type:
    data_path = os.path.join(data_dir, data_type)
    data_df = pd.DataFrame(columns=['utt_id', 'speaker_id', 'utt_wav_path', 'speaker_gender', 'speaker_age'], index=None)
    speaker_age_dict = {}
    
    with open(os.path.join(data_path, 'utt2age'), 'r') as utt2age_file:
        for line in utt2age_file.readlines():
            i += 1 
            line = line.strip('\n')
            utt_id, age = line.split(' ')
            speaker_id = utt_id.split('_')[0]
            if speaker_id not in speaker_age_dict.keys():
                speaker_age_dict[speaker_id] = age
    
    with open(os.path.join(data_path, 'wav.scp')) as utt2path_file:
        i = 0 
        for line in utt2path_file:
            i += 1 
            line = line.strip('\n')
            utt_id, utt_wav_path = line.split(' ')
            speaker_id = utt_id.split('_')[0]
            data_record = {
                'utt_id': utt_id,
                'speaker_id': speaker_id,
                'utt_wav_path': utt_wav_path,
                'speaker_gender': speaker_gender_dict[speaker_id],
                'speaker_age': speaker_age_dict[speaker_id]
            }
            print(data_record)
            data_df = pd.concat([data_df, pd.DataFrame.from_records([data_record])], ignore_index=True)
    data_df.to_csv(os.path.join(data_path, '{}.csv'.format(data_type)), index=False)
    print('Data saved at ', data_path)