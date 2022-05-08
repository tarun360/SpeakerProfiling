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
                       default='/home/project/12001458/ductuan0/ISCAP_Age_Estimation/data/',
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

data_df = pd.DataFrame(columns=['utt_id', 'speaker_id', 'wav_path', 'Use', 'Sex', 'age'], index=None)
for data_type in list_data_type:
    print('Processing {} dataset'.format(data_type))
    data_path = os.path.join(data_dir, data_type)
    speaker_age_dict = {}
    
    with open(os.path.join(data_path, 'utt2age'), 'r') as utt2age_file:
        for line in utt2age_file.readlines():
            line = line.strip('\n')
            utt_id, age = line.split(' ')
            speaker_id = utt_id.split('_')[0]
            if speaker_id.startswith('sre2010'):
                speaker_id = speaker_id[:14]
            if speaker_id not in speaker_age_dict.keys():
                speaker_age_dict[speaker_id] = age
    
    with open(os.path.join(data_path, 'wav.scp')) as utt2path_file:
        lst_data_record = []
        for line in utt2path_file:
            line = line.strip('\n')
            utt_id, wav_path = line.split(' ')
            speaker_id = utt_id.split('_')[0]
            if speaker_id.startswith('sre2010'):
                speaker_id = speaker_id[:14]
            data_record = {
                'utt_id': utt_id,
                'speaker_id': speaker_id,
                'wav_path': wav_path,
                'Use': data_type,
                'Sex': speaker_gender_dict[speaker_id],
                'age': speaker_age_dict[speaker_id]
            }
            lst_data_record.append(data_record)
    data_df = pd.concat([data_df, pd.DataFrame.from_records(lst_data_record)], ignore_index=True)
data_df.to_csv(os.path.join(data_dir, 'data_info_age.csv'), index=False)
print('Data saved at ', data_dir)
