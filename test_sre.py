from config import ModelConfig

from argparse import ArgumentParser
from multiprocessing import Pool
import os

from TIMIT.dataset import TIMITDataset
from SRE.dataset import SREDataset

from TIMIT.lightning_model_uncertainty_loss import LightningModel

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pytorch_lightning as pl

import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn_utils

from tqdm import tqdm 
import pandas as pd
import numpy as np
from statistics import mean, mode


def collate_fn(batch):
    (seq, age, gender) = zip(*batch)
    seql = [x.reshape(-1,) for x in seq]
    seq_length = [x.shape[0] for x in seql]
    data = rnn_utils.pad_sequence(seql, batch_first=True, padding_value=0)
    return data, age, gender, seq_length

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=ModelConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=ModelConfig.speaker_csv_path)
    parser.add_argument('--batch_size', type=int, default=ModelConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=ModelConfig.epochs)
    parser.add_argument('--num_layers', type=int, default=ModelConfig.num_layers)
    parser.add_argument('--feature_dim', type=int, default=ModelConfig.feature_dim)
    parser.add_argument('--lr', type=float, default=ModelConfig.lr)
    parser.add_argument('--gpu', type=int, default=ModelConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=ModelConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=ModelConfig.model_checkpoint)
    parser.add_argument('--upstream_model', type=str, default=ModelConfig.upstream_model)
    parser.add_argument('--model_type', type=str, default=ModelConfig.model_type)
    parser.add_argument('--narrow_band', type=str, default=ModelConfig.narrow_band)
    
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    # Check device
    if not torch.cuda.is_available():
        device = 'cpu'
        hparams.gpu = 0
        print(f'Testing Model on SRE Dataset on CPU')
    else:        
        device = 'cuda'
        print(f'Testing Model on SRE Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')
    
    # Testing Dataset
    test_set = SREDataset(
        hparams = hparams,
        data_type='test',
        is_train=True
    )

    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=12, 
        shuffle=False, 
        num_workers=hparams.n_workers,
        collate_fn = collate_fn,
    )

    csv_path = hparams.speaker_csv_path
    df = pd.read_csv(csv_path)
    a_mean = df[df['Use'] == 'train']['age'].mean()
    a_std = df[df['Use'] == 'train']['age'].std()

    #Testing the Model
    if hparams.model_checkpoint:
        model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=vars(hparams))
        model.to(device)
        model.eval()
        record2predgender_dict = {}
        record2predage_dict = {}
        record2labelgender_dict = {}
        record2labelage_dict = {}
        for batch in tqdm(testloader):
            utt_id, x, y_a, y_g, x_len = batch
            x = x.to(device)
            y_a = torch.stack(y_a).reshape(-1,)
            y_g = torch.stack(y_g).reshape(-1,)
            
            y_hat_h, y_hat_a, y_hat_g = model(x, x_len)
            y_hat_h = y_hat_h.to('cpu')
            y_hat_a = y_hat_a.to('cpu')
            y_hat_g = y_hat_g.to('cpu')
            age_pred = [age.item() * a_std + a_mean for age in y_hat_a]
            gender_pred = [gender.item() > 0.5 for gender in y_hat_g]
            age_true = [age.item() * a_std + a_mean for age in y_a]
            gender_true = y_g.tolist()

            for i, utt in enumerate(utt_id.tolist()):
                record_id = "_".join(utt.split("_")[:-2])
                if record_id in record2predgender_dict.keys():
                    record2predgender_dict[record_id].append(gender_pred[i])
                    record2predage_dict[record_id].append(age_pred[i])
                else:    
                    record2predgender_dict[record_id] = [gender_pred[i]]
                    record2predage_dict[record_id] = [age_pred[i]]
                    record2labelgender_dict[record_id] = gender_true[i]
                    record2labelage_dict[record_id] = age_true[i]

        age_pred = []
        gender_pred = []
        age_true = []
        gender_true = []
        for record_id in record2predgender_dict.keys():
            age_pred.append(mean(record2predage_dict[record_id]))
            gender_pred.append(mode(record2predgender_dict[record_id]))
            age_true.append(record2labelage_dict[record_id])
            gender_true.append(record2labelgender_dict[record_id])
        
        female_idx = np.where(np.array(gender_true) == 1)[0].reshape(-1).tolist()
        male_idx = np.where(np.array(gender_true) == 0)[0].reshape(-1).tolist()

        age_true = np.array(age_true)
        age_pred = np.array(age_pred)

        amae = mean_absolute_error(age_true[male_idx], age_pred[male_idx])
        armse = mean_squared_error(age_true[male_idx], age_pred[male_idx], squared=False)
        print(armse, amae)

        amae = mean_absolute_error(age_true[female_idx], age_pred[female_idx])
        armse = mean_squared_error(age_true[female_idx], age_pred[female_idx], squared=False)
        print(armse, amae)
        
        amae = mean_absolute_error(age_true, age_pred)
        armse = mean_squared_error(age_true, age_pred, squared=False)
        print(armse, amae)
        gender_pred_ = [int(pred == True) for pred in gender_pred]
        print(accuracy_score(gender_true, gender_pred_))
    else:
        print('Model chekpoint not found for Testing !!!')
