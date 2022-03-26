import torch
import torch.nn as nn
import torch.nn.functional as F
# torch.use_deterministic_algorithms(True)

import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError  as MSE
from pytorch_lightning.metrics.classification import Accuracy

import pandas as pd
import torch_optimizer as optim
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


from Model.models import UpstreamTransformerA

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.models = {
            'UpstreamTransformerA': UpstreamTransformerA
        }
        
        self.model = self.models[HPARAMS['model_type']](upstream_model=HPARAMS['upstream_model'], num_layers=HPARAMS['num_layers'], feature_dim=HPARAMS['feature_dim'], unfreeze_last_conv_layers=HPARAMS['unfreeze_last_conv_layers'])
            
        self.classification_criterion = MSE()
        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()
        self.accuracy = Accuracy()

        self.lr = HPARAMS['lr']

        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path)
        if HPARAMS['gender_type'] is None:
            self.list_gender = [0, 1]
        else:
            self.list_gender = [HPARAMS['gender_type']]
        self.h_mean = self.df[(self.df['Use'] == 'TRN') & (self.df['Sex'].isin(self.list_gender))]['height'].mean()
        self.h_std = self.df[(self.df['Use'] == 'TRN') & (self.df['Sex'].isin(self.list_gender))]['height'].std()
        self.a_mean = self.df[(self.df['Use'] == 'TRN') & (self.df['Sex'].isin(self.list_gender))]['age'].mean()
        self.a_std = self.df[(self.df['Use'] == 'TRN') & (self.df['Sex'].isin(self.list_gender))]['age'].std()

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
            return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_len):
        return self.model(x, x_len)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y_h, y_a, y_g, x_len = batch
        y_h = torch.stack(y_h).reshape(-1,)
        y_a = torch.stack(y_a).reshape(-1,)
        y_g = torch.stack(y_g).reshape(-1,)
        
        y_hat_a = self(x, x_len)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        age_loss = self.regression_criterion(y_hat_a, y_a)
        loss = age_loss 

        age_mae = self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)

        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False)

        return {'loss':loss, 
                'train_age_mae':age_mae.item(),
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        age_mae = torch.tensor([x['train_age_mae'] for x in outputs]).sum()/n_batch

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/a',age_mae.item(), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_h, y_a, y_g, x_len = batch
        y_h = torch.stack(y_h).reshape(-1,)
        y_a = torch.stack(y_a).reshape(-1,)
        y_g = torch.stack(y_g).reshape(-1,)
        
        y_hat_a = self(x, x_len)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        age_loss = self.regression_criterion(y_hat_a, y_a)
        loss = age_loss 

        age_mae = self.mae_criterion(y_hat_a*self.a_std+self.a_mean, y_a*self.a_std+self.a_mean)

        return {'val_loss':loss, 
                'val_age_mae':age_mae.item()
                }

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        age_mae = torch.tensor([x['val_age_mae'] for x in outputs]).sum()/n_batch
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/h',age_mae.item(), on_step=False, on_epoch=True, prog_bar=True)
   
    def test_step(self, batch, batch_idx):
        x, y_h, y_a, y_g, x_len = batch
        y_h = torch.stack(y_h).reshape(-1,)
        y_a = torch.stack(y_a).reshape(-1,)
        y_g = torch.stack(y_g).reshape(-1,)
        
        y_hat_a = self(x, x_len)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_a = y_hat_a.view(-1).float()

        idx = y_g.view(-1).long()
        female_idx = torch.nonzero(idx).view(-1)
        male_idx = torch.nonzero(1-idx).view(-1)

        if 0 in self.list_gender:
            male_age_mae = self.mae_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean).item()
            male_age_rmse = self.rmse_criterion(y_hat_a[male_idx]*self.a_std+self.a_mean, y_a[male_idx]*self.a_std+self.a_mean).item()
            female_age_mae = 0
            femal_age_rmse = 0
        if 1 in self.list_gender:
            male_age_mae = 0
            male_age_rmse = 0
            female_age_mae = self.mae_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean).item()
            femal_age_rmse = self.rmse_criterion(y_hat_a[female_idx]*self.a_std+self.a_mean, y_a[female_idx]*self.a_std+self.a_mean).item()
        return {
                'male_age_mae':male_age_mae,
                'female_age_mae':female_age_mae,
                'male_age_rmse':male_age_rmse,
                'femal_age_rmse':femal_age_rmse,
        }
    
    def test_epoch_end(self, outputs):
        n_batch = len(outputs)
        male_age_mae = torch.tensor([x['male_age_mae'] for x in outputs]).mean()
        female_age_mae = torch.tensor([x['female_age_mae'] for x in outputs]).mean()
    
        male_age_rmse = torch.tensor([x['male_age_rmse'] for x in outputs]).mean()
        femal_age_rmse = torch.tensor([x['femal_age_rmse'] for x in outputs]).mean()
    
    
        pbar = {'male_age_mae' : male_age_mae.item(),
                'female_age_mae':female_age_mae.item(),
                'male_age_rmse' : male_age_rmse.item(),
                'femal_age_rmse':femal_age_rmse.item(),
                }
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)
