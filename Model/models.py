import torch
import torch.nn as nn
import fairseq
from functools import partial

class Wav2vec2BiEncoderAgeEstimation(nn.Module):
    def __init__(self, upstream_model='wav2vec2_local',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model, ckpt='/home/project/12001458/ductuan0/ISCAP_Age_Estimation/libri960_basemodel_sre0810_finetune_48epoch.pt')
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.age_regressor = nn.Linear(1536, 1)

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        layer = 'hidden_state_9'
        while True:
            list_layer = self.upstream(x)
            if layer in list_layer.keys():
                x = list_layer[layer]
                break
        x = self.transformer_encoder(x)
        x = torch.cat((torch.mean(x, dim=1), torch.std(x, dim=1)), dim=1)
        age = self.age_regressor(x)
        return age

