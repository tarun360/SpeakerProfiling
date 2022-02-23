import torch
import torch.nn as nn
from conformer.encoder import ConformerEncoder
from IPython import embed

class UpstreamTransformer(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        self.mfcc_feature_dim = 128
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        self.transformer_encoder_1 = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True), num_layers=num_layers)
        #self.transformer_encoder_2 = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=self.mfcc_feature_dim, nhead=8, batch_first=True), num_layers=num_layers)
        
        self.final_feature_dim = 896
        
        self.height_regressor = nn.Sequential(
            nn.Linear(self.final_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.age_regressor = nn.Sequential(
            nn.Linear(self.final_feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(self.final_feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mfcc_x, x_len):
        x = [wav for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        output_1 = self.transformer_encoder_1(x)
        #output_2 = self.transformer_encoder_2(mfcc_x.squeeze(1).cuda())
        output_2 = mfcc_x.squeeze(1).cuda()
        output_averaged_1 = torch.mean(output_1, dim=1)
        output_averaged_2 = torch.mean(output_2, dim=1)
        concat_output = torch.cat((output_averaged_1, output_averaged_2), dim=-1)
        height = self.height_regressor(concat_output)
        age = self.age_regressor(concat_output)
        gender = self.gender_classifier(concat_output)
        return height, age, gender

# height only models

class UpstreamTransformerH(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.height_regressor = nn.Linear(feature_dim, 1)

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        output = self.transformer_encoder(x)
        output_averaged = torch.mean(output, dim=1)
        height = self.height_regressor(output_averaged)
        return height
    
