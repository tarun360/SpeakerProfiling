import torch
import torch.nn as nn
from conformer.encoder import ConformerEncoder
from IPython import embed
from .CompactBilinearPooling import CompactBilinearPooling

class UpstreamTransformer(nn.Module):
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
        self.age_regressor = nn.Linear(feature_dim, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        output = self.transformer_encoder(x)
        output_averaged = torch.mean(output, dim=1)
        height = self.height_regressor(output_averaged)
        age = self.age_regressor(output_averaged)
        gender = self.gender_classifier(output_averaged)
        return height, age, gender

class UpstreamTransformerBPooling(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream1 = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstream2 = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
#         self.upstream1.model.encoder.layers = self.upstream1.model.encoder.layers[0:9]
#         self.upstream2.model.encoder.layers = self.upstream2.model.encoder.layers[0:9]
        
        for param in self.upstream1.model.parameters():
            param.requires_grad = True
        for param in self.upstream2.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream1.model.feature_extractor.conv_layers[0:4].parameters():
                param.requires_grad = False
            for param in self.upstream2.model.feature_extractor.conv_layers[0:4].parameters():
                param.requires_grad = False
        
        self.bilinear_pooling =  CompactBilinearPooling(feature_dim, feature_dim, 1024)
            
        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        xlist = [wav for wav in x.squeeze(1)]
        x1 = self.upstream1(xlist)['last_hidden_state'].contiguous()
        x2 = self.upstream2(xlist)['last_hidden_state'].contiguous()
        output = self.bilinear_pooling(x1,x2)
        output_averaged = torch.mean(output, dim=(1))
        height = self.height_regressor(output_averaged)
        age = self.age_regressor(output_averaged)
        gender = self.gender_classifier(output_averaged)
        return height, age, gender

class UpstreamTransformer2(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream1 = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstream2 = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
#         self.upstream1.model.encoder.layers = self.upstream1.model.encoder.layers[0:9]
#         self.upstream2.model.encoder.layers = self.upstream2.model.encoder.layers[0:9]
        
        for param in self.upstream1.model.parameters():
            param.requires_grad = True
        for param in self.upstream2.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream1.model.feature_extractor.conv_layers[0:4].parameters():
                param.requires_grad = False
            for param in self.upstream2.model.feature_extractor.conv_layers[0:4].parameters():
                param.requires_grad = False
        
        self.fc = nn.Linear(768, 4096)
        
        self.height_regressor = nn.Linear(4096, 1)
        self.age_regressor = nn.Linear(4096, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        xlist = [wav for wav in x.squeeze(1)]
        x1 = self.upstream1(xlist)['last_hidden_state'].contiguous()
        x2 = self.upstream2(xlist)['last_hidden_state'].contiguous()
#         x1 = torch.transpose(x1,2,1).unsqueeze(3).contiguous()
#         x2 = torch.transpose(x2,2,1).unsqueeze(3).contiguous()
        x1_averaged = torch.mean(x1, dim=1)
        x2_averaged = torch.mean(x2, dim=1)
        out = self.fc(x1_averaged+x2_averaged)
        height = self.height_regressor(out)
        age = self.age_regressor(out)
        gender = self.gender_classifier(out)
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

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        output = self.transformer_encoder(x)
        output_averaged = torch.mean(output, dim=1)
        height = self.height_regressor(output_averaged)
        return height
    