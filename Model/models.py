from turtle import down
import torch
import torch.nn as nn
from conformer.encoder import ConformerEncoder
from IPython import embed
from torchvision.models import resnet34, resnet50

class UpstreamTransformer(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        downstream_backbone = resnet50(pretrained=False)
        self.num_final_filters = downstream_backbone.fc.in_features
        feature_extractor_layer = list(downstream_backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*feature_extractor_layer)
        
        for param in self.upstream.parameters():
            param.requires_grad = True
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        self.height_regressor = nn.Sequential(
            nn.Linear(self.num_final_filters, 128),
            nn.Linear(128, 1),
        )
        self.age_regressor = nn.Sequential(
            nn.Linear(self.num_final_filters, 128),
            nn.Linear(128, 1),
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(self.num_final_filters, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [wav for (i,wav) in enumerate(x)]
        x = self.upstream(x)['last_hidden_state'].unsqueeze(1)
        x = torch.repeat_interleave(x, 3, dim=1)
        x = self.feature_extractor(x).squeeze().squeeze()
        height = self.height_regressor(x)
        age = self.age_regressor(x)
        gender = self.gender_classifier(x)
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
    
