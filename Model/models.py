import torch
import torch.nn as nn
from conformer.encoder import ConformerEncoder
from IPython import embed

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

class UpstreamTransformerCls(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False

        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(feature_dim, 1024)

        self.dropout = nn.Dropout(0.5)
        
        self.age_regressor = nn.Linear(1024, 1)
        self.height_classifier = nn.Sequential(
            nn.Linear(1024, 7),
            nn.Softmax()
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(self.fc(x))
        height = self.height_classifier(x)
        age = self.age_regressor(x)
        gender = self.gender_classifier(x)
        return height, age, gender

class UpstreamTransformerCls2(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False

        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(2*feature_dim, 1024)

        self.dropout = nn.Dropout(0.5)
        
        self.age_regressor = nn.Linear(1024, 1)
        self.height_classifier = nn.Sequential(
            nn.Linear(1024, 7),
            nn.Softmax()
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = self.transformer_encoder(x)
        x = self.dropout(torch.cat((torch.mean(x, dim=1), torch.std(x, dim=1)), dim=1))
        x = self.dropout(self.fc(x))
        height = self.height_classifier(x)
        age = self.age_regressor(x)
        gender = self.gender_classifier(x)
        return height, age, gender    
    
class UpstreamTransformerCls3(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False

#         for param in self.upstream.model.encoder.layers.parameters():
#             param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
                param.requires_grad = True
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.age_regressor = nn.Linear(feature_dim, 1)
        self.height_classifier = nn.Sequential(
            nn.Linear(feature_dim, 7),
            nn.Softmax()
        )
        self.gender_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        height = self.height_classifier(x)
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

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        output = self.transformer_encoder(x)
        output_averaged = torch.mean(output, dim=1)
        height = self.height_regressor(output_averaged)
        return height
    