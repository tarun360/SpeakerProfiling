import torch
import torch.nn as nn

class UpstreamTransformer(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        for param in self.upstream.parameters():
            param.requires_grad = False

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
    
class UpstreamConformer(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        for param in self.upstream.parameters():
            param.requires_grad = False

        self.conformer_encoder = ConformerEncoder(encoder_dim=feature_dim, num_attention_heads=8, num_layers=num_layers)
        self.height_regressor = nn.Linear(feature_dim, 1)
        self.age_regressor = nn.Linear(feature_dim, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        output = self.conformer_encoder(x, x.shape[0])[0]
        output_averaged = torch.mean(output, dim=1)
        height = self.height_regressor(output_averaged)
        age = self.age_regressor(output_averaged)
        gender = self.gender_classifier(output_averaged)
        return height, age, gender

# height only models

class UpstreamTransformerH(nn.Module):
    def __init__(self, num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', 'wav2vec2')
        for param in self.upstream.parameters():
            param.requires_grad = False

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

class UpstreamConformerH(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        for param in self.upstream.parameters():
            param.requires_grad = False

        self.conformer_encoder = ConformerEncoder(encoder_dim=feature_dim, num_attention_heads=8, num_layers=num_layers)
        self.height_regressor = nn.Linear(feature_dim, 1)

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        output = self.conformer_encoder(x, x.shape[0])[0]
        output_averaged = torch.mean(output, dim=1)
        height = self.height_regressor(output_averaged)
        return height
