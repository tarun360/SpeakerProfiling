import torch
import torch.nn as nn
from conformer.encoder import ConformerEncoder
from IPython import embed
from area_attention import AreaAttention, MultiHeadAreaAttention

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
    
class UpstreamTransformerFC(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        self.fc1 = nn.Linear(feature_dim, 1024)
        
        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        height = self.height_regressor(x)
        age = self.age_regressor(x)
        gender = self.gender_classifier(x)
        return height, age, gender
    
class UpstreamTransformer3SeparateFC(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.fc2 = nn.Linear(feature_dim, 1024)
        self.fc3 = nn.Linear(feature_dim, 1024)
        
        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = torch.mean(x, dim=1)
        height = self.height_regressor(self.fc1(x))
        age = self.age_regressor(self.fc2(x))
        gender = self.gender_classifier(self.fc3(x))
        return height, age, gender

class UpstreamTransformerMoE(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        self.fcM = nn.Linear(feature_dim, 1024)
        self.fcF = nn.Linear(feature_dim, 1024)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = torch.mean(x, dim=1)
        xM = self.fcM(x)
        xF = self.fcF(x)
        xMF = torch.cat((xM,xF), dim=1)
        gender = self.gender_classifier(xMF)
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender
    
class UpstreamTransformerMoE2(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        self.fcM = nn.Linear(feature_dim, 1024)
        self.fcF = nn.Linear(feature_dim, 1024)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = torch.mean(x, dim=1)
        xM = self.fcM(x)
        xF = self.fcF(x)
        gender = self.gender_classifier(x)
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender

class UpstreamTransformerMoE3(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fcM = nn.Linear(feature_dim, 1024)
        self.fcF = nn.Linear(feature_dim, 1024)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        gender = self.gender_classifier(x)
        xM = self.fcM(x)
        xF = self.fcF(x)
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender
    
class UpstreamTransformerMoE4(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fcM = nn.Linear(2*feature_dim, 1024)
        self.fcF = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = self.transformer_encoder(x)
        x = torch.cat((torch.mean(x, dim=1), torch.std(x, dim=1)), dim=1)
        gender = self.gender_classifier(x)
        xM = self.dropout(self.fcM(x))
        xF = self.dropout(self.fcF(x))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender
    
class UpstreamTransformerMoE5(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_M = torch.nn.TransformerEncoder(encoder_layer_M, num_layers=num_layers)
        
        encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_F = torch.nn.TransformerEncoder(encoder_layer_F, num_layers=num_layers)
        
        self.fcM = nn.Linear(2*feature_dim, 1024)
        self.fcF = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        xM = self.transformer_encoder_M(x)
        xF = self.transformer_encoder_F(x)
        xM = self.dropout(torch.cat((torch.mean(xM, dim=1), torch.std(xM, dim=1)), dim=1))
        xF = self.dropout(torch.cat((torch.mean(xF, dim=1), torch.std(xF, dim=1)), dim=1))
        xM = self.dropout(self.fcM(xM))
        xF = self.dropout(self.fcF(xF))
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender
    
class UpstreamTransformerAAFC(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
        self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False
        
        for param in self.upstream.model.encoder.layers.parameters():
            param.requires_grad = True
        
        area_attention = AreaAttention(
            key_query_size=feature_dim,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=3,
            max_area_width=3,
            memory_height=1,
            memory_width=1,
            dropout_rate=0.2,
            top_k_areas=0
        )
        self.multi_head_area_attention = MultiHeadAreaAttention(
            area_attention=area_attention,
            num_heads=8,
            key_query_size=feature_dim,
            key_query_size_hidden=feature_dim,
            value_size=feature_dim,
            value_size_hidden=feature_dim
        )
        
        self.fc1 = nn.Linear(feature_dim, 1024)
        
        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        x = torch.mean(x, dim=1)
        x = x.unsqueeze(dim=1)
        x = self.multi_head_area_attention(x,x,x)
        x = self.fc1(x)
        x = x.squeeze(dim=1)
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

    def forward(self, x):
        x = [wav for wav in x.squeeze(1)]
        x = self.upstream(x)['last_hidden_state']
        output = self.transformer_encoder(x)
        output_averaged = torch.mean(output, dim=1)
        height = self.height_regressor(output_averaged)
        return height
    