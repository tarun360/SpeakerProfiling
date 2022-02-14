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

class UpstreamTransformerNew1H(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        for param in self.upstream.model.feature_extractor.conv_layers[:4].parameters():
            param.requires_grad = False
                
        self.height_regressor = nn.Linear(feature_dim, 1)
     
    def forward(self, x, x_len, lpcc):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        pooling = torch.mean(x, dim=1)
        height = self.height_regressor(pooling)
        return height
    
class CNNLPCC(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        # https://github.com/iPRoBe-lab/1D-Triplet-CNN/blob/master/models/OneD_Triplet_CNN.py
        self.conv_features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3,1), stride=1, padding='same' , dilation = (2,1)),
            nn.SELU(),

            nn.Conv2d(16, 32, kernel_size=(3,1), stride=1, padding='same', dilation = (2,1)),
            nn.SELU(),

            nn.Conv2d(32, 64, kernel_size=(7,1), padding='same', dilation = (2,1)),
            nn.SELU(),

            nn.Conv2d(64, 128, kernel_size=(9,1), padding='same', dilation = (3,1)),
            nn.SELU()
        )

        self.conv_regularization = nn.Sequential(
            nn.AlphaDropout(p=0.25)
        )
                
        self.height_regressor = nn.Linear(128, 1)
     
    def forward(self, x, x_len, lpcc):
        lpcc = lpcc.float()
        o = self.conv_features(lpcc)
        o = self.conv_regularization(o)
        o = torch.mean(o.view(o.size(0), o.size(1), -1), dim=2)
        height = self.height_regressor(o)
        return height
    
class CNNLPCC2(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        # https://github.com/iPRoBe-lab/1D-Triplet-CNN/blob/master/models/OneD_Triplet_CNN.py
        self.conv_features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=(3,1), stride=1, padding='same' , dilation = (2,1)),
            nn.SELU(),

            nn.Conv2d(16, 32, kernel_size=(3,1), stride=1, padding='same', dilation = (2,1)),
            nn.SELU(),

            nn.Conv2d(32, 64, kernel_size=(7,1), padding='same', dilation = (2,1)),
            nn.SELU(),
            
            nn.Conv2d(64, 128, kernel_size=(7,1), padding='same', dilation = (2,1)),
            nn.SELU(),

            nn.Conv2d(128, 256, kernel_size=(9,1), padding='same', dilation = (3,1)),
            nn.SELU()
        )

        self.conv_regularization = nn.Sequential(
            nn.AlphaDropout(p=0.25)
        )
                
        self.height_regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.Linear(64, 1),
        )
     
    def forward(self, x, x_len, lpcc):
        lpcc = lpcc.float()
        o = self.conv_features(lpcc)
        o = self.conv_regularization(o)
        o = torch.mean(o.view(o.size(0), o.size(1), -1), dim=2)
        height = self.height_regressor(o)
        return height
    
class UpstreamTransformerNew2H(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
                
        self.height_regressor = nn.Linear(2*feature_dim, 1)
     
    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        maxX,_ = torch.max(x, dim=1) 
        pooling = torch.cat((torch.mean(x, dim=1), maxX), dim=1)
        height = self.height_regressor(pooling)
        return height
    
class UpstreamTransformerNew3H(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
         
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.7)
        self.height_regressor = nn.Linear(512, 1)
     
    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        pooling = torch.mean(x, dim=1)
        out1 = self.fc1(self.dropout(pooling))
        out2 = self.fc2(self.dropout(out1))
        height = self.height_regressor(out2)
        return height   
    
class UpstreamTransformerNew4H(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
         
        self.fc1 = nn.Linear(2*feature_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.7)
        self.height_regressor = nn.Linear(512, 1)
         
    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        maxX,_ = torch.max(x, dim=1) 
        pooling = torch.cat((torch.mean(x, dim=1), maxX), dim=1)
        out1 = self.fc1(self.dropout(pooling))
        out2 = self.fc2(self.dropout(out1))
        height = self.height_regressor(out2)
        return height   
    
class UpstreamTransformerNewBilinear1H(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
         
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        self.upstream.model.feature_extractor.register_forward_hook(get_activation('feature_extractor'))

        self.bilinear_pooling =  CompactBilinearPooling(768, 512, 4096)
        
        self.height_regressor = nn.Linear(4096, 1)
     
    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        feature_extractor_output = torch.transpose(self.activation['feature_extractor'], 1, 2)
        output = self.bilinear_pooling(x,feature_extractor_output)
        output_averaged = torch.mean(output, dim=(1))
        height = self.height_regressor(output_averaged)
        return height
    
class UpstreamTransformerNewBilinear2H(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
         
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        self.upstream.model.feature_extractor.register_forward_hook(get_activation('feature_extractor'))

        self.bilinear_pooling =  CompactBilinearPooling(768, 512, 4096)
        
        self.dropout = nn.Dropout(0.7)
        
        self.fc = nn.Linear(feature_dim, 512)
        
        self.height_regressor = nn.Linear(4096, 1)
     
    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        feature_extractor_output = torch.transpose(self.activation['feature_extractor'], 1, 2)
        output = self.bilinear_pooling(x,feature_extractor_output)
        output_averaged = torch.mean(output, dim=(1))
        out = self.fc(self.dropout(output_averaged))
        height = self.height_regressor(out)
        return height
    
    
class UpstreamTransformerNewBilinear4H(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
         
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        self.upstream.model.feature_extractor.register_forward_hook(get_activation('feature_extractor'))

        self.bilinear_pooling =  CompactBilinearPooling(768, 512, 1024)
        
        self.dropout = nn.Dropout(0.7)
        
        self.fc1 = nn.Linear(2*1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.7)
        self.height_regressor = nn.Linear(512, 1)
        
    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        feature_extractor_output = torch.transpose(self.activation['feature_extractor'], 1, 2)
        output = self.bilinear_pooling(x,feature_extractor_output)
        maxX,_ = torch.max(output, dim=1) 
        pooling = torch.cat((torch.mean(output, dim=1), maxX), dim=1)
        out1 = self.fc1(self.dropout(pooling))
        out2 = self.fc2(self.dropout(out1))
        height = self.height_regressor(out2)
        return height
    