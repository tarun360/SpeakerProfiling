import torch
import torch.nn as nn
from conformer.encoder import ConformerEncoder
from IPython import embed
from area_attention import AreaAttention, MultiHeadAreaAttention
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

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        output = self.transformer_encoder(x)
        output_averaged = torch.mean(output, dim=1)
        height = self.height_regressor(output_averaged)
        age = self.age_regressor(output_averaged)
        gender = self.gender_classifier(output_averaged)
        return height, age, gender

class UpstreamTransformer2(nn.Module):
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
        
        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        x = self.transformer_encoder(x)
        x = self.dropout(torch.cat((torch.mean(x, dim=1), torch.std(x, dim=1)), dim=1))
        x = self.dropout(self.fc(x))
        height = self.height_regressor(x)
        age = self.age_regressor(x)
        gender = self.gender_classifier(x)
        return height, age, gender
    
class UpstreamTransformerMoE5(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # phase 1
#         for param in self.upstream.parameters():
#             param.requires_grad = False
       
        # phase 2
        for param in self.upstream.parameters():
            param.requires_grad = True
        for param in self.upstream.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False

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

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
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

class UpstreamTransformerMoE5Bilinear(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
#         self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = True
       
        for param in self.upstream.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False
                
#         for param in self.upstream.model.encoder.layers.parameters():
#             param.requires_grad = True

#         if unfreeze_last_conv_layers:
#             for param in self.upstream.model.feature_extractor.conv_layers[5:].parameters():
#                 param.requires_grad = True
        
        encoder_layer_M = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_M = torch.nn.TransformerEncoder(encoder_layer_M, num_layers=num_layers)
        
        encoder_layer_F = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8, batch_first=True)
        self.transformer_encoder_F = torch.nn.TransformerEncoder(encoder_layer_F, num_layers=num_layers)
        
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        self.upstream.model.feature_extractor.register_forward_hook(get_activation('feature_extractor'))

        self.bilinear_pooling =  CompactBilinearPooling(768, 512, 4096)
        
        self.fcM = nn.Linear(4096, 1024)
        self.fcF = nn.Linear(4096, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        xM = self.transformer_encoder_M(x)
        xF = self.transformer_encoder_F(x)
        feature_extractor_output = torch.transpose(self.activation['feature_extractor'], 1, 2)
        xM = self.dropout(self.bilinear_pooling(xM, feature_extractor_output))
        xF = self.dropout(self.bilinear_pooling(xF, feature_extractor_output))
        xM = torch.mean(xM, dim=1)
        xF = torch.mean(xF, dim=1)
        xM = self.dropout(self.fcM(xM))
        xF = self.dropout(self.fcF(xF))
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender

class UpstreamTransformerMoE7(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[0:4].parameters():
                param.requires_grad = False
        
        self.fcMH = nn.Linear(2*feature_dim, 1024)
        self.fcFH = nn.Linear(2*feature_dim, 1024)
        
        self.fcMA = nn.Linear(2*feature_dim, 1024)
        self.fcFA = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        pooling = torch.cat((torch.mean(x, dim=1), torch.std(x, dim=1)), dim=1)
        x = self.dropout(pooling)
        xMH = self.dropout(self.fcMH(x))
        xFH = self.dropout(self.fcFH(x))
        xMA = self.dropout(self.fcMA(x))
        xFA = self.dropout(self.fcFA(x))
        gender = self.gender_classifier(pooling)
        xH = (1-gender)*xMH + gender*xFH
        xA = (1-gender)*xMA + gender*xFA
        height = self.height_regressor(xH)
        age = self.age_regressor(xA)
        return height, age, gender

class UpstreamTransformerMoE8(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers[0:4].parameters():
                param.requires_grad = False
        
        self.fcMH = nn.Linear(2*feature_dim, 1024)
        self.fcFH = nn.Linear(2*feature_dim, 1024)
        
        self.fcMA = nn.Linear(2*feature_dim, 1024)
        self.fcFA = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        maxX,_ = torch.max(x, dim=1) 
        pooling = torch.cat((torch.mean(x, dim=1), maxX), dim=1)
        x = self.dropout(pooling)
        xMH = self.dropout(self.fcMH(x))
        xFH = self.dropout(self.fcFH(x))
        xMA = self.dropout(self.fcMA(x))
        xFA = self.dropout(self.fcFA(x))
        gender = self.gender_classifier(pooling)
        xH = (1-gender)*xMH + gender*xFH
        xA = (1-gender)*xMA + gender*xFA
        height = self.height_regressor(xH)
        age = self.age_regressor(xA)
        return height, age, gender

class UpstreamTransformerMoE9(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstream.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers.parameters():
                param.requires_grad = False
                
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        self.upstream.model.feature_extractor.register_forward_hook(get_activation('feature_extractor'))
        
        self.fcMH = nn.Linear(2*feature_dim+512, 1024)
        self.fcFH = nn.Linear(2*feature_dim+512, 1024)
        
        self.fcMA = nn.Linear(2*feature_dim+512, 1024)
        self.fcFA = nn.Linear(2*feature_dim+512, 1024)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*feature_dim+512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        maxX,_ = torch.max(x, dim=1) 
        feature_extractor_output = self.activation['feature_extractor'].mean(dim=2)
        pooling = torch.cat((torch.mean(x, dim=1), maxX, feature_extractor_output), dim=1)
        x = self.dropout(pooling)
        xMH = self.dropout(self.fcMH(x))
        xFH = self.dropout(self.fcFH(x))
        xMA = self.dropout(self.fcMA(x))
        xFA = self.dropout(self.fcFA(x))
        gender = self.gender_classifier(pooling)
        xH = (1-gender)*xMH + gender*xFH
        xA = (1-gender)*xMA + gender*xFA
        height = self.height_regressor(xH)
        age = self.age_regressor(xA)
        return height, age, gender

class UpstreamTransformerMoE10(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamM = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamF = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamM.model.parameters():
            param.requires_grad = True
        for param in self.upstreamF.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstreamM.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamF.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
                
        self.activationM = {}
        def get_activationM(name):
            def hook(model, input, output):
                self.activationM[name] = output
            return hook
        
        self.activationF = {}
        def get_activationF(name):
            def hook(model, input, output):
                self.activationF[name] = output
            return hook
        self.upstreamM.model.feature_extractor.register_forward_hook(get_activationM('feature_extractor'))
        self.upstreamF.model.feature_extractor.register_forward_hook(get_activationF('feature_extractor'))
        
        self.fcMH = nn.Linear(2*feature_dim+512, 1024)
        self.fcFH = nn.Linear(2*feature_dim+512, 1024)
        
        self.fcMA = nn.Linear(2*feature_dim+512, 1024)
        self.fcFA = nn.Linear(2*feature_dim+512, 1024)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(4*feature_dim+2*512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xM = self.upstreamM(x)['last_hidden_state']
        maxXM,_ = torch.max(xM, dim=1) 
        feature_extractor_output_M = self.activationM['feature_extractor'].mean(dim=2)
        poolingM = torch.cat((torch.mean(xM, dim=1), maxXM, feature_extractor_output_M), dim=1)
        
        xF = self.upstreamF(x)['last_hidden_state']
        maxXF,_ = torch.max(xF, dim=1) 
        feature_extractor_output_F = self.activationF['feature_extractor'].mean(dim=2)
        poolingF = torch.cat((torch.mean(xF, dim=1), maxXF, feature_extractor_output_F), dim=1)
        
        xM = self.dropout(poolingM)
        xF = self.dropout(poolingF)
        
        xMH = self.dropout(self.fcMH(xM))
        xFH = self.dropout(self.fcFH(xF))
        xMA = self.dropout(self.fcMA(xM))
        xFA = self.dropout(self.fcFA(xF))
        gender = self.gender_classifier(torch.cat((poolingM,poolingF), dim=1))
        xH = (1-gender)*xMH + gender*xFH
        xA = (1-gender)*xMA + gender*xFA
        height = self.height_regressor(xH)
        age = self.age_regressor(xA)
        return height, age, gender    
    
class UpstreamTransformerMoE11(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamM = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamF = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamM.model.parameters():
            param.requires_grad = True
        for param in self.upstreamF.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstreamM.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamF.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
                
        self.activationM = {}
        def get_activationM(name):
            def hook(model, input, output):
                self.activationM[name] = output
            return hook
        
        self.activationF = {}
        def get_activationF(name):
            def hook(model, input, output):
                self.activationF[name] = output
            return hook
        self.upstreamM.model.feature_extractor.register_forward_hook(get_activationM('feature_extractor'))
        self.upstreamF.model.feature_extractor.register_forward_hook(get_activationF('feature_extractor'))
        
        self.fcMH = nn.Linear(2*feature_dim+512, 128)
        self.fcFH = nn.Linear(2*feature_dim+512, 128)
        
        self.fcMA = nn.Linear(2*feature_dim+512, 128)
        self.fcFA = nn.Linear(2*feature_dim+512, 128)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(128, 1)
        self.age_regressor = nn.Linear(128, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(4*128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xM = self.upstreamM(x)['last_hidden_state']
        maxXM,_ = torch.max(xM, dim=1) 
        feature_extractor_output_M = self.activationM['feature_extractor'].mean(dim=2)
        poolingM = torch.cat((torch.mean(xM, dim=1), maxXM, feature_extractor_output_M), dim=1)
        
        xF = self.upstreamF(x)['last_hidden_state']
        maxXF,_ = torch.max(xF, dim=1) 
        feature_extractor_output_F = self.activationF['feature_extractor'].mean(dim=2)
        poolingF = torch.cat((torch.mean(xF, dim=1), maxXF, feature_extractor_output_F), dim=1)
        
        xM = self.dropout(poolingM)
        xF = self.dropout(poolingF)
        
        xMH = self.dropout(self.fcMH(xM))
        xFH = self.dropout(self.fcFH(xF))
        xMA = self.dropout(self.fcMA(xM))
        xFA = self.dropout(self.fcFA(xF))
        gender = self.gender_classifier(torch.cat((xMH, xFH, xMA, xFA), dim=1))
        xH = (1-gender)*xMH + gender*xFH
        xA = (1-gender)*xMA + gender*xFA
        height = self.height_regressor(xH)
        age = self.age_regressor(xA)
        return height, age, gender

class UpstreamTransformerMoE12(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamM = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamF = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamM.model.parameters():
            param.requires_grad = True
        for param in self.upstreamF.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstreamM.model.feature_extractor.conv_layers.parameters():
                param.requires_grad = False
            for param in self.upstreamF.model.feature_extractor.conv_layers.parameters():
                param.requires_grad = False
                
        self.activationM = {}
        def get_activationM(name):
            def hook(model, input, output):
                self.activationM[name] = output
            return hook
        
        self.activationF = {}
        def get_activationF(name):
            def hook(model, input, output):
                self.activationF[name] = output
            return hook
        self.upstreamM.model.feature_extractor.register_forward_hook(get_activationM('feature_extractor'))
        self.upstreamF.model.feature_extractor.register_forward_hook(get_activationF('feature_extractor'))
        
        self.fcMH = nn.Linear(2*feature_dim+512, 512)
        self.fcFH = nn.Linear(2*feature_dim+512, 512)
        
        self.fcMA = nn.Linear(2*feature_dim+512, 512)
        self.fcFA = nn.Linear(2*feature_dim+512, 512)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(512, 1)
        self.age_regressor = nn.Linear(512, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(4*512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xM = self.upstreamM(x)['last_hidden_state']
        maxXM,_ = torch.max(xM, dim=1) 
        feature_extractor_output_M = self.activationM['feature_extractor'].mean(dim=2)
        poolingM = torch.cat((torch.mean(xM, dim=1), maxXM, feature_extractor_output_M), dim=1)
        
        xF = self.upstreamF(x)['last_hidden_state']
        maxXF,_ = torch.max(xF, dim=1) 
        feature_extractor_output_F = self.activationF['feature_extractor'].mean(dim=2)
        poolingF = torch.cat((torch.mean(xF, dim=1), maxXF, feature_extractor_output_F), dim=1)
        
        xM = self.dropout(poolingM)
        xF = self.dropout(poolingF)
        
        xMH = self.dropout(self.fcMH(xM))
        xFH = self.dropout(self.fcFH(xF))
        xMA = self.dropout(self.fcMA(xM))
        xFA = self.dropout(self.fcFA(xF))
        gender = self.gender_classifier(torch.cat((xMH, xFH, xMA, xFA), dim=1))
        xH = (1-gender)*xMH + gender*xFH
        xA = (1-gender)*xMA + gender*xFA
        height = self.height_regressor(xH)
        age = self.age_regressor(xA)
        return height, age, gender

class UpstreamTransformerMoE13(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamM = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamF = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamM.model.parameters():
            param.requires_grad = True
        for param in self.upstreamF.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstreamM.model.feature_extractor.conv_layers.parameters():
                param.requires_grad = False
            for param in self.upstreamF.model.feature_extractor.conv_layers.parameters():
                param.requires_grad = False
                
        self.activationM = {}
        def get_activationM(name):
            def hook(model, input, output):
                self.activationM[name] = output
            return hook
        
        self.activationF = {}
        def get_activationF(name):
            def hook(model, input, output):
                self.activationF[name] = output
            return hook
        self.upstreamM.model.feature_extractor.register_forward_hook(get_activationM('feature_extractor'))
        self.upstreamF.model.feature_extractor.register_forward_hook(get_activationF('feature_extractor'))
        
        self.fcG = nn.Linear(4*feature_dim+2*512, 1024)
        
        self.fcMH1 = nn.Linear(2*feature_dim+512, 1024)
        self.fcFH1 = nn.Linear(2*feature_dim+512, 1024)
        self.fcMA1 = nn.Linear(2*feature_dim+512, 1024)
        self.fcFA1 = nn.Linear(2*feature_dim+512, 1024)
        
        self.fcMH2 = nn.Linear(1024, 512)
        self.fcFH2 = nn.Linear(1024, 512)
        self.fcMA2 = nn.Linear(1024, 512)
        self.fcFA2 = nn.Linear(1024, 512)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(512, 1)
        self.age_regressor = nn.Linear(512, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xM = self.upstreamM(x)['last_hidden_state']
        maxXM,_ = torch.max(xM, dim=1) 
        feature_extractor_output_M = self.activationM['feature_extractor'].mean(dim=2)
        poolingM = torch.cat((torch.mean(xM, dim=1), maxXM, feature_extractor_output_M), dim=1)
        
        xF = self.upstreamF(x)['last_hidden_state']
        maxXF,_ = torch.max(xF, dim=1) 
        feature_extractor_output_F = self.activationF['feature_extractor'].mean(dim=2)
        poolingF = torch.cat((torch.mean(xF, dim=1), maxXF, feature_extractor_output_F), dim=1)
        
        xM = self.dropout(poolingM)
        xF = self.dropout(poolingF)
        
        xG = self.fcG(torch.cat((xM, xF),dim=1))
        gender = self.gender_classifier(xG)
        
        xMH1 = self.dropout(self.fcMH1(xM))
        xFH1 = self.dropout(self.fcFH1(xF))
        xMA1 = self.dropout(self.fcMA1(xM))
        xFA1 = self.dropout(self.fcFA1(xF))
        
        xMH2 = self.dropout(self.fcMH2(xMH1))
        xFH2 = self.dropout(self.fcFH2(xFH1))
        xMA2 = self.dropout(self.fcMA2(xMA1))
        xFA2 = self.dropout(self.fcFA2(xFA1))
        
        xH = (1-gender)*xMH2 + gender*xFH2
        xA = (1-gender)*xMA2 + gender*xFA2
        height = self.height_regressor(xH)
        age = self.age_regressor(xA)
        
        return height, age, gender

class UpstreamTransformerMoE14(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamH = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamA = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamG = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamH.model.parameters():
            param.requires_grad = True
        for param in self.upstreamA.model.parameters():
            param.requires_grad = True
        for param in self.upstreamG.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstreamH.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamA.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamG.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
                
        self.fcH1 = nn.Linear(2*feature_dim, 1024)
        self.fcH2 = nn.Linear(1024, 512)
        
        self.fcA1 = nn.Linear(2*feature_dim, 1024)
        self.fcA2 = nn.Linear(1024, 512)
        
        self.fcG1 = nn.Linear(2*feature_dim, 1024)
        self.fcG2 = nn.Linear(1024, 512)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(512, 1)
        self.age_regressor = nn.Linear(512, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xH = self.upstreamH(x)['last_hidden_state']
        maxXH,_ = torch.max(xH, dim=1) 
        poolingH = torch.cat((torch.mean(xH, dim=1), maxXH), dim=1)
        
        xA = self.upstreamA(x)['last_hidden_state']
        maxXA,_ = torch.max(xA, dim=1) 
        poolingA = torch.cat((torch.mean(xA, dim=1), maxXA), dim=1)
        
        xG = self.upstreamG(x)['last_hidden_state']
        maxXG,_ = torch.max(xG, dim=1) 
        poolingG = torch.cat((torch.mean(xG, dim=1), maxXG), dim=1)
        
        xH1 = self.fcH1(self.dropout(poolingH))
        xH2 = self.fcH2(self.dropout(xH1))
        
        xA1 = self.fcA1(self.dropout(poolingA))
        xA2 = self.fcA2(self.dropout(xA1))
        
        xG1 = self.fcG1(self.dropout(poolingG))
        xG2 = self.fcG2(self.dropout(xG1))
        
        gender = self.gender_classifier(xG2)
        height = self.height_regressor(xH2)
        age = self.age_regressor(xA2)
        
        return height, age, gender
    
class UpstreamTransformerMoE15(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamH = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamA = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamG = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamH.model.parameters():
            param.requires_grad = True
        for param in self.upstreamA.model.parameters():
            param.requires_grad = True
        for param in self.upstreamG.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstreamH.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamA.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamG.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
                
        self.fcMH1 = nn.Linear(2*feature_dim, 1024)
        self.fcMH2 = nn.Linear(1024, 512)
        self.fcFH1 = nn.Linear(2*feature_dim, 1024)
        self.fcFH2 = nn.Linear(1024, 512)
        
        self.fcMA1 = nn.Linear(2*feature_dim, 1024)
        self.fcMA2 = nn.Linear(1024, 512)
        self.fcFA1 = nn.Linear(2*feature_dim, 1024)
        self.fcFA2 = nn.Linear(1024, 512)
        
        self.fcG1 = nn.Linear(2*feature_dim, 1024)
        self.fcG2 = nn.Linear(1024, 512)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(512, 1)
        self.age_regressor = nn.Linear(512, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xH = self.upstreamH(x)['last_hidden_state']
        maxXH,_ = torch.max(xH, dim=1) 
        poolingH = torch.cat((torch.mean(xH, dim=1), maxXH), dim=1)
        
        xA = self.upstreamA(x)['last_hidden_state']
        maxXA,_ = torch.max(xA, dim=1) 
        poolingA = torch.cat((torch.mean(xA, dim=1), maxXA), dim=1)
        
        xG = self.upstreamG(x)['last_hidden_state']
        maxXG,_ = torch.max(xG, dim=1) 
        poolingG = torch.cat((torch.mean(xG, dim=1), maxXG), dim=1)
        
        xMH1 = self.fcMH1(self.dropout(poolingH))
        xMH2 = self.fcMH2(self.dropout(xMH1))
        xFH1 = self.fcFH1(self.dropout(poolingH))
        xFH2 = self.fcFH2(self.dropout(xFH1))
        
        xMA1 = self.fcMA1(self.dropout(poolingA))
        xMA2 = self.fcMA2(self.dropout(xMA1))
        xFA1 = self.fcFA1(self.dropout(poolingA))
        xFA2 = self.fcFA2(self.dropout(xFA1))
        
        xG1 = self.fcG1(self.dropout(poolingG))
        xG2 = self.fcG2(self.dropout(xG1))
        
        gender = self.gender_classifier(xG2)
        height = self.height_regressor((1-gender)*xMH2+gender*xFH2)
        age = self.age_regressor((1-gender)*xMA2+gender*xFA2)
        
        return height, age, gender
    
class UpstreamTransformerMoE16(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamH = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamA = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamG = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamH.model.parameters():
            param.requires_grad = True
        for param in self.upstreamA.model.parameters():
            param.requires_grad = True
        for param in self.upstreamG.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstreamH.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamA.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamG.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
                
        self.fcH1 = nn.Linear(2*feature_dim, 1024)
        self.fcH2 = nn.Linear(1024, 512)
        
        self.fcMA1 = nn.Linear(2*feature_dim, 1024)
        self.fcMA2 = nn.Linear(1024, 512)
        self.fcFA1 = nn.Linear(2*feature_dim, 1024)
        self.fcFA2 = nn.Linear(1024, 512)
        
        self.fcG1 = nn.Linear(2*feature_dim, 1024)
        self.fcG2 = nn.Linear(1024, 512)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(512, 1)
        self.age_regressor = nn.Linear(512, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xH = self.upstreamH(x)['last_hidden_state']
        maxXH,_ = torch.max(xH, dim=1) 
        poolingH = torch.cat((torch.mean(xH, dim=1), maxXH), dim=1)
        
        xA = self.upstreamA(x)['last_hidden_state']
        maxXA,_ = torch.max(xA, dim=1) 
        poolingA = torch.cat((torch.mean(xA, dim=1), maxXA), dim=1)
        
        xG = self.upstreamG(x)['last_hidden_state']
        maxXG,_ = torch.max(xG, dim=1) 
        poolingG = torch.cat((torch.mean(xG, dim=1), maxXG), dim=1)
        
        xH1 = self.fcH1(self.dropout(poolingH))
        xH2 = self.fcH2(self.dropout(xH1))
        
        xMA1 = self.fcMA1(self.dropout(poolingA))
        xMA2 = self.fcMA2(self.dropout(xMA1))
        xFA1 = self.fcFA1(self.dropout(poolingA))
        xFA2 = self.fcFA2(self.dropout(xFA1))
        
        xG1 = self.fcG1(self.dropout(poolingG))
        xG2 = self.fcG2(self.dropout(xG1))
        
        gender = self.gender_classifier(xG2)
        height = self.height_regressor(xH2)
        age = self.age_regressor((1-gender)*xMA2+gender*xFA2)
        
        return height, age, gender
    
    
class UpstreamTransformerMoE17(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamH = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamA = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamG = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamH.model.parameters():
            param.requires_grad = True
        for param in self.upstreamA.model.parameters():
            param.requires_grad = True
        for param in self.upstreamG.model.parameters():
            param.requires_grad = True
        
        if unfreeze_last_conv_layers:
            for param in self.upstreamH.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamA.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
            for param in self.upstreamG.model.feature_extractor.conv_layers[:4].parameters():
                param.requires_grad = False
                
        self.fcH1 = nn.Linear(2*feature_dim, 1024)
        self.fcH2 = nn.Linear(1024, 512)
        
        self.fcMA1 = nn.Linear(2*feature_dim, 1024)
        self.fcMA2 = nn.Linear(1024, 512)
        self.fcFA1 = nn.Linear(2*feature_dim, 1024)
        self.fcFA2 = nn.Linear(1024, 512)
        
        self.fcG1 = nn.Linear(2*feature_dim, 1024)
        self.fcG2 = nn.Linear(1024, 512)
        
        self.dropout = nn.Dropout(0.7)

        self.height_regressor = nn.Linear(512, 1)
        self.age_regressor = nn.Linear(512, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xH = self.upstreamH(x)['last_hidden_state']
        maxXH,_ = torch.max(xH, dim=1) 
        poolingH = torch.cat((torch.mean(xH, dim=1), maxXH), dim=1)
        
        xA = self.upstreamA(x)['last_hidden_state']
        maxXA,_ = torch.max(xA, dim=1) 
        poolingA = torch.cat((torch.mean(xA, dim=1), maxXA), dim=1)
        
        xG = self.upstreamG(x)['last_hidden_state']
        maxXG,_ = torch.max(xG, dim=1) 
        poolingG = torch.cat((torch.mean(xG, dim=1), maxXG), dim=1)
        
        xH1 = self.fcH1(self.dropout(poolingH))
        xH2 = self.fcH2(self.dropout(xH1))
        
        xMA1 = self.fcMA1(self.dropout(poolingA))
        xMA2 = self.fcMA2(self.dropout(xMA1))
        xFA1 = self.fcFA1(self.dropout(poolingA))
        xFA2 = self.fcFA2(self.dropout(xFA1))
        
        xG1 = self.fcG1(self.dropout(poolingG))
        xG2 = self.fcG2(self.dropout(xG1))
        
        gender = self.gender_classifier(xG2)
        height = self.height_regressor(self.dropout(xH2))
        age = self.age_regressor(self.dropout((1-gender)*xMA2+gender*xFA2))
        
        return height, age, gender
    
    
class UpstreamTransformerMoE6(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstream = torch.hub.load('s3prl/s3prl', upstream_model)
        
        # Selecting the 9th encoder layer (out of 12)
#         self.upstream.model.encoder.layers = self.upstream.model.encoder.layers[0:9]
        
        for param in self.upstream.parameters():
            param.requires_grad = False

        if unfreeze_last_conv_layers:
            for param in self.upstream.model.feature_extractor.conv_layers.parameters():
                param.requires_grad = True
        
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        self.upstream.model.feature_extractor.register_forward_hook(get_activation('feature_extractor'))
        
        self.fcM = nn.Linear(512, 1024)
        self.fcF = nn.Linear(512, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        x = self.upstream(x)['last_hidden_state']
        feature_extractor_output = self.activation['feature_extractor'].mean(dim=2)
        xM = self.dropout(feature_extractor_output)
        xF = self.dropout(feature_extractor_output)
        xM = self.dropout(self.fcM(xM))
        xF = self.dropout(self.fcF(xF))
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
        return height, age, gender  
    
class UpstreamTransformerMoE20(nn.Module):
    def __init__(self, upstream_model='wav2vec2',num_layers=6, feature_dim=768, unfreeze_last_conv_layers=False):
        super().__init__()
        self.upstreamM = torch.hub.load('s3prl/s3prl', upstream_model)
        self.upstreamF = torch.hub.load('s3prl/s3prl', upstream_model)
        
        for param in self.upstreamM.parameters():
            param.requires_grad = True
        for param in self.upstreamF.parameters():
            param.requires_grad = True
       
        for param in self.upstreamM.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False
        for param in self.upstreamF.model.feature_extractor.conv_layers[:5].parameters():
            param.requires_grad = False
        
        self.fcM = nn.Linear(2*feature_dim, 1024)
        self.fcF = nn.Linear(2*feature_dim, 1024)
        
        self.dropout = nn.Dropout(0.5)

        self.height_regressor = nn.Linear(1024, 1)
        self.age_regressor = nn.Linear(1024, 1)
        self.gender_classifier = nn.Sequential(
            nn.Linear(2*1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, x_len):
        x = [torch.narrow(wav,0,0,x_len[i]) for (i,wav) in enumerate(x.squeeze(1))]
        xM = self.upstreamM(x)['last_hidden_state']
        xF = self.upstreamF(x)['last_hidden_state']
        xM = self.dropout(torch.cat((torch.mean(xM, dim=1), torch.std(xM, dim=1)), dim=1))
        xF = self.dropout(torch.cat((torch.mean(xF, dim=1), torch.std(xF, dim=1)), dim=1))
        xM = self.dropout(self.fcM(xM))
        xF = self.dropout(self.fcF(xF))
        gender = self.gender_classifier(torch.cat((xM, xF), dim=1))
        output = (1-gender)*xM + gender*xF
        height = self.height_regressor(output)
        age = self.age_regressor(output)
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
    