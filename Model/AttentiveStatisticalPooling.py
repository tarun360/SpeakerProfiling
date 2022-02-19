# Attention Statistics Pooling
# Code taken from Hexin
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttensiveStatisticsPooling(nn.Module):
    def __init__(self, inputdim, outputdim, attn_dropout=0.0):
        super(AttensiveStatisticsPooling, self).__init__()
        self.inputdim = inputdim
        self.outputdim = outputdim
        self.attn_dropout = attn_dropout
        self.linear_projection = nn.Linear(inputdim, outputdim)
        self.v = torch.nn.Parameter(torch.randn(outputdim))

    def weighted_sd(self, inputs, attention_weights, mean):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs, el_mat_prod)
        variance = torch.sum(hadmard_prod, 1) - torch.mul(mean, mean)
        return variance

    def stat_attn_pool(self, inputs, attention_weights):
        el_mat_prod = torch.mul(inputs, attention_weights.unsqueeze(2).expand(-1, -1, inputs.shape[-1]))
        mean = torch.mean(el_mat_prod, dim=1)
        variance = self.weighted_sd(inputs, attention_weights, mean)
        stat_pooling = torch.cat((mean, variance), 1)
        return stat_pooling

    def forward(self,inputs):
        inputs = inputs.transpose(1,2)
        # print("input shape: {}".format(inputs.shape))
        lin_out = self.linear_projection(inputs)
        # print('lin_out shape:',lin_out.shape)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        # print("v's shape after expand:",v_view.shape)
        attention_weights = F.relu(lin_out.bmm(v_view).squeeze(2))
        # print("attention weight shape:",attention_weights.shape)
        attention_weights = F.softmax(attention_weights, dim=1)
        statistics_pooling_out = self.stat_attn_pool(inputs, attention_weights)
        # print(statistics_pooling_out.shape)
        return statistics_pooling_out

    def get_output_dim(self):
        return self.inputdim*2