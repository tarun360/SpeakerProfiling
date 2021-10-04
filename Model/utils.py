import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
from torch.nn.functional import cross_entropy, interpolate, mse_loss

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))

class UncertaintyLoss(Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        self.loss = None
        self.loss_height = None
        self.loss_age = None
        self.loss_gender = None
        self.log_vars_height = Parameter(torch.tensor([0], requires_grad=True, dtype=torch.float32).cuda())
        self.log_vars_age = Parameter(torch.tensor([0], requires_grad=True, dtype=torch.float32).cuda())
        self.log_vars_gender = Parameter(torch.tensor([0], requires_grad=True, dtype=torch.float32).cuda())

    def forward(self, input, target):

        pred_arr = torch.split(input, 1)
        height_pred, age_pred, gender_pred = pred_arr

        target_arr = torch.split(target, 1)
        height_target, age_target, gender_target = target_arr

        self.loss_gender = cross_entropy(input=gender_pred, target=gender_target)
        self.loss_gender_var = cross_entropy(input=gender_pred*torch.exp(-self.log_vars_gender), target=gender_target)
        
        self.loss_height = torch.sum((height_target - height_pred) ** 2)
        self.loss_height_var = torch.exp(-self.log_vars_height) * self.loss_height + self.log_vars_height

        self.loss_age = torch.sum((age_target - age_pred) ** 2)
        self.loss_age_var = torch.exp(-self.log_vars_age) * self.loss_age + self.log_vars_age

        self.loss = self.loss_gender + self.loss_height + self.loss_age
        
        self.loss_var = self.loss_gender_var + self.loss_height_var + self.loss_age_var

        return self.loss_var

