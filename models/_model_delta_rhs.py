import torch
import torch.nn as nn


class DeltaRHRectifier(nn.Module):
    def __init__(self, last_activation, rh_idx:int=0):
        super(DeltaRHRectifier, self).__init__()
        self.activation = last_activation
        self.rh_idx = rh_idx

    def forward(self, x):
        rh = x[:,self.rh_idx:self.rh_idx+1,:,:]
        x_l = self.activation(x[:,self.rh_idx+1:,:,:])
        x_s = - self.activation(x[:,:self.rh_idx,:,:])
        out = torch.cat((x_s, rh, x_l), dim=1)
        return out

