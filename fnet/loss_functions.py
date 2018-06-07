import torch


import pdb

class MSELoss_aleotoric(torch.nn.Module):
    def __init__(self):
        super(MSELoss_aleotoric, self).__init__()
    
    def forward(self, input, target):
        
        input_mu = input[:,0]
        input_log_var = input[:,1]
        
        target = torch.squeeze(target)
        
        diff = 0.5 * torch.exp(-(input_log_var*2)) * (target-input_mu)**2 + 0.5*(input_log_var*2)
        
        err = torch.mean(diff)

        return err
    
class MSELoss_aleotoric_laplace(torch.nn.Module):
    def __init__(self):
        super(MSELoss_aleotoric_laplace, self).__init__()
    
    def forward(self, input, target):
        
        input_mu = input[:,0]
        input_log_var = input[:,1]
        
        target = torch.squeeze(target)
        
        diff = torch.exp(-(input_log_var)) * torch.abs(target-input_mu) + (input_log_var)
        
        err = torch.mean(diff)

        return err    