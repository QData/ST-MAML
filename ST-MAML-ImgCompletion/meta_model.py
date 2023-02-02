import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
import torch.distributions as distributions
from collections import OrderedDict
import torch.nn.init as init



def linear_block(iput, aug, param, prefix):

    n_data, _ = iput.size()
    out = F.linear(iput, weight=param[prefix+'linear1.weight'], bias=param[prefix+'linear1.bias'])
    out = F.relu(out, inplace=True)
    out = torch.cat([out, aug], dim=-1)
   
    out = F.linear(out, weight=param[prefix+'linear2.weight'], bias=param[prefix+'linear2.bias'])
    out = F.relu(out, inplace=True)

    out = F.linear(out, weight=param[prefix+'linear3.weight'], bias=param[prefix+'linear3.bias'])
    out = F.relu(out, inplace=True)

    out = F.linear(out, weight=param[prefix+'linear4.weight'], bias=param[prefix+'linear4.bias'])
    out = F.relu(out, inplace=True)

    out = F.linear(out, weight=param[prefix+'linear5.weight'], bias=param[prefix+'linear5.bias'])
    # out = torch.matmul(out, param[prefix+'linear3.weight'].permute(0, 2, 1)) + param[prefix+'linear3.bias']
    return out


class LR(nn.Module):
    def __init__(self, n_input, n_h, n_output, n_out_aug):
        super(LR, self).__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module('linear1', nn.Linear(n_input, n_h[0]))
        self.mlp.add_module('relu1', nn.ReLU(True))
        self.mlp.add_module('linear2', nn.Linear(n_h[0]+n_out_aug, n_h[1]))
        self.mlp.add_module('relu2', nn.ReLU(True))
        self.mlp.add_module('linear3', nn.Linear(n_h[1], n_h[2]))
        self.mlp.add_module('relu3', nn.ReLU(True))
        self.mlp.add_module('linear4', nn.Linear(n_h[2], n_h[3]))
        self.mlp.add_module('relu4', nn.ReLU(True))
        self.mlp.add_module('linear5', nn.Linear(n_h[3], n_output))
        

        for layer in self.mlp.modules():
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0) 



        
    def forward(self, x, aug, param=None):
        
        prefix = 'mlp.'
        y = linear_block(x, aug, param, prefix)
        return y
        
    def cloned_state_dict(self):

        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict
    
    
    
class TaskEnc(nn.Module):
    def __init__(self, n_xyaug, n_hid, n_out):
        super(TaskEnc, self).__init__()
        self.encoder = nn.Sequential()
        self.encoder.add_module('xy2hid', nn.Linear(n_xyaug, n_hid))
        self.encoder.add_module('relu1', nn.ReLU(inplace=True))
        self.encoder.add_module('hid2out', nn.Linear(n_hid, n_out))
      
        for layer in self.encoder.modules():
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)


    def forward(self, aug):
        out = self.encoder(aug)
        return out







class AugInfo(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(AugInfo, self).__init__()
        self.aug_enc = nn.Sequential()
        self.aug_enc.add_module('xy2hid', nn.Linear(n_in, n_hid[0]))
        self.aug_enc.add_module('relu1', nn.ReLU(inplace=True))
        self.aug_enc.add_module('hid2hid', nn.Linear(n_hid[0], n_hid[1]))
        self.aug_enc.add_module('relu2', nn.ReLU(inplace=True))
        self.aug_enc.add_module('hid2out', nn.Linear(n_hid[1], n_out))

        for layer in self.aug_enc.modules():
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in')
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
        
     
        
    def forward(self, x, y):
        n_spt, _ = x.size()
        xy_concat = torch.cat([x, y] ,dim=-1)
        out = self.aug_enc(xy_concat)
        out = out.mean(dim=0, keepdim=True)
        return out

        
        
        
        
        
        
        

class EncModel(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_xyaug, n_hid_emb, n_out_emb, n_xy, n_hid_aug, n_out_aug, aug_feature=True):
        super(EncModel, self).__init__()

        assert (n_in+n_out==n_xy), 'dimension mismatching.'
        
        self.aug_feature = aug_feature

        self.augenc = AugInfo(n_xy, n_hid_aug, n_out_aug)
           
        self.learner = LR(n_in, n_hid, n_out, n_out_aug)
        self.encoder = TaskEnc(n_xyaug, n_hid_emb, n_out_emb)
            
        
    def task_encoder(self, x_spt, y_spt):
        self.aug = self.augenc(x_spt, y_spt)

    def param_encoder(self):
        self.task_emb = self.encoder(self.aug)

        
            

        
    def encode_param(self, param=None):
        if param == None:
            adapted_state_dict = self.learner.cloned_state_dict()
            adapted_params = OrderedDict()
            for (key, val) in self.learner.named_parameters():

                if key == 'mlp.linear5.weight':
                    code = self.task_emb
                    adapted_params[key] = torch.sigmoid(code)*val                    
                    adapted_state_dict[key] = adapted_params[key]
                else:
                    adapted_params[key] = val
                    adapted_state_dict[key] = adapted_params[key]
            return adapted_state_dict
        else:
            adapted_state_dict = self.learner.cloned_state_dict()
            adapted_params = OrderedDict()
            for (key, val) in param.items():

                if key == 'mlp.linear5.weight':
                    code = self.task_emb
                    adapted_params[key] = torch.sigmoid(code)*val
                    adapted_state_dict[key] = adapted_params[key]
                else:
                    adapted_params[key] = val
                    adapted_state_dict[key] = adapted_params[key]
            return adapted_state_dict
            
    

                
         
    def forward(self, x, param):
      
        self.aug_vec = self.aug.repeat(x.size(0), 1)
        y = self.learner.forward(x, self.aug_vec, param)
        
        return y
            



