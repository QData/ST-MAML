import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models 
import torch.distributions as distributions
from collections import OrderedDict
import torch.nn.init as init
import torch.distributions as distributions


def linear_block(iput, aug, param, prefix):

    n_data, _ = iput.size()
    out = F.linear(iput, weight=param[prefix+'linear1.weight'], bias=param[prefix+'linear1.bias'])
    out = F.relu(out, inplace=True)
    out = torch.cat([out, aug], dim=-1)
   
    out = F.linear(out, weight=param[prefix+'linear2.weight'], bias=param[prefix+'linear2.bias'])
    out = F.relu(out, inplace=True)

    out = F.linear(out, weight=param[prefix+'linear3.weight'], bias=param[prefix+'linear3.bias'])
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
        self.mlp.add_module('linear3', nn.Linear(n_h[1], n_output))
        

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
    
    
    
class ParamEnc(nn.Module):
    def __init__(self, n_xyaug, n_hid, n_out):
        super(ParamEnc, self).__init__()
        self.param_encoder = nn.Sequential()
        self.param_encoder.add_module('xy2hid', nn.Linear(n_xyaug, n_hid[0]))
        self.param_encoder.add_module('relu1', nn.ReLU(inplace=True))
        self.param_encoder.add_module('hid2hid', nn.Linear(n_hid[0], n_hid[1]))
        self.param_encoder.add_module('relu2', nn.ReLU(inplace=True))
        self.param_encoder.add_module('linear3', nn.Linear(n_hid[1], n_out))
        
        for layer in self.param_encoder.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)


    def forward(self, aug, x=None, y=None):
        if y is None:
            out = self.param_encoder(aug)
            out = out.mean(dim=0, keepdim=True)
        else:
            n_spt, _ = x.size()
            aug = aug.repeat(n_spt, 1)
            xyaug_concat = torch.cat([x, y, aug] ,dim=-1)
            out = self.param_encoder(xyaug_concat)
            out = out.mean(dim=0, keepdim=True)
        return out







class TaskEnc(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(TaskEnc, self).__init__()
        self.task_enc = nn.Sequential()
        self.task_enc.add_module('xy2hid', nn.Linear(n_in, n_hid[0]))
        self.task_enc.add_module('relu1', nn.ReLU(inplace=True))
        self.task_enc.add_module('hid2hid', nn.Linear(n_hid[0], n_hid[1]))
        self.task_enc.add_module('relu2', nn.ReLU(inplace=True))
        self.task_enc.add_module('hid2out', nn.Linear(n_hid[1], n_out))

        for layer in self.task_enc.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)
        
     
        
    def forward(self, x, y):
        n_spt, _ = x.size()
        xy_concat = torch.cat([x, y] ,dim=-1)
        out = self.task_enc(xy_concat)
        out = out.mean(dim=0, keepdim=True)
        return out

        
class Deter2Var(nn.Module):
    def __init__(self, n_in, n_out, act_type):
        super(Deter2Var, self).__init__()
        self.deter2mean = nn.Linear(n_in, n_out)
        self.deter2var = nn.Linear(n_in, n_out)
        self.act_type = act_type
    
    def forward(self, iput):
        mean = self.deter2mean(iput)
        prevar = self.deter2var(iput)
        if self.act_type == 'softplus':
            var = 0.1 + 0.9 * F.softplus(prevar)
        elif self.act_type == 'sigmoid':
            var = 0.1 + 0.9 * torch.sigmoid(prevar)
        dist = distributions.Normal(mean, var)
        return dist

        
        
        
        

class EncModel(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_xyaug, n_hid_emb, n_out_emb, n_xy, n_hid_aug, n_out_aug, aug_enc=False, model_type='prob', act_type='softplus'):
        super(EncModel, self).__init__()

        self.aug_enc = aug_enc
        self.deter_taskenc = TaskEnc(n_xy, n_hid_aug, n_out_aug)
        self.model_type = model_type
        if model_type=='prob':
            self.var_taskenc = Deter2Var(n_out_aug, n_out_aug, act_type)
        elif model_type=='deter':
            self.var_taskenc = nn.Linear(n_out_aug, n_out_aug)

        # self.var_taskenc = Deter2Var(n_out_aug, n_out_aug, act_type)
           
        self.learner = LR(n_in, n_hid, n_out, n_out_aug)
        if not aug_enc:
            self.encoder = ParamEnc(n_out_aug, n_hid_emb, n_out_emb)
        else:
            self.encoder = ParamEnc(n_out_aug+n_in+n_out, n_hid_emb, n_out_emb)
        self.aug_vec = nn.Linear(n_out_aug, n_out_aug)
            
        
    def task_encoder(self, x_spt, y_spt, x_qry, y_qry=None):
        if self.model_type == 'prob':
            prior_deter_task_code = self.deter_taskenc(x_spt, y_spt)
            self.prior_task_code = self.var_taskenc(prior_deter_task_code)
            if y_qry is not None:
                x_comb = torch.cat([x_spt, x_qry], dim=0)
                y_comb = torch.cat([y_spt, y_qry], dim=0)
                post_deter_task_code = self.deter_taskenc(x_comb, y_comb)
                self.post_task_code = self.var_taskenc(post_deter_task_code)
            else:
                self.post_task_code = self.prior_task_code

            
        elif self.model_type == 'deter':
            prior_deter_task_code = self.deter_taskenc(x_spt, y_spt)
            self.prior_task_code = self.var_taskenc(prior_deter_task_code)
            self.task_code = self.prior_task_code
            self.post_task_code=None

    def aug_and_customize(self, x=None, y=None):
        self.task_code = self.post_task_code.rsample()
        if y is None:
            self.aug = self.aug_vec(self.task_code)
            self.task_emb = self.encoder(self.task_code)
        else:
            self.aug = self.aug_vec(self.task_code)
            self.task_emb = self.encoder(self.task_code, x, y)


        
            

        
    def encode_param(self, param=None):
        if param == None:
            adapted_state_dict = self.learner.cloned_state_dict()
            adapted_params = OrderedDict()
            for (key, val) in self.learner.named_parameters():

                if key == 'mlp.linear3.weight':
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

                if key == 'mlp.linear3.weight':
                    code = self.task_emb
                    adapted_params[key] = torch.sigmoid(code)*val
                    adapted_state_dict[key] = adapted_params[key]
                else:
                    adapted_params[key] = val
                    adapted_state_dict[key] = adapted_params[key]
            return adapted_state_dict
            
    

                
         
    def forward(self, x, param):
      
        self.aug_repeat = self.aug.repeat(x.size(0), 1)
        y = self.learner.forward(x, self.aug_repeat, param)
        
        return y
            



