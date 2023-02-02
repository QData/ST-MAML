import backbone
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from meta_template import MetaTemplate 
import torch.nn.init as init
from collections import OrderedDict
import torch.distributions as distributions
import torch.distributions.kl as kl




class ST_MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = False):
        super(ST_MAML, self).__init__( model_func,  n_way, n_support)

        self.task_enc_dim=200     #80
        self.task_enc = TaskEnc(self.in_dim+self.out_dim, hid_dim=[40, 80], out_dim=self.task_enc_dim)
        self.classifier = Linear_fw(self.hid_dim[-1], self.out_dim)    
        self.feature = model_func()
        self.loss_fn = nn.MSELoss()
        #self.loss_fn = nn.SmoothL1Loss(beta=1.)
        self.aug_vec = nn.Linear(self.task_enc_dim, self.aug_dim)
        self.encoder = ParamEnc(self.task_enc_dim, self.hid_dim[-1])
        self.var_taskenc = Deter2Var(self.task_enc_dim, self.task_enc_dim)
        self.n_task     = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.in_weight_rest = 0.1
        self.framework = 'prob'
        self.kl_weight = 0.1
        

    def forward(self,x, adapted_state_dict):
        x = torch.cat([x, self.aug.repeat(x.size(0), 1)], dim=-1)
        out  = self.feature.forward(x)
        scores = F.linear(out, adapted_state_dict['weight'], None)
        return scores



    def task_encoder(self, x_spt, y_spt, x_qry=None, y_qry=None):

        if self.framework=='prob':
            piror_task_code = self.task_enc(x_spt, y_spt)
            self.prior_task_dist = self.var_taskenc(piror_task_code)
            if y_qry is not None:
                x_qry = torch.cat([x_spt, x_qry], dim=0)
                y_qry = torch.cat([y_spt, y_qry], dim=0)
                post_task_code = self.task_enc(x_qry, y_qry)
                self.post_task_dist = self.var_taskenc(post_task_code)
            else:
                self.post_task_dist = self.prior_task_dist

            self.task_code = self.post_task_dist.rsample()
        elif self.framework=='deter':
            prior_task_code = self.task_enc(x_spt, y_spt)
            self.prior_task_code = self.var_taskenc(prior_task_code)
            self.task_code = self.prior_task_code
            self.post_task_code=None
        
  

    def aug_and_customize(self, x=None, y=None):
        if y is None:
            self.aug = self.aug_vec(self.task_code)
            self.task_emb = self.encoder(self.task_code)
        else:
            self.aug = self.aug_vec(self.task_code)
            self.task_emb = self.encoder(self.task_code, x, y)


    def encode_param(self):
        if self.classifier.weight.fast is None:
            adapted_state_dict = OrderedDict((name, param) for (name, param) in self.classifier.named_parameters())

            for (key, val) in self.classifier.named_parameters():

                if key == 'weight':
                    code = self.task_emb
                    adapted_state_dict[key] = torch.sigmoid(code) * val

                else:
                    adapted_state_dict[key] = val
            return adapted_state_dict
        else:
            adapted_state_dict = OrderedDict((name, param) for (name, param) in self.classifier.named_parameters())

            for (key, val) in self.classifier.named_parameters():

                if key == 'weight':
                    code = self.task_emb
                    adapted_state_dict[key] = torch.sigmoid(code) * val.fast

                else:
                    adapted_state_dict[key] = val
            return adapted_state_dict




    def set_forward(self,x, y, is_feature = False):
        assert is_feature == False, 'MAML do not support fixed feature'
        x = x.cuda()
        y = y.cuda()
        x_var = x
        y_var = y
        x_a_i = x_var[:self.n_support] #support data
        x_b_i = x_var[self.n_support:,] #query data
        y_a_i = y_var[:self.n_support].view(-1, 1)
        if self.training:
            y_b_i =  y_var[self.n_support:,].view(-1, 1) #label for support data
        else:
            y_b_i = None
        for weight in list(self.feature.parameters())+list(self.classifier.parameters()):
            weight.fast = None
        self.task_encoder(x_a_i, y_a_i, x_b_i, y_b_i)

        self.aug_and_customize()


        adapted_state_dict = self.encode_param()


        fast_parameters = list(self.feature.parameters())+list(self.classifier.parameters())


        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i, adapted_state_dict)
            set_loss = self.loss_fn( scores, y_a_i)
            
    
            grads = torch.autograd.grad(set_loss, fast_parameters+[self.aug, self.task_emb], create_graph=True, allow_unused=True)
            grad_aug = grads[-2]
            grad_taskemb = grads[-1]
            grad = grads[:-2]
            
            fast_parameters = []
            for k, weight in enumerate(list(self.feature.parameters()) + list((self.classifier.parameters()))):

                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k].clamp_(min=-3,max=3) #create weight.fast
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k].clamp_(min=-3,max=3)
                fast_parameters.append(weight.fast)
            self.aug = self.aug -  self.in_weight_rest*grad_aug[0].clamp_(min=-1,max=1)
            self.task_emb = self.task_emb - self.in_weight_rest*grad_taskemb[0].clamp_(min=-1,max=1)
            adapted_state_dict = self.encode_param()
            

        scores = self.forward(x_b_i, adapted_state_dict)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x, y):
        scores = self.set_forward(x, y, is_feature = False)
        y_b_i =  y[self.n_support:,].view(-1, 1).cuda()
        loss = self.loss_fn(scores, y_b_i)
        loss_kl= kl.kl_divergence(self.post_task_dist, self.prior_task_dist).mean()

        return loss, loss_kl

    def train_loop(self, epoch, train_loader, optimizer): #overwrite parrent function
        print_freq = 100

        #train
        for i, (batch_x, batch_y, _) in enumerate(train_loader):
            
            avg_loss=0
            loss_all = []
            avg_loss_nll = 0
            optimizer.zero_grad()
            for task_id in range(batch_x.size(0)):
                x = batch_x[task_id]
                y = batch_y[task_id]
                loss, loss_kl = self.set_forward_loss(x, y)
                loss_whole = loss + self.kl_weight*loss_kl
                avg_loss_nll = avg_loss_nll+loss.item()
                
                avg_loss = avg_loss+loss_whole.item()
                loss_all.append(loss_whole)
            
            loss_q = torch.stack(loss_all).sum(0)
            loss_q.backward()

            optimizer.step()
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | NLL Loss {:f} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss_nll/float(batch_x.size(0)),  avg_loss/float(batch_x.size(0))))
            
    def test_loop(self, test_loader):        
        correct =0
        count = 0
        acc_all = []

        iter_num = len(test_loader)
        for i, (x,y, _) in enumerate(test_loader):
            x = x.squeeze(dim=0).cuda()
            y = y.squeeze(dim=0).cuda()
            loss = self.correct(x, y)
            acc_all.append(loss.item())
            if i == len(test_loader):
                break
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        print('%d Test MSE = %4.2f' %(iter_num,  acc_mean))
       
        return acc_mean



    def correct(self, x, y):       
        scores = self.set_forward(x, y)
        y_b_i =  y[self.n_support:,].view(-1, 1).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return loss





class TaskEnc(nn.Module):
    def __init__(self, task_xy,  hid_dim, out_dim):
        super(TaskEnc, self).__init__()
        trunk = [nn.Linear(task_xy, hid_dim[0]), nn.ReLU(), nn.Linear(hid_dim[0], hid_dim[1]), nn.ReLU(),
        nn.Linear(hid_dim[1], out_dim)]
        self.trunk = nn.Sequential(*trunk)




    def forward(self, x, y):
        iput = torch.cat([x, y], dim=1)
        out = self.trunk(iput)
        out = out.mean(dim=0, keepdim=True)
        return out




class ParamEnc(nn.Module):
    def __init__(self, task_enc_dim, clf_in_dim):
        super(ParamEnc, self).__init__()
        self.param_encoder = nn.Sequential()
        self.param_encoder.add_module('xy2hid', nn.Linear(task_enc_dim, clf_in_dim))
        self.param_encoder.add_module('relu1', nn.ReLU(inplace=True))
        # self.param_encoder.add_module('hid2out', nn.Linear(cls_in_dim, cls_out_dim))


        for layer in self.param_encoder.modules():
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    init.constant_(layer.bias, 0)

    def forward(self, aug, x=None, y=None):
        if y is None:
            out = self.param_encoder(aug)
            # out = out.mean(dim=0, keepdim=True)
        else:
            n_spt, _ = x.size()
            aug = aug.repeat(n_spt, 1)
            xyaug_concat = torch.cat([x, y, aug], dim=-1)
            out = self.param_encoder(xyaug_concat)
            # out = out.mean(dim=0, keepdim=True)
        return out

class Deter2Var(nn.Module):
    def __init__(self, n_in, n_out, act_type='softplus'):
        super(Deter2Var, self).__init__()
        self.deter2mean = nn.Linear(n_in, n_out)
        self.deter2var = nn.Linear(n_in, n_out)
        self.act_type=act_type

    def forward(self, iput):
        mean = self.deter2mean(iput)
        prevar = self.deter2var(iput)
        if self.act_type == 'softplus':
            var = 0.1 + 0.9 * F.softplus(prevar)
        elif self.act_type == 'sigmoid':
            var = 0.1 + 0.9 * torch.sigmoid(prevar)
        dist = distributions.Normal(mean, var)
        return dist


class Linear_fw(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features, bias=None)
        self.weight.fast = None
        

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.linear(x, self.weight.fast, None)
        else:
            out = super(Linear_fw, self).forward(x)
        return out
