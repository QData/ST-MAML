import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from meta_template import MetaTemplate


class MAML(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, approx = False):
        super(MAML, self).__init__( model_func,  n_way, n_support)

        self.task_enc_dim=120
        self.classifier = backbone.Linear_fw(self.hid_dim[-1], self.out_dim)    
        self.feature = model_func()
        self.loss_fn = nn.MSELoss()
        self.n_task  = 4
        self.task_update_num = 5
        self.train_lr = 0.01
        self.in_weight_rest = 0.1

        

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores




    def set_forward(self,x, y, is_feature = False):
        x = x.cuda()
        y = y.cuda()
        x_var = x
        y_var = y
        x_a_i = x_var[:self.n_support] #support data
        x_b_i = x_var[self.n_support:,] #query data
        y_a_i = y_var[:self.n_support].view(-1, 1)
        
        for weight in list(self.parameters()):
            weight.fast = None

        fast_parameters = list(self.parameters()) #the first gradient calcuated in line 45 is based on original weight


        self.zero_grad()

        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)
            set_loss = self.loss_fn( scores, y_a_i)
            
            """
            CLIP = 1.
            grads = [x.clamp(-CLIP, CLIP) for x in torch.autograd.grad(set_loss, fast_parameters+[self.aug, self.task_emb], create_graph=True, allow_unused=True)]
            """
            grads = torch.autograd.grad(set_loss, fast_parameters, create_graph=True, allow_unused=True)
            
            fast_parameters = []
            for k, weight in enumerate(list(self.feature.parameters()) + list((self.classifier.parameters()))):
                #for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
                if weight.fast is None:
                    if grads[k] is None:
                        print(grads[k].size())
                    
                    weight.fast = weight - self.train_lr * grads[k].clamp_(min=-3,max=3) #create weight.fast
                else:
                    if grads[k] is None:
                        print(grads[k].size())
                    weight.fast = weight.fast - self.train_lr * grads[k].clamp_(min=-3,max=3) #create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(weight.fast) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
           
        
        scores = self.forward(x_b_i)
        return scores

    def set_forward_adaptation(self,x, is_feature = False): #overwrite parrent function
        raise ValueError('MAML performs further adapation simply by increasing task_upate_num')


    def set_forward_loss(self, x, y):
        scores = self.set_forward(x, y, is_feature = False)
        y_b_i =  y[self.n_support:,].view(-1, 1).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return loss

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
                loss = self.set_forward_loss(x, y)
               
                avg_loss_nll = avg_loss_nll+loss.item()
                
                loss_all.append(loss)
            
            loss_q = torch.stack(loss_all).sum(0)
            loss_q.backward()

            optimizer.step()
            optimizer.zero_grad()
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | NLL Loss {:f} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss_nll/float(batch_x.size(0)),  avg_loss/float(batch_x.size(0))))
            
    def test_loop(self, test_loader, return_std = False): #overwrite parrent function
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
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f +- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean



    def correct(self, x, y):       
        scores = self.set_forward(x, y)
        y_b_i =  y[self.n_support:,].view(-1, 1).cuda()
        loss = self.loss_fn(scores, y_b_i)
        return loss





