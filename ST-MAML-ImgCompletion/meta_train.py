import torch
import numpy as np
from collections import OrderedDict
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
import torch.nn.functional as F



def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()

def meta_train_one_epoch(model, dataset, opt, epoch, args, mode='train', device='cuda', print_freq=100):
   
    if mode == 'train':
        model.train()
        epoch_loss = 0.
        batch_idx = 0
        inner_loop_grad_clip = args.inner_loop_grad_clip
        for task_train, task_val in dataset:
            index = np.random.permutation(args.meta_bs)
            task_train = [task_train[index[i]] for i in range(len(task_train))]
            task_val = [task_val[index[i]] for i in range(len(task_val))]
            batch_loss = []

            for train, val in zip(task_train, task_val):
  
                x_context = train.x
                y_context = train.y
                x_target = val.x
                y_target = val.y
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)

                model.task_encoder(x_context, y_context)
                model.param_encoder()
                coded_param = model.encode_param()
                
            

            
                y_pred = model(x_context, coded_param)
                if args.loss_type == 'MSEloss':
                    loss = torch.nn.functional.mse_loss(y_pred, y_context)
                elif args.loss_type == 'BCEloss':
                    loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)
                
                for i in range(args.inner_step):
                    if i==0:
                        grads = torch.autograd.grad(loss, coded_param.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        
                    else:
                        y_pred = model(x_context, out_adapted_params)
                        if args.loss_type == 'MSEloss':
                            loss = torch.nn.functional.mse_loss(y_pred, y_context)
                        elif args.loss_type == 'BCEloss':
                            loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)
                        grads = torch.autograd.grad(loss, out_adapted_params.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        coded_param = out_adapted_params
                        
                    fast_weights = OrderedDict((name, param - 1e-3*grad.clamp_(min=-args.inner_loop_grad_clip, max=args.inner_loop_grad_clip)) for ((name, param), grad) in zip(coded_param.items(), grads))
                    model.aug = model.aug - grads_aug[0]
                    model.param_encoder()
                    out_adapted_params = model.encode_param(fast_weights)
           
               
                y_pred = model(x_target, out_adapted_params)
                
                if args.loss_type == 'MSEloss':
                    loss = torch.nn.functional.mse_loss(y_pred, y_target)
                elif args.loss_type == 'BCEloss':
                    loss = binary_cross_entropy(torch.sigmoid(y_pred), y_target)



                batch_loss.append(loss)


            loss = torch.mean(torch.stack(batch_loss))
            opt.zero_grad()
            loss.backward()
            opt.step()

            batch_idx += 1
            epoch_loss += loss.item()

        print_str = 'Epoch: {}, Epoch Avg Loss: {:.3f}'.format(epoch, epoch_loss/batch_idx)
        f = open(args.save_epoch_loss, "a")
        f.write(print_str + '\n')
        f.close()
        print(print_str)  
    


def meta_test_one_epoch(model, dataset, opt, epoch, args, mode='eval', device='cuda', print_freq=100):
   
    if mode == 'eval':
        model.eval()
        epoch_loss = 0.
        batch_idx = 0
        inner_loop_grad_clip = args.inner_loop_grad_clip
        for task_train, task_val in dataset:
            index = np.random.permutation(args.meta_bs)
            task_train = [task_train[index[i]] for i in range(len(task_train))]
            task_val = [task_val[index[i]] for i in range(len(task_val))]
            batch_loss = []

            for train, val in zip(task_train, task_val):
  
                x_context = train.x
                y_context = train.y
                x_target = val.x
                y_target = val.y
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)

                model.task_encoder(x_context, y_context)
                model.param_encoder()
                coded_param = model.encode_param()
                
            

            
                y_pred = model(x_context, coded_param)
                if args.loss_type == 'MSEloss':
                    loss = torch.nn.functional.mse_loss(y_pred, y_context)
                elif args.loss_type == 'BCEloss':
                    loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)
                
                for i in range(args.inner_step):
                    if i==0:
                        grads = torch.autograd.grad(loss, coded_param.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        
                    else:
                        y_pred = model(x_context, out_adapted_params)
                        if args.loss_type == 'MSEloss':
                            loss = torch.nn.functional.mse_loss(y_pred, y_context)
                        elif args.loss_type == 'BCEloss':
                            loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)
                        grads = torch.autograd.grad(loss, out_adapted_params.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        coded_param = out_adapted_params
                        
                    fast_weights = OrderedDict((name, param - 1e-3*grad.clamp_(min=-args.inner_loop_grad_clip, max=args.inner_loop_grad_clip)) for ((name, param), grad) in zip(coded_param.items(), grads))
                    model.aug = model.aug - grads_aug[0]
                    model.param_encoder()
                    out_adapted_params = model.encode_param(fast_weights)
           
               
                y_pred = model(x_target, out_adapted_params)
                
                if args.loss_type == 'MSEloss':
                    loss = torch.nn.functional.mse_loss(y_pred, y_target)
                elif args.loss_type == 'BCEloss':
                    loss = binary_cross_entropy(torch.sigmoid(y_pred), y_target)



                batch_loss.append(loss)

            loss = torch.mean(torch.stack(batch_loss))
   
            batch_idx += 1
            epoch_loss += loss.item()

        print_str = 'Epoch: {}, Test Epoch Avg Loss: {:.3f}'.format(epoch, epoch_loss/batch_idx)
        print(print_str) 




