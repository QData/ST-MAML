import torch
import numpy as np
from collections import OrderedDict
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
import torch.distributions.kl as kl
import torch.nn.functional as F


def meta_train_one_epoch(model, dataset, opt, epoch, args, mode='train', device='cuda', print_freq=100):
    if mode == 'train':
        model.train()
        epoch_loss = 0.
        batch_idx = 0
        epoch_rec_loss = 0.
        inner_loop_grad_clip = args.inner_loop_grad_clip
        for task_train, task_val in dataset:
            index = np.random.permutation(args.meta_bs)
            task_train = [task_train[index[i]] for i in range(len(task_train))]
            task_val = [task_val[index[i]] for i in range(len(task_val))]
            batch_loss = []
            rec_loss = []
            for train, val in zip(task_train, task_val):

                x_context = train.x
                y_context = train.y
                x_target = val.x
                y_target = val.y
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)

                model.task_encoder(x_context, y_context, x_target, y_target)
                if not args.aug_enc:
                    model.aug_and_customize()
                else:
                    model.aug_and_customize(x_context, y_context)



                adapted_state_dict = model.encode_param()


                fast_parameters = OrderedDict((name, param) for (name, param) in model.learner.named_parameters()) #the first gradient calcuated in line 45 is based on original weight


                model.zero_grad()

                for task_step in range(args.inner_step):
                    scores = model.forward(x_context, adapted_state_dict)
                    if args.loss_type == 'MSEloss':
                        set_loss = torch.nn.functional.mse_loss(scores, y_context)
                    elif args.loss_type == 'BCEloss':
                        set_loss = F.binary_cross_entropy(torch.sigmoid(scores), y_context)
                    grads_all = torch.autograd.grad(set_loss, list(fast_parameters.values())+[model.post_task_code.loc], create_graph=True)


                    grad_z = grads_all[-1]
                    grads = grads_all[:-1]

                    
                    fast_parameters = OrderedDict((name, param - args.inner_lr * grad.clamp_(min=-args.inner_loop_grad_clip,
                                                                                 max=args.inner_loop_grad_clip)) for
                                               ((name, param), grad) in zip(fast_parameters.items(), grads)) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

                    model.post_task_code.loc = model.post_task_code.loc - args.in_weight_rest*grad_z[0]
                    model.aug_and_customize(x_context, y_context)
                    adapted_state_dict = model.encode_param(fast_parameters)

                y_pred = model.forward(x_target, adapted_state_dict)

                loss, loss_kl = loss_function(y_pred, y_target, args, model.prior_task_code, model.post_task_code)
                batch_loss.append(loss + args.kl_weight*loss_kl)
                rec_loss.append(loss)

            loss_comp = torch.mean(torch.stack(batch_loss))
            loss_single = torch.mean(torch.stack(rec_loss))
            opt.zero_grad()
            loss_comp.backward()
            opt.step()

            batch_idx += 1
            epoch_loss += loss_comp.item()
            epoch_rec_loss += loss_single.item()

        print_str = 'Epoch: {}, Epoch Avg Loss: {:.3f}, Epoch Avg Rec Loss: {:.3f}'.format(epoch,
                                                                                           epoch_loss / batch_idx,
                                                                                           epoch_rec_loss / batch_idx)
        f = open(args.save_epoch_loss, "a")
        f.write(print_str + '\n')
        f.close()
        print(print_str)


def loss_function(pred, gt, args, prior=None, posterior=None):
    if args.model_type=='prob':
        kl_loss = kl.kl_divergence(posterior, prior).mean()
    else:
        kl_loss = 0
    if args.loss_type == 'MSEloss':
        rec_loss = torch.nn.functional.mse_loss(pred, gt)
    elif args.loss_type == 'BCEloss':
        rec_loss = F.binary_cross_entropy(torch.sigmoid(pred), gt)
    # loss = rec_loss + args.kl_weight*kl_loss
    return rec_loss, kl_loss



def meta_eval_one_epoch(model, dataset, opt, epoch, args, mode='eval', device='cuda', print_freq=100):
    if mode == 'eval':
        model.eval()
        epoch_loss = 0.
        batch_idx = 0
        epoch_rec_loss = 0.
        for task_train, task_val in dataset:
            index = np.random.permutation(args.meta_bs)
            task_train = [task_train[index[i]] for i in range(len(task_train))]
            task_val = [task_val[index[i]] for i in range(len(task_val))]
            batch_loss = []
            rec_loss = []
            for train, val in zip(task_train, task_val):

                x_context = train.x
                y_context = train.y
                x_target = val.x
                y_target = val.y
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target = x_target.to(device)
                y_target = y_target.to(device)

                model.task_encoder(x_context, y_context, x_target, None)
                if not args.aug_enc:
                    model.aug_and_customize()
                else:
                    model.aug_and_customize(x_context, y_context)



                adapted_state_dict = model.encode_param()


                fast_parameters = OrderedDict((name, param) for (name, param) in model.learner.named_parameters()) #the first gradient calcuated in line 45 is based on original weight


                model.zero_grad()

                for task_step in range(args.inner_step):
                    scores = model.forward(x_context, adapted_state_dict)
                    if args.loss_type == 'MSEloss':
                        set_loss = torch.nn.functional.mse_loss(scores, y_context)
                    elif args.loss_type == 'BCEloss':
                        set_loss = F.binary_cross_entropy(torch.sigmoid(scores), y_context)
                    grads_all = torch.autograd.grad(set_loss, list(fast_parameters.values())+[model.post_task_code.loc], create_graph=True)


                    grad_z = grads_all[-1]
                    grads = grads_all[:-1]

                    
                    fast_parameters = OrderedDict((name, param - args.inner_lr * grad.clamp_(min=-args.inner_loop_grad_clip,
                                                                                 max=args.inner_loop_grad_clip)) for
                                               ((name, param), grad) in zip(fast_parameters.items(), grads)) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

                    model.post_task_code.loc = model.post_task_code.loc - args.in_weight_rest*grad_z[0]
                    model.aug_and_customize(x_context, y_context)
                    adapted_state_dict = model.encode_param(fast_parameters)

                y_pred = model.forward(x_target, adapted_state_dict)

                loss, loss_kl = loss_function(y_pred, y_target, args, model.prior_task_code, model.post_task_code)
                batch_loss.append(loss + args.kl_weight*loss_kl)
                rec_loss.append(loss)

            loss_comp = torch.mean(torch.stack(batch_loss))
            loss_single = torch.mean(torch.stack(rec_loss))
          
            batch_idx += 1
            epoch_loss += loss_comp.item()
            epoch_rec_loss += loss_single.item()

        print_str = 'Epoch: {}, Eval Avg Loss: {:.3f}, Epoch Avg Eval Loss: {:.3f}'.format(epoch,
                                                                                           epoch_loss / batch_idx,
                                                                                           epoch_rec_loss / batch_idx)
        f = open(args.save_epoch_loss, "a")
        f.write(print_str + '\n')
        f.close()
        return epoch_rec_loss / batch_idx






def meta_test_plot(model, dataset, args, mode='eval', device='cuda'):
    if mode == 'eval':
        output_list = []
        model.eval()
        
        for task_train, task_val in dataset:
            index = np.random.permutation(args.meta_bs)
            task_train = [task_train[index[i]] for i in range(len(task_train))]
            task_val = [task_val[index[i]] for i in range(len(task_val))]
            batch_loss = []
            rec_loss = []
            for train, val in zip(task_train, task_val):

                x_context = train.x
                y_context = train.y
                x_target = val.x
                y_target = val.y
                x_context = x_context.to(device)
                y_context = y_context.to(device)
                x_target_dim_1 = torch.linspace(-5, 5, steps=100).unsqueeze(dim=-1)
                x_target_dim_2 = torch.ones_like(x_target_dim_1)
                x_target = torch.cat([x_target_dim_1, x_target_dim_2], dim=-1)
                x_target = x_target.to(device)

                model.task_encoder(x_context, y_context, x_target, None)
                if not args.aug_enc:
                    model.aug_and_customize()
                else:
                    model.aug_and_customize(x_context, y_context)



                adapted_state_dict = model.encode_param()


                fast_parameters = OrderedDict((name, param) for (name, param) in model.learner.named_parameters()) #the first gradient calcuated in line 45 is based on original weight


                model.zero_grad()

                for task_step in range(args.inner_step):
                    scores = model.forward(x_context, adapted_state_dict)
                    if args.loss_type == 'MSEloss':
                        set_loss = torch.nn.functional.mse_loss(scores, y_context)
                    elif args.loss_type == 'BCEloss':
                        set_loss = F.binary_cross_entropy(torch.sigmoid(scores), y_context)
                    grads_all = torch.autograd.grad(set_loss, list(fast_parameters.values())+[model.post_task_code.loc], create_graph=True)


                    grad_z = grads_all[-1]
                    grads = grads_all[:-1]

                    
                    fast_parameters = OrderedDict((name, param - args.inner_lr * grad.clamp_(min=-args.inner_loop_grad_clip,
                                                                                 max=args.inner_loop_grad_clip)) for
                                               ((name, param), grad) in zip(fast_parameters.items(), grads)) #gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts

                    model.post_task_code.loc = model.post_task_code.loc - args.in_weight_rest*grad_z[0]
                    model.aug_and_customize(x_context, y_context)
                    adapted_state_dict = model.encode_param(fast_parameters)

                y_pred = model.forward(x_target, adapted_state_dict)
                task_list = [train, y_pred]
                output_list.append(task_list)

                

       
        return output_list