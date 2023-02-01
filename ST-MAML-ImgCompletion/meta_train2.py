
import torch
import numpy as np
from collections import OrderedDict
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
import torch.distributions.kl as kl
import torch.nn.functional as F


def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()


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

                coded_param = model.encode_param()

                y_pred = model(x_context, coded_param)
                if args.loss_type == 'MSEloss':
                    loss = torch.nn.functional.mse_loss(y_pred, y_context)
                elif args.loss_type == 'BCEloss':
                    loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)


                fast_weights = OrderedDict((name, param) for (name, param) in coded_param.items())
                for i in range(args.inner_step):
                    if i == 0:
                        grads = torch.autograd.grad(loss, coded_param.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        # grads_taskemb = torch.autograd.grad(loss, model.task_emb, create_graph=True)
                    else:
                        y_pred = model(x_context, fast_weights)
                        if args.loss_type == 'MSEloss':
                            loss = torch.nn.functional.mse_loss(y_pred, y_context)
                        elif args.loss_type == 'BCEloss':
                            loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)
                        grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        # grads_taskemb = torch.autograd.grad(loss, model.task_emb, create_graph=True)
                    fast_weights = OrderedDict((name, param - args.inner_lr * grad.clamp_(min=-args.inner_loop_grad_clip,
                                                                                 max=args.inner_loop_grad_clip)) for
                                               ((name, param), grad) in zip(fast_weights.items(), grads))
                    model.aug = model.aug - args.in_weight_rest*grads_aug[0]
                    # model.task_emb = model.task_emb - args.in_weight_rest*grads_taskemb[0]
                    # out_adapted_params = model.encode_param(fast_weights)

                y_pred = model(x_target, fast_weights)

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

        print_str = 'Train | Epoch: {} Epoch Avg Loss: {:.3f}, Epoch Avg Rec Loss: {:.3f}'.format(epoch,
                                                                                           epoch_loss / batch_idx,
                                                                                           epoch_rec_loss / batch_idx)
        f = open(args.save_epoch_loss, "a")
        f.write(print_str + '\n')
        f.close()
        print(print_str)




def meta_eval_one_epoch(model, dataset, epoch, args, mode='eval', device='cuda'):
    if mode == 'eval':
        model.eval()
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

                model.task_encoder(x_context, y_context, x_target, None)
                if not args.aug_enc:
                    model.aug_and_customize()
                else:
                    model.aug_and_customize(x_context, y_context)

                coded_param = model.encode_param()

                y_pred = model(x_context, coded_param)
                if args.loss_type == 'MSEloss':
                    loss = torch.nn.functional.mse_loss(y_pred, y_context)
                elif args.loss_type == 'BCEloss':
                    loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)


                fast_weights = OrderedDict((name, param) for (name, param) in coded_param.items())
                for i in range(args.inner_step):
                    if i == 0:
                        grads = torch.autograd.grad(loss, coded_param.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        # grads_taskemb = torch.autograd.grad(loss, model.task_emb, create_graph=True)
                    else:
                        y_pred = model(x_context, fast_weights)
                        if args.loss_type == 'MSEloss':
                            loss = torch.nn.functional.mse_loss(y_pred, y_context)
                        elif args.loss_type == 'BCEloss':
                            loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)
                        grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        # grads_taskemb = torch.autograd.grad(loss, model.task_emb, create_graph=True)
                    fast_weights = OrderedDict((name, param - args.inner_lr * grad.clamp_(min=-args.inner_loop_grad_clip,
                                                                                 max=args.inner_loop_grad_clip)) for
                                               ((name, param), grad) in zip(fast_weights.items(), grads))
                    model.aug = model.aug - args.in_weight_rest*grads_aug[0]
                    # model.task_emb = model.task_emb - args.in_weight_rest*grads_taskemb[0]
                    # out_adapted_params = model.encode_param(fast_weights)

                y_pred = model(x_target, fast_weights)

                loss, loss_kl = loss_function(y_pred, y_target, args, model.prior_task_code, model.post_task_code)


                batch_loss.append(loss + args.kl_weight*loss_kl)
                rec_loss.append(loss)

            loss_comp = torch.mean(torch.stack(batch_loss))
            loss_single = torch.mean(torch.stack(rec_loss))
            batch_idx += 1
            epoch_loss += loss_comp.item()
            epoch_rec_loss += loss_single.item()

        print_str = 'Eval | Epoch: {}, Epoch Avg Loss: {:.3f}, Epoch Avg Rec Loss: {:.3f}'.format(epoch,
                                                                                           epoch_loss / batch_idx,
                                                                                           epoch_rec_loss / batch_idx)
        f = open(args.save_epoch_loss, "a")
        f.write(print_str + '\n')
        f.close()
        print(print_str)
        return print_str, epoch_loss/batch_idx



def meta_test_one_epoch(model, dataset, args, mode='test', device='cuda'):
    if mode == 'test':
        model.eval()
        epoch_loss = 0.
        batch_idx = 0
        epoch_rec_loss = 0.
        inner_loop_grad_clip = args.inner_loop_grad_clip
        var_list = []
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

                coded_param = model.encode_param()

                y_pred = model(x_context, coded_param)
                if args.loss_type == 'MSEloss':
                    loss = torch.nn.functional.mse_loss(y_pred, y_context)
                elif args.loss_type == 'BCEloss':
                    loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)


                fast_weights = OrderedDict((name, param) for (name, param) in coded_param.items())
                for i in range(args.inner_step):
                    if i == 0:
                        grads = torch.autograd.grad(loss, coded_param.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        # grads_taskemb = torch.autograd.grad(loss, model.task_emb, create_graph=True)
                    else:
                        y_pred = model(x_context, fast_weights)
                        if args.loss_type == 'MSEloss':
                            loss = torch.nn.functional.mse_loss(y_pred, y_context)
                        elif args.loss_type == 'BCEloss':
                            loss = binary_cross_entropy(torch.sigmoid(y_pred), y_context)
                        grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                        grads_aug = torch.autograd.grad(loss, model.aug, create_graph=True)
                        # grads_taskemb = torch.autograd.grad(loss, model.task_emb, create_graph=True)
                    fast_weights = OrderedDict((name, param - args.inner_lr * grad.clamp_(min=-args.inner_loop_grad_clip,
                                                                                 max=args.inner_loop_grad_clip)) for
                                               ((name, param), grad) in zip(fast_weights.items(), grads))
                    model.aug = model.aug - args.in_weight_rest*grads_aug[0]
                    # model.task_emb = model.task_emb - args.in_weight_rest*grads_taskemb[0]
                    # out_adapted_params = model.encode_param(fast_weights)

                y_pred = model(x_target, fast_weights)

                loss, loss_kl = loss_function(y_pred, y_target, args, model.prior_task_code, model.post_task_code)


                batch_loss.append(loss + args.kl_weight*loss_kl)
                rec_loss.append(loss)
                

            loss_comp = torch.mean(torch.stack(batch_loss))
            loss_single = torch.mean(torch.stack(rec_loss))
            batch_idx += 1
            epoch_loss += loss_comp.item()
            epoch_rec_loss += loss_single.item()
            var_list.append(loss_single.item())
        breakpoint()
        print_str = 'Eval | Epoch Avg Loss: {:.3f}, Epoch Avg Rec Loss: {:.3f}'.format(
                                                                                           epoch_loss / batch_idx,
                                                                                           epoch_rec_loss / batch_idx)
        print(print_str)
        return print_str, epoch_loss/batch_idx











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
