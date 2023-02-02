from random import choices
import backbone
import argparse
import os
import glob
import numpy as np
model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4S = backbone.Conv4S,
            Conv6 = backbone.Conv6,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101,
            MLP = backbone.Regressor,
            MLP_MAML = backbone.RegressorMAML)


def get_arguments():
    parser = argparse.ArgumentParser(description='ST-MAML')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gpu_id', type=int, default=5)
    parser.add_argument('--meta_bs', type=int, default=20)
    parser.add_argument('--stop_epoch', type=int, default=200)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--inner_loop_grad_clip', type=float, default=10.)
    parser.add_argument('--resume_train', default=True, action='store_true')
    parser.add_argument('--inner_lr', default=1e-3, type=float)
    parser.add_argument('--inner_step', default=5, type=int)
    parser.add_argument('--output_folder', default='./new_results/')
    parser.add_argument('--kl_weight', default=0.1, type=float) 
    parser.add_argument('--in_weight_rest', default=1e-3, type=float)
    parser.add_argument('--aug_enc', default=False, action='store_true')
    parser.add_argument('--act_type', default='softplus', choices=['softplus', 'sigmoid'])
    parser.add_argument('--model', default='MLP', choices=['Conv4', 'Conv4s', 'MLP', 'MLP_MAML'])
    parser.add_argument('--train_n_way' , default=1, type=int,  help='class num to classify for training') #baseline and baseline++ would ignore this parameter
    parser.add_argument('--test_n_way'  , default=1, type=int,  help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot'  , default=10, type=int)
    parser.add_argument('--img_size', default=84, type=int)
    parser.add_argument('--train_aug', default=False, action='store_true', help='data augmentation during traning or not')
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--method', default='ST_MAML')
    parser.add_argument('--loader_type', default=1, type=int, choices=[0, 1])
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--aug_fraction', default=0.25, type=float)
    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
