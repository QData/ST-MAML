## ST-MAML

Code for the UAI 2022 paper "ST-MAML: A Stochastic-Task based Method forTask-Heterogeneous Meta-Learning (https://openreview.net/forum?id=rrlMyPUs9gc)"

Code is adapted from [A Closer Look at Few-Shot Classification](https://github.com/wyharveychen/CloserLookFewShot).


To reproduce the temperature prediction expriments, run with:

cd ST-MAML-Weather
> python main.py --method ST_MAML
> python main.py --method MAML --model MLP_MAML

For the cross dataset image completion experiments, we provide both the probabistic and deterministic model:

cd ST-MAML-ImgCompletion
> python meta_main.py --aug_enc --kl_weight=0.1 --in_weight_rest=1.0 --inner_lr=0.005 --model_type='prob'
> python meta_main.py --aug_enc --in_weight_rest=0.5 --inner_lr=0.005 --model_type='deter'


