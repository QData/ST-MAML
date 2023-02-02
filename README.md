## ST-MAML

Code for the UAI 2022 paper "ST-MAML: A Stochastic-Task based Method forTask-Heterogeneous Meta-Learning (https://openreview.net/forum?id=rrlMyPUs9gc)"

Code is adapted from [A Closer Look at Few-Shot Classification](https://github.com/wyharveychen/CloserLookFewShot).


To reproduce the temperature prediction expriments, run with:

> cd ST-MAML-Weather

> python main.py --method ST_MAML

> python main.py --method MAML --model MLP_MAML


For the cross dataset image completion experiments, run with:

> cd ST-MAML-ImgCompletion

> python meta_main.py

For the regression fitting, train a model with: 

> cd ST-MAML-Reg

> python python meta_main.py --aug_enc --kl_weight=2.0  --in_weight_rest=0.1 --model_type='prob' --output_folder='results'

For visualization purpose, run with:

> python visual.py --aug_enc --kl_weight=2.0  --in_weight_rest=0.1 --model_type='prob' --output_folder='results' 