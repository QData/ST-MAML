# ST-MAML: A Stochastic-Task based Method for Task-Heterogeneous Meta-Learning

Code repository for UAI 2022 paper "ST-MAML: A Stochastic-Task based Method for Task-Heterogeneous Meta-Learning". [[paper]](https://openreview.net/forum?id=rrlMyPUs9gc)

## Run Experiments
```
# Temperature prediction expriments

cd ST-MAML-Weather

python main.py --method ST_MAML

# Cross dataset image completion experiments

cd ST-MAML-ImgCompletion

python meta_main.py

# Regression fitting

cd ST-MAML-Reg

python python meta_main.py --aug_enc --kl_weight=2.0  --in_weight_rest=0.1 --model_type='prob' --output_folder='results'

For visualization purpose, 

python visual.py --aug_enc --kl_weight=2.0  --in_weight_rest=0.1 --model_type='prob' --output_folder='results' 
```

## Acknowledgements
The code for ST-MAML is based on [A Closer Look at Few-Shot Classification](https://github.com/wyharveychen/CloserLookFewShot).

## References
Please cite our paper as:
```		
@inproceedings{
wang2022stmaml,
title={{ST}-{MAML}: A Stochastic-Task based Method for Task-Heterogeneous Meta-Learning},
author={Zhe Wang and Jake Grigsby and Arshdeep Sekhon and Yanjun Qi},
booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
year={2022},
url={https://openreview.net/forum?id=rrlMyPUs9gc}
}
```