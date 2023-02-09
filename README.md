# MetaBIN-paddle
【飞桨论文复现营】Meta Batch-Instance Normalization for Generalizable Person Re-Identification, [[CVPR 2021](https://arxiv.org/abs/2011.14670)]

# 运行环境
Cuda   10.2  
Paddle 2.4.0  

# 模型指标
Market1501 --> DukeMTMC
| Models           | Recall@1 | Recall@5 | Recall@10 | mAP |
|:--:|:--:|:--:|:--:|:--:|
| ResNet50 (MetaBIN)            | 55.2 | 69.0 | 74.4 | 33.1 |

DukeMTMC --> Market1501
| Models           | Recall@1 | Recall@5 | Recall@10 | mAP |
|:--:|:--:|:--:|:--:|:--:|
| ResNet50 (MetaBIN)            | 69.2 | 83.1 | 87.8 | 35.9 |
