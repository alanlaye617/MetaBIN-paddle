# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import paddle
import paddle.nn.functional as F
from .utils import euclidean_dist, normalize, cosine_dist, cosine_sim


def domain_SCT_loss(embedding, domain_labels, norm_feat, type):

    # type = 'cosine' # 'cosine', 'euclidean'
    # eps=1e-05
    if norm_feat: embedding = normalize(embedding, axis=-1)
    unique_label = paddle.unique(domain_labels)
    embedding_all = list()
    for i, x in enumerate(unique_label):
        embedding_all.append(embedding[x == domain_labels])
    num_domain = len(embedding_all)
    loss_all = []
    for i in range(num_domain):
        feat = embedding_all[i]
        center_feat = paddle.mean(feat, 0)
        if type == 'euclidean':
            loss = paddle.mean(euclidean_dist(center_feat.unsqueeze(0), feat))
            loss_all.append(-loss)
        elif type == 'cosine':
            loss = paddle.mean(cosine_dist(center_feat.unsqueeze(0), feat))
            loss_all.append(-loss)
        elif type == 'cosine_sim':
            loss = paddle.mean(cosine_sim(center_feat.unsqueeze(0), feat))
            loss_all.append(loss)

    loss_all = paddle.mean(paddle.stack(loss_all))

    return loss_all
