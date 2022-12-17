# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict

import numpy as np
#import torch
#import torch.nn.functional as F
import paddle
import paddle.nn.functional as F
from tabulate import tabulate

#from .evaluator import DatasetEvaluator
from .rank import evaluate_rank


class ReidEvaluator(object):
    def __init__(self, num_query):
        self._num_query = num_query

        self.features = []
        self.pids = []
        self.camids = []

    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []

    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"].numpy())
        self.camids.extend(inputs["camid"].numpy())
        self.features.append(outputs)

    @staticmethod
    def cal_dist(metric: str, query_feat: paddle.Tensor, gallery_feat: paddle.Tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            query_feat = F.normalize(query_feat, dim=1)
            gallery_feat = F.normalize(gallery_feat, dim=1)
            dist = 1 - paddle.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = paddle.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = paddle.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(1, -2, query_feat, gallery_feat.t())
            dist = dist.clip(min=1e-12).sqrt()  # for numerical stability
        return dist.numpy()

    def evaluate(self, metric='cosine'):
        features = paddle.concat(self.features, dim=0)

        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(self.pids[:self._num_query])
        query_camids = np.asarray(self.camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(self.pids[self._num_query:])
        gallery_camids = np.asarray(self.camids[self._num_query:])

        self._results = OrderedDict()

        dist = self.cal_dist(metric, query_features, gallery_features)

        cmc, all_AP, all_INP = evaluate_rank(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP
        
        #tprs = evaluate_roc(dist, query_pids, gallery_pids, query_camids, gallery_camids)
        #fprs = [1e-4, 1e-3, 1e-2]
        #for i in range(len(fprs)):
        #    self._results["TPR@FPR={}".format(fprs[i])] = tprs[i]

        return copy.deepcopy(self._results)
