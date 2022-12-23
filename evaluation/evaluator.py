# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict
import time
import datetime
import numpy as np
#import torch
#import torch.nn.functional as F
import paddle
import paddle.nn.functional as F
from tabulate import tabulate
from contextlib import contextmanager
#from .evaluator import DatasetEvaluator
from .rank import evaluate_rank
from tqdm import tqdm
#from utils import translate_inputs_t2p


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
            query_feat = F.normalize(query_feat, axis=1)
            gallery_feat = F.normalize(gallery_feat, axis=1)
            dist = 1 - paddle.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = paddle.pow(query_feat, 2).sum(1, keepdim=True).expand([m, n])
            yy = paddle.pow(gallery_feat, 2).sum(1, keepdim=True).expand([n, m]).t()
            dist = xx + yy
            dist.addmm_(1, -2, query_feat, gallery_feat.t())
            dist = dist.clip(min=1e-12).sqrt()  # for numerical stability
        return dist.numpy()

    def evaluate(self, metric='cosine'):
        features = paddle.concat(self.features, axis=0)

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


def inference_on_dataset(model, data_loader, evaluator, opt=None):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.
    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    logger = logging.getLogger(__name__)
    if opt == None:
        logger.info("Start inference on {} images".format(len(data_loader.dataset)))

    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), paddle.no_grad():
        for idx, inputs in tqdm(enumerate(data_loader)):
          #  inputs = translate_inputs_t2p(inputs, is_train=False)
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)

            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            idx += 1
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_batch = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_batch > 30:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
            #    log_every_n_seconds(
            #        logging.INFO,
            #        "Inference done {}/{}. {:.4f} s / batch. ETA={}".format(
            #            idx + 1, total, seconds_per_batch, str(eta)
           #         ),
           #         n=30,
            #    )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    # NOTE this format is parsed by grep

    if opt == None:
        logger.info(
            "Total inference time: {} ({:.6f} s / batch per device)".format(
                total_time_str, total_time / (total - num_warmup)
            )
        )
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / batch per device)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup)
            )
        )
    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    #training_mode = model.training
    model.eval()
    yield
    model.train()
