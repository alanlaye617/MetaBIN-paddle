import sys
sys.path.append('.')
from train import Trainer
from utils.build_ref import build_ref_trainer
from reprod_log import ReprodLogger, ReprodDiffHelper
import paddle
import torch


reprod_log_ref = ReprodLogger()
reprod_log_pad = ReprodLogger()
cls_heads_ref = build_ref_trainer(16).model.heads
reprod_log_ref.add('cls_w_mean', torch.mean(cls_heads_ref.classifier_fc.weight).cpu().detach().numpy())
reprod_log_ref.add('cls_w_std', torch.std(cls_heads_ref.classifier_fc.weight).cpu().detach().numpy())
cls_heads_pad = Trainer(16).model.heads
reprod_log_pad.add('cls_w_mean', paddle.mean(cls_heads_pad.classifier_fc.weight).cpu().detach().numpy())
reprod_log_pad.add('cls_w_std', paddle.std(cls_heads_pad.classifier_fc.weight).cpu().detach().numpy())

reprod_log_ref.save('./result/init_ref.npy')
reprod_log_pad.save('./result/init_paddle.npy')

diff_helper = ReprodDiffHelper()

info1 = diff_helper.load_info("./result/init_paddle.npy")
info2 = diff_helper.load_info("./result/init_ref.npy")

diff_helper.compare_info(info1, info2)

diff_helper.report(
    diff_method="mean", diff_threshold=1e-3, path="./result/log/init_diff.log")

print()

