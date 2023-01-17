# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import paddle
import paddle.nn.functional as F


def cross_entropy_loss(pred_class_logits, gt_classes, eps, alpha=0.2):
    num_classes = pred_class_logits.shape[1]

    if eps >= 0:
        smooth_param = eps
    else:
        # Adaptive label smooth regularization
        soft_label = F.softmax(pred_class_logits, axis=1)
        smooth_param = alpha * soft_label[paddle.arange(soft_label.shape[0]), gt_classes].unsqueeze(1)
    log_probs = F.log_softmax(pred_class_logits, axis=1)
    with paddle.no_grad():
        targets = F.one_hot(gt_classes.cast(paddle.int64), num_classes=num_classes)
        targets = F.label_smooth(targets, epsilon=num_classes/(num_classes-1) * smooth_param)
        #targets = paddle.ones_like(log_probs)
        #targets *= smooth_param / (num_classes - 1)
        #for i, idx in enumerate(gt_classes):
        #    targets[i, idx] = 1 - smooth_param
        
    loss = (-targets * log_probs).sum(axis=1)

    """
    # confidence penalty
    conf_penalty = 0.3
    probs = F.softmax(pred_class_logits, dim=1)
    entropy = torch.sum(-probs * log_probs, dim=1)
    loss = torch.clamp_min(loss - conf_penalty * entropy, min=0.)
    """

    with paddle.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).shape[0], 1)

    loss = loss.sum() / non_zero_cnt

    return loss
