import paddle 
from paddle import nn
import paddle.vision.transforms as T
import random
import numpy as np
from paddle.vision.transforms import BaseTransform


def build_transforms(is_train=True, is_fake=False):
    res = []

    if is_train:
        # horizontal filp
        do_flip = True  # cfg.INPUT.DO_FLIP
        flip_prob = 0.5 # cfg.INPUT.FLIP_PROB

        # padding
        do_pad = True   # cfg.INPUT.DO_PAD
        padding = 10    # cfg.INPUT.PADDING
        padding_mode = 'constant'   # cfg.INPUT.PADDING_MODE

        # color jitter
        do_cj = True    # cfg.INPUT.CJ.ENABLED
        cj_prob = 1.0   # cfg.INPUT.CJ.PROB
        cj_brightness = 0.15    # cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = 0.15      # cfg.INPUT.CJ.CONTRAST
        cj_saturation = 0.1     # cfg.INPUT.CJ.SATURATION
        cj_hue = 0.1    # cfg.INPUT.CJ.HUE

        """
        # augmix augmentation
        do_augmix = False   # cfg.INPUT.DO_AUGMIX

        # auto augmentation
        do_autoaug = False  # cfg.INPUT.DO_AUTOAUG
        total_iter = 120 # cfg.SOLVER.MAX_ITER

        # random erasing
        do_rea = False  # cfg.INPUT.REA.ENABLED
        rea_prob = 0.5  # cfg.INPUT.REA.PROB
        rea_mean = [123.675, 116.28, 103.53]      # cfg.INPUT.REA.MEAN

        # random patch
        do_rpt = False  # cfg.INPUT.RPT.ENABLED
        rpt_prob = False    # cfg.INPUT.RPT.PROB


        if do_augmix:
            res.append(AugMix())
        if do_autoaug:
            res.append(AutoAugment(total_iter))
        if do_rea:
            res.append(RandomErasing(probability=rea_prob, mean=rea_mean))
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))

        """
        
        size_train = (256, 128)     # cfg.INPUT.SIZE_TRAIN
        #res.append(T.Resize(size_train, interpolation=3))

        res.append(T.Resize(size_train, interpolation='bicubic'))
        if do_flip:
            res.append(T.RandomHorizontalFlip(prob=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_cj:
            res.append(T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue))
        """
        if is_fake:
            if cfg.META.DATA.SYNTH_FLAG == 'jitter':
                res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=1.0))
            elif cfg.META.DATA.SYNTH_FLAG == 'augmix':
                res.append(AugMix())
            elif cfg.META.DATA.SYNTH_FLAG == 'both':
                res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
                res.append(AugMix())
        """
    else:
        size_test = (256, 128) 
        res.append(T.Resize(size_test, interpolation='bicubic'))
    res.append(ToTensor())  # 源码用的fastreid中的ToTensor()
    return T.Compose(res)


class ToTensor(BaseTransform):
    def __init__(self, data_format='CHW', keys=None):
        super(ToTensor, self).__init__(keys)
        self.data_format = data_format

    def _apply_image(self, img):
        """
        Args:
            img (PIL.Image|np.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return to_tensor(img, self.data_format)


def to_tensor(pic, data_format='CHW'):
    """Converts a ``PIL.Image`` to paddle.Tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL.Image): Image to be converted to tensor.
        data_format (str, optional): Data format of output tensor, should be 'HWC' or 
            'CHW'. Default: 'CHW'.

    Returns:
        Tensor: Converted image.

    """

    if data_format not in ['CHW', 'HWC']:
        raise ValueError(
            'data_format should be CHW or HWC. Got {}'.format(data_format))

    # PIL Image
    if pic.mode == 'I':
        img = paddle.to_tensor(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        # cast and reshape not support int16
        img = paddle.to_tensor(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'F':
        img = paddle.to_tensor(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * paddle.to_tensor(np.array(pic, np.uint8, copy=False))
    else:
        img = paddle.to_tensor(np.array(pic, copy=False))

    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    dtype = paddle.fluid.data_feeder.convert_dtype(img.dtype)
    if dtype == 'uint8':
        img = paddle.cast(img, np.float32)

    img = img.reshape([pic.size[1], pic.size[0], nchannel])

    if data_format == 'CHW':
        img = img.transpose([2, 0, 1])

    return img