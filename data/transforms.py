import paddle 
from paddle import nn
import paddle.vision.transforms as T

class RandomApply(nn.Layer):
    def __init__(self, transforms, p=0.5, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.transforms = transforms
        self.p = p

    def forward(self, img):
        if self.p < paddle.rand(1):
            return img
        for t in self.transforms:
            img = t(img)
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n    p={}'.format(self.p)
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def build_transforms(is_train=True, is_fake=False):
    res = []

    if is_train:
        size_train = (256, 128)     # cfg.INPUT.SIZE_TRAIN

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
        

        res.append(T.Resize(size_train, interpolation=3))
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)])
        if do_cj:
            res.append(RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
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
        res.append(T.Resize(size_test, interpolation=3))
    res.append(T.to_tensor())  # 源码用的fastreid中的ToTensor()
    return T.Compose(res)