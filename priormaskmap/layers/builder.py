from mmcv.utils import Registry

FUSERS = Registry("fusers")


def build_fuser(cfg):
    return FUSERS.build(cfg)
