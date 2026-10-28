import argparse
import os
import time

import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmdet.apis import set_random_seed
from mmdet3d.models import build_model

from priormaskmap.apis.mmdet_train import custom_train_detector
from priormaskmap.datasets.builder import custom_build_dataset


def main():
    parser = argparse.ArgumentParser(description="Train PriorMaskMap.")
    parser.add_argument("config")
    parser.add_argument("--work-dir")
    parser.add_argument("--resume-from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.resume_from:
        cfg.resume_from = args.resume_from
    distributed = args.launcher == "pytorch"
    if distributed:
        init_dist("pytorch", **cfg.dist_params)
    cfg.gpu_ids = list(range(get_dist_info()[1]))
    cfg.seed = args.seed
    set_random_seed(args.seed)
    mmcv.mkdir_or_exist(cfg.work_dir)
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))
    model = build_model(cfg.model)
    model.init_weights()
    dataset = custom_build_dataset(cfg.data.train)
    model.CLASSES = dataset.CLASSES
    cfg.checkpoint_config.meta = dict(CLASSES=dataset.CLASSES)
    custom_train_detector(
        model,
        [dataset],
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=time.strftime("%Y%m%d_%H%M%S"),
        meta=dict(seed=args.seed, config=cfg.pretty_text),
    )


if __name__ == "__main__":
    main()
