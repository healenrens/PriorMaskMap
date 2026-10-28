import argparse
import os
from pathlib import Path

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet.apis import set_random_seed
from mmdet3d.models import build_model

from priormaskmap.apis.test import custom_multi_gpu_test
from priormaskmap.datasets.builder import build_dataloader, custom_build_dataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate PriorMaskMap.")
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--work-dir", default="work_dirs/evaluation")
    parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--launcher", choices=["none", "pytorch"], default="none")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args()
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    distributed = args.launcher == "pytorch"
    if distributed:
        init_dist("pytorch", **cfg.dist_params)
    set_random_seed(args.seed)
    dataset = custom_build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    model = build_model(cfg.model)
    if cfg.get("fp16"):
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        outputs = custom_multi_gpu_test(model, loader)
    else:
        model = MMDataParallel(model.cuda(), device_ids=[0])
        model.eval()
        outputs = []
        progress = mmcv.ProgressBar(len(dataset))
        with torch.no_grad():
            for batch in loader:
                result = model(return_loss=False, rescale=True, **batch)
                outputs.extend(result)
                progress.update(len(result))
    rank, _ = get_dist_info()
    if rank == 0:
        output_dir = Path(args.work_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.out:
            mmcv.dump(outputs, args.out)
        metrics = dataset.evaluate(
            outputs, metric="chamfer", jsonfile_prefix=str(output_dir / "predictions")
        )
        mmcv.dump(metrics, str(output_dir / "metrics.json"))
        print(metrics)


if __name__ == "__main__":
    main()
