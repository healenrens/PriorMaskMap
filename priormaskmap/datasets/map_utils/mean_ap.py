# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing import Pool
from shapely.geometry import LineString
import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
from os import path as osp
from functools import partial
from priormaskmap.datasets.map_utils.tpfp import custom_tpfp_gen


def average_precision(recalls, precisions, mode="area"):
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == "area":
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum((mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == "11points":
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 0.001, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
        ap /= 11
    if no_scale:
        ap = ap[0]
    return ap


def get_cls_results(
    gen_results,
    annotations,
    num_sample=100,
    num_pred_pts_per_instance=30,
    eval_use_same_gt_sample_num_flag=False,
    class_id=0,
    fix_interval=False,
    code_size=2,
):
    cls_gens, cls_scores = ([], [])
    for res in gen_results["vectors"]:
        if res["type"] == class_id:
            if len(res["pts"]) < 2:
                continue
            if not eval_use_same_gt_sample_num_flag:
                sampled_points = np.array(res["pts"])
            else:
                line = res["pts"]
                line = LineString(line)
                if fix_interval:
                    distances = list(np.arange(1.0, line.length, 1.0))
                    distances = [0] + distances + [line.length]
                    sampled_points = np.array(
                        [
                            list(line.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, code_size)
                else:
                    line_pts = np.array(line.coords)
                    if line_pts.shape[0] == num_sample:
                        sampled_points = line_pts
                    else:
                        distances = np.linspace(0, line.length, num_sample)
                        sampled_points = np.array(
                            [
                                list(line.interpolate(distance).coords)
                                for distance in distances
                            ]
                        ).reshape(-1, code_size)
            cls_gens.append(sampled_points)
            cls_scores.append(res["confidence_level"])
    num_res = len(cls_gens)
    if num_res > 0:
        cls_gens = np.stack(cls_gens).reshape(num_res, -1)
        cls_scores = np.array(cls_scores)[:, np.newaxis]
        cls_gens = np.concatenate([cls_gens, cls_scores], axis=-1)
    elif not eval_use_same_gt_sample_num_flag:
        cls_gens = np.zeros((0, num_pred_pts_per_instance * code_size + 1))
    else:
        cls_gens = np.zeros((0, num_sample * code_size + 1))
    cls_gts = []
    for ann in annotations["vectors"]:
        if ann["type"] == class_id:
            line = ann["pts"]
            line = LineString(line)
            distances = np.linspace(0, line.length, num_sample)
            sampled_points = np.array(
                [list(line.interpolate(distance).coords) for distance in distances]
            ).reshape(-1, code_size)
            cls_gts.append(sampled_points)
    num_gts = len(cls_gts)
    if num_gts > 0:
        cls_gts = np.stack(cls_gts).reshape(num_gts, -1)
    else:
        cls_gts = np.zeros((0, num_sample * code_size))
    return (cls_gens, cls_gts)


def format_res_gt_by_classes(
    result_path,
    gen_results,
    annotations,
    cls_names=None,
    num_pred_pts_per_instance=30,
    eval_use_same_gt_sample_num_flag=False,
    pc_range=[-15.0, -30.0, -5.0, 15.0, 30.0, 3.0],
    code_size=2,
    nproc=24,
):
    timer = mmcv.Timer()
    num_fixed_sample_pts = 100
    fix_interval = False
    output_dir = osp.join(*osp.split(result_path)[:-1])
    pool = Pool(nproc)
    cls_gens, cls_gts = ({}, {})
    formatting_file = "cls_formatted.pkl"
    formatting_file = osp.join(output_dir, formatting_file)
    if False:
        from PIL import Image
        import matplotlib.pyplot as plt

        show_dir = osp.join(output_dir, "vis_json")
        mmcv.mkdir_or_exist(osp.abspath(show_dir))
        car_img = Image.open("./figs/lidar_car.png")
        colors_plt = ["r", "b", "g"]
        for i in range(20):
            plt.figure(figsize=(2, 4))
            plt.xlim(pc_range[0], pc_range[3])
            plt.ylim(pc_range[1], pc_range[4])
            plt.axis("off")
            for line in gen_results[i]["vectors"]:
                l = np.array(line["pts"])
                plt.plot(l[:, 0], l[:, 1], "-", color="red")
            for line in annotations[i]["vectors"]:
                l = np.array(line["pts"])
                plt.plot(l[:, 0], l[:, 1], "-", color="blue")
            plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
            map_path = osp.join(show_dir, "COMPARE_MAP_{}.jpg".format(i))
            plt.savefig(map_path, bbox_inches="tight", dpi=400)
            plt.close()
    for i, clsname in enumerate(cls_names):
        gengts = pool.starmap(
            partial(
                get_cls_results,
                num_sample=num_fixed_sample_pts,
                num_pred_pts_per_instance=num_pred_pts_per_instance,
                eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
                class_id=i,
                fix_interval=fix_interval,
                code_size=code_size,
            ),
            zip(gen_results, annotations),
        )
        gens, gts = tuple(zip(*gengts))
        cls_gens[clsname] = gens
        cls_gts[clsname] = gts
    mmcv.dump([cls_gens, cls_gts], formatting_file)
    pool.close()
    return (cls_gens, cls_gts)


def eval_map(
    gen_results,
    annotations,
    cls_gens,
    cls_gts,
    threshold=0.5,
    cls_names=None,
    logger=None,
    tpfp_fn=None,
    pc_range=[-15.0, -30.0, -5.0, 15.0, 30.0, 3.0],
    metric=None,
    num_pred_pts_per_instance=30,
    code_size=2,
    nproc=24,
):
    timer = mmcv.Timer()
    pool = Pool(nproc)
    eval_results = []
    for i, clsname in enumerate(cls_names):
        cls_gen = cls_gens[clsname]
        cls_gt = cls_gts[clsname]
        tpfp_fn = custom_tpfp_gen
        tpfp_fn = partial(
            tpfp_fn, threshold=threshold, metric=metric, code_size=code_size
        )
        args = []
        tpfp = pool.starmap(tpfp_fn, zip(cls_gen, cls_gt, *args))
        tp, fp = tuple(zip(*tpfp))
        num_gts = 0
        for j, bbox in enumerate(cls_gt):
            num_gts += bbox.shape[0]
        cls_gen = np.vstack(cls_gen)
        num_dets = cls_gen.shape[0]
        sort_inds = np.argsort(-cls_gen[:, -1])
        tp = np.hstack(tp)[sort_inds]
        fp = np.hstack(fp)[sort_inds]
        tp = np.cumsum(tp, axis=0)
        fp = np.cumsum(fp, axis=0)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum(tp + fp, eps)
        mode = "area"
        ap = average_precision(recalls, precisions, mode)
        eval_results.append(
            {
                "num_gts": num_gts,
                "num_dets": num_dets,
                "recall": recalls,
                "precision": precisions,
                "ap": ap,
            }
        )
    pool.close()
    aps = []
    for cls_result in eval_results:
        if cls_result["num_gts"] > 0:
            aps.append(cls_result["ap"])
    mean_ap = np.array(aps).mean().item() if len(aps) else 0.0
    print_map_summary(mean_ap, eval_results, class_name=cls_names, logger=logger)
    return (mean_ap, eval_results)


def print_map_summary(
    mean_ap, results, class_name=None, scale_ranges=None, logger=None
):
    if logger == "silent":
        return
    if isinstance(results[0]["ap"], np.ndarray):
        num_scales = len(results[0]["ap"])
    else:
        num_scales = 1
    num_classes = len(results)
    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result["recall"].size > 0:
            recalls[:, i] = np.array(cls_result["recall"], ndmin=2)[:, -1]
        aps[:, i] = cls_result["ap"]
        num_gts[:, i] = cls_result["num_gts"]
    label_names = class_name
    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]
    header = ["class", "gts", "dets", "recall", "ap"]
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f"Scale range {scale_ranges[i]}", logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j],
                num_gts[i, j],
                results[j]["num_dets"],
                f"{recalls[i, j]:.3f}",
                f"{aps[i, j]:.3f}",
            ]
            table_data.append(row_data)
        table_data.append(["mAP", "", "", "", f"{mean_ap[i]:.3f}"])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log("\n" + table.table, logger=logger)
