import copy
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import torch
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmcv.parallel import DataContainer as DC
import random


@DATASETS.register_module()
class PriorCustomNuScenesDataset(NuScenesDataset):
    def __init__(
        self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size

    def prepare_train_data(self, index):
        data_queue = []
        prev_indexs_list = list(range(index - self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict["frame_idx"]
        scene_token = input_dict["scene_token"]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any()
        ):
            return None
        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if (
                input_dict["frame_idx"] < frame_idx
                and input_dict["scene_token"] == scene_token
            ):
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and (
                    example is None or ~(example["gt_labels_3d"]._data != -1).any()
                ):
                    return None
                frame_idx = input_dict["frame_idx"]
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)

    def union2one(self, queue):
        imgs_list = [each["img"].data for each in queue]
        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each["img_metas"].data
            if i == 0:
                metas_map[i]["prev_bev"] = False
                prev_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                prev_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] = 0
                metas_map[i]["can_bus"][-1] = 0
            else:
                metas_map[i]["prev_bev"] = True
                tmp_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                tmp_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] -= prev_pos
                metas_map[i]["can_bus"][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        queue[-1]["img"] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]["img_metas"] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            prev_idx=info["prev"],
            next_idx=info["next"],
            scene_token=info["scene_token"],
            can_bus=info["can_bus"],
            frame_idx=info["frame_idx"],
            timestamp=info["timestamp"] / 1000000.0,
        )
        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos
        rotation = Quaternion(input_dict["ego2global_rotation"])
        translation = input_dict["ego2global_translation"]
        can_bus = input_dict["can_bus"]
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle
        return input_dict

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
