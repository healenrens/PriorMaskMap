import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class PriorPadMultiViewImage(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
        results["ori_shape"] = [img.shape for img in results["img"]]
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class PriorPadMultiViewImageDepth(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
            padded_gt_depth = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["gt_depth"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
            padded_gt_depth = [
                mmcv.impad_to_multiple(
                    img.numpy(), self.size_divisor, pad_val=self.pad_val
                )
                for img in results["gt_depth"]
            ]
        results["ori_shape"] = [img.shape for img in results["img"]]
        results["img"] = padded_img
        results["gt_depth"] = np.stack(padded_gt_depth)
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class PriorNormalizeMultiviewImage(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class PriorCustomCollect3D(object):
    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "prev_idx",
            "next_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "scene_token",
            "camera_intrinsics",
            "can_bus",
            "lidar2global",
            "cam2lidar",
            "lidar2cam",
            "camera2ego",
            "cam_intrinsic",
            "img_aug_matrix",
            "lidar2ego",
            "lidar_aug_matrix",
            "timestamp",
            "img_inputs",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_depth",
            "map_location",
            "ego2global_translation",
            "ego2global_rotation",
            "index",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data["img_metas"] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(keys={self.keys}, meta_keys={self.meta_keys})"
        )


@PIPELINES.register_module()
class PriorRandomScaleImageMultiViewImage(object):
    def __init__(self, scales=[]):
        self.scales = scales

    def __call__(self, results):
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]
        y_size = [int(img.shape[0] * rand_scale) for img in results["img"]]
        x_size = [int(img.shape[1] * rand_scale) for img in results["img"]]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results["img"] = [
            mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False)
            for idx, img in enumerate(results["img"])
        ]
        lidar2img = [scale_factor @ l2i for l2i in results["lidar2img"]]
        img_aug_matrix = [scale_factor for _ in results["lidar2img"]]
        results["lidar2img"] = lidar2img
        results["img_aug_matrix"] = img_aug_matrix
        results["img_shape"] = [img.shape for img in results["img"]]
        results["ori_shape"] = [img.shape for img in results["img"]]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.scales}, "
        return repr_str
