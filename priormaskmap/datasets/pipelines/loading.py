import numpy as np
from mmdet.datasets.builder import PIPELINES
import torch
from pyquaternion import Quaternion


@PIPELINES.register_module()
class PriorCustomPointToMultiViewDepth(object):
    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = (height // self.downsample, width // self.downsample)
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (
            (coor[:, 0] >= 0)
            & (coor[:, 0] < width)
            & (coor[:, 1] >= 0)
            & (coor[:, 1] < height)
            & (depth < self.grid_config["depth"][1])
            & (depth >= self.grid_config["depth"][0])
        )
        coor, depth = (coor[kept1], depth[kept1])
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.0).argsort()
        coor, depth, ranks = (coor[sort], depth[sort], ranks[sort])
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = ranks[1:] != ranks[:-1]
        coor, depth = (coor[kept2], depth[kept2])
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results["points"]
        imgs = np.stack(results["img"])
        img_aug_matrix = results["img_aug_matrix"]
        post_rots = [
            torch.tensor(single_aug_matrix[:3, :3]).to(torch.float)
            for single_aug_matrix in img_aug_matrix
        ]
        post_trans = torch.stack(
            [
                torch.tensor(single_aug_matrix[:3, 3]).to(torch.float)
                for single_aug_matrix in img_aug_matrix
            ]
        )
        intrins = results["camera_intrinsics"]
        depth_map_list = []
        for cid in range(len(imgs)):
            lidar2lidarego = torch.tensor(results["lidar2ego"]).to(torch.float32)
            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results["ego2global_rotation"]
            ).rotation_matrix
            lidarego2global[:3, 3] = results["ego2global_translation"]
            lidarego2global = torch.from_numpy(lidarego2global)
            cam2camego = torch.tensor(results["camera2ego"][cid])
            camego2global = results["camego2global"][cid]
            cam2img = torch.tensor(intrins[cid]).to(torch.float32)
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego)
            )
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T.to(torch.float)
            ) + lidar2img[:3, 3].to(torch.float).unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1
            )
            points_img = (
                points_img.matmul(post_rots[cid].T) + post_trans[cid : cid + 1, :]
            )
            depth_map = self.points2depthmap(points_img, imgs.shape[1], imgs.shape[2])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results["gt_depth"] = depth_map
        return results
