import random
import numpy as np
import torch
from priormaskmap.priors.map_change_detector.patch_mixer import MapPatchMixer
from priormaskmap.priors.offline_store import Modmap


class GlobalMap:
    """Read offline raster and vector priors for a batch of sample tokens."""

    def __init__(self, map_cfg):
        self.mod_map = Modmap(map_cfg["map_modification_data_path"])
        self.gt_map = Modmap(map_cfg["load_map_path"])
        self.change_dict = map_cfg["change_type"]["train"]
        self.change_dict_val = map_cfg["change_type"]["val"]
        self.patch_mixer = MapPatchMixer(
            patch_size=map_cfg["patch_size"], mix_ratio=0.5
        )

    def transform_map_channels(self, input_map):
        _, height, width = input_map.shape
        output_map = torch.full(
            (3, height, width), -1.0, dtype=torch.float32, device=input_map.device
        )
        output_map[0][input_map[0] == 1] = 150.0
        output_map[1][input_map[1] == 1] = 150.0
        output_map[2][input_map[2] == 1] = 150.0
        return output_map + 100

    def parse_vector_map(self, vector_map, target_group_number):
        vector_map = vector_map.reshape(-1, 20, 3)
        vector_map[:, :, 0] = (vector_map[:, :, 0] + 15) / 30
        vector_map[:, :, 1] = (vector_map[:, :, 1] + 30) / 60
        if vector_map.shape[0] < target_group_number:
            vector_map = np.concatenate(
                [
                    vector_map,
                    np.zeros((target_group_number - vector_map.shape[0], 20, 3)),
                ],
                axis=0,
            )
        else:
            vector_map = vector_map[:target_group_number, :, :]
        return vector_map

    def get_patch_mix_data(self, batch_data, device, state):
        original_maps = []
        modified_maps = []
        vector_maps_gt = []
        vector_maps_modified = []
        choices = self.change_dict if state == "train" else self.change_dict_val
        for sample_token in batch_data["sample_idx"]:
            original_map, _, original_vectors = self.gt_map.get(
                sample_token,
                ["geo_deformer", "no_change", "ED_0_level_low"],
                return_vec=True,
            )
            choice = random.choice(choices)
            modified_map, _, modified_vectors = self.mod_map.get(
                sample_token, choice, return_vec=True
            )
            original_map = self.transform_map_channels(original_map)
            modified_map = self.transform_map_channels(modified_map)
            if torch.max(torch.unique(original_map)) > 1:
                original_map = original_map.float() / 255.0
            if torch.max(torch.unique(modified_map)) > 1:
                modified_map = modified_map.float() / 255.0
            original_maps.append(original_map)
            modified_maps.append(modified_map)
            vector_maps_gt.append(self.parse_vector_map(original_vectors, 35))
            vector_maps_modified.append(self.parse_vector_map(modified_vectors, 35))
        original_maps = torch.stack(original_maps).to(device)
        modified_maps = torch.stack(modified_maps).to(device)
        map_available = torch.ones(len(original_maps), dtype=torch.int64).to(device)
        vector_maps_gt = torch.tensor(vector_maps_gt).to(torch.float32).to(device)
        vector_maps_modified = (
            torch.tensor(vector_maps_modified).to(torch.float32).to(device)
        )
        mixed_maps, grid_labels, pixel_labels = self.patch_mixer.generate_batch_mix(
            original_maps, modified_maps
        )
        return (
            mixed_maps,
            grid_labels,
            pixel_labels,
            original_maps,
            modified_maps,
            map_available,
            vector_maps_gt,
            vector_maps_modified,
        )
