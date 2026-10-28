import torch
import random

BLANK_THRESHOLD = 100.0 / 255.0
ORIGINAL_PATCH_LABEL = 0
MODIFIED_PATCH_LABEL = 1
BLANK_PIXEL_LABEL = 2
BLANK_BLANK_PATCH_MARKER = 3
PADDING_PATCH_MARKER = 4


class MapPatchMixer:
    def __init__(
        self, patch_size=32, patch_stride=None, mix_ratio=0.3, min_patch_size=16
    ):
        self.patch_size = max(1, patch_size)
        self.mix_ratio = mix_ratio

    def split_map_to_patches(self, map_tensor):
        c, h, w = map_tensor.shape
        padded_h = (h + self.patch_size - 1) // self.patch_size * self.patch_size
        padded_w = (w + self.patch_size - 1) // self.patch_size * self.patch_size
        padded_map = torch.zeros(
            (c, padded_h, padded_w), dtype=map_tensor.dtype, device=map_tensor.device
        )
        padded_map[:, :h, :w] = map_tensor
        padding_mask = torch.zeros(
            (padded_h, padded_w), dtype=torch.bool, device=map_tensor.device
        )
        if padded_h > h or padded_w > w:
            if padded_h > h:
                padding_mask[h:, :] = True
            if padded_w > w:
                padding_mask[:, w:] = True
        grid_h = padded_h // self.patch_size
        grid_w = padded_w // self.patch_size
        patches = []
        positions = []
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * self.patch_size
                x_start = j * self.patch_size
                y_end = y_start + self.patch_size
                x_end = x_start + self.patch_size
                patch = padded_map[:, y_start:y_end, x_start:x_end]
                patches.append(patch)
                positions.append((x_start, y_start))
        return (patches, positions, padded_map, padding_mask)

    def merge_patches_to_map(
        self, patches, positions, original_map_shape, padding_mask
    ):
        c, orig_h, orig_w = original_map_shape
        padded_h = (orig_h + self.patch_size - 1) // self.patch_size * self.patch_size
        padded_w = (orig_w + self.patch_size - 1) // self.patch_size * self.patch_size
        padded_merged = torch.zeros(
            (c, padded_h, padded_w), dtype=patches[0].dtype, device=patches[0].device
        )
        for patch, (x, y) in zip(patches, positions):
            y_end = y + self.patch_size
            x_end = x + self.patch_size
            padded_merged[:, y:y_end, x:x_end] = patch
        merged_map = padded_merged[:, :orig_h, :orig_w]
        return merged_map

    def create_patch_mix(self, original_map, modified_map, change_mask=None):
        c, h, w = original_map.shape
        if original_map.shape != modified_map.shape:
            max_h = max(original_map.shape[1], modified_map.shape[1])
            max_w = max(original_map.shape[2], modified_map.shape[2])
            if original_map.shape[1] < max_h or original_map.shape[2] < max_w:
                resized_orig = torch.zeros(
                    (c, max_h, max_w),
                    dtype=original_map.dtype,
                    device=original_map.device,
                )
                resized_orig[:, : original_map.shape[1], : original_map.shape[2]] = (
                    original_map
                )
                original_map = resized_orig
            if modified_map.shape[1] < max_h or modified_map.shape[2] < max_w:
                resized_mod = torch.zeros(
                    (c, max_h, max_w),
                    dtype=modified_map.dtype,
                    device=modified_map.device,
                )
                resized_mod[:, : modified_map.shape[1], : modified_map.shape[2]] = (
                    modified_map
                )
                modified_map = resized_mod
            h, w = (max_h, max_w)
        orig_patches, positions, padded_orig_map, padding_mask = (
            self.split_map_to_patches(original_map)
        )
        mod_patches, _, padded_mod_map, _ = self.split_map_to_patches(modified_map)
        num_patches = len(orig_patches)
        if num_patches == 1:
            if random.random() < self.mix_ratio:
                num_mixed_patches = 1
            else:
                num_mixed_patches = 0
        else:
            num_mixed_patches = int(num_patches * self.mix_ratio)
        mixed_indices_set = (
            set(random.sample(range(num_patches), num_mixed_patches))
            if num_mixed_patches > 0
            else set()
        )
        mixed_patches_list = [None] * num_patches
        target_grid_labels_flat = torch.full(
            (num_patches,), fill_value=-1, dtype=torch.long, device=original_map.device
        )
        target_pixel_labels = torch.zeros(
            (h, w), dtype=torch.long, device=original_map.device
        )
        padded_h, padded_w = (padded_orig_map.shape[1], padded_orig_map.shape[2])
        grid_h = padded_h // self.patch_size
        grid_w = padded_w // self.patch_size
        for patch_idx in range(num_patches):
            orig_p = orig_patches[patch_idx]
            mod_p = mod_patches[patch_idx]
            grid_i = patch_idx // grid_w
            grid_j = patch_idx % grid_w
            y_start = grid_i * self.patch_size
            x_start = grid_j * self.patch_size
            y_end = y_start + self.patch_size
            x_end = x_start + self.patch_size
            patch_padding_mask = padding_mask[y_start:y_end, x_start:x_end]
            has_padding = torch.any(patch_padding_mask)
            if has_padding:
                mixed_patches_list[patch_idx] = orig_p
                target_grid_labels_flat[patch_idx] = PADDING_PATCH_MARKER
                current_label = PADDING_PATCH_MARKER
            else:
                threshold_val = torch.tensor(
                    BLANK_THRESHOLD, device=orig_p.device, dtype=orig_p.dtype
                )
                if orig_p.dtype == torch.uint8:
                    threshold_val = torch.tensor(
                        100, device=orig_p.device, dtype=orig_p.dtype
                    )
                is_orig_blank = torch.all(orig_p <= threshold_val)
                is_mod_blank = torch.all(mod_p <= threshold_val)
                if is_orig_blank and is_mod_blank:
                    mixed_patches_list[patch_idx] = orig_p
                    target_grid_labels_flat[patch_idx] = BLANK_BLANK_PATCH_MARKER
                    current_label = BLANK_BLANK_PATCH_MARKER
                elif patch_idx in mixed_indices_set:
                    mixed_patches_list[patch_idx] = mod_p
                    target_grid_labels_flat[patch_idx] = MODIFIED_PATCH_LABEL
                    current_label = MODIFIED_PATCH_LABEL
                else:
                    mixed_patches_list[patch_idx] = orig_p
                    target_grid_labels_flat[patch_idx] = ORIGINAL_PATCH_LABEL
                    current_label = ORIGINAL_PATCH_LABEL
            pixel_y_start = grid_i * self.patch_size
            pixel_x_start = grid_j * self.patch_size
            pixel_y_end = min(pixel_y_start + self.patch_size, h)
            pixel_x_end = min(pixel_x_start + self.patch_size, w)
            if has_padding:
                target_pixel_labels[
                    pixel_y_start:pixel_y_end, pixel_x_start:pixel_x_end
                ] = PADDING_PATCH_MARKER
            elif is_orig_blank and is_mod_blank:
                target_pixel_labels[
                    pixel_y_start:pixel_y_end, pixel_x_start:pixel_x_end
                ] = BLANK_BLANK_PATCH_MARKER
            else:
                actual_patch_h = pixel_y_end - pixel_y_start
                actual_patch_w = pixel_x_end - pixel_x_start
                orig_patch_region = orig_p[:, :actual_patch_h, :actual_patch_w]
                mod_patch_region = mod_p[:, :actual_patch_h, :actual_patch_w]
                orig_pixel_is_blank = torch.all(
                    orig_patch_region <= threshold_val, dim=0
                )
                mod_pixel_is_blank = torch.all(mod_patch_region <= threshold_val, dim=0)
                both_blank = orig_pixel_is_blank & mod_pixel_is_blank
                region_labels = torch.full(
                    (actual_patch_h, actual_patch_w),
                    current_label,
                    dtype=torch.long,
                    device=original_map.device,
                )
                region_labels[both_blank] = BLANK_PIXEL_LABEL
                target_pixel_labels[
                    pixel_y_start:pixel_y_end, pixel_x_start:pixel_x_end
                ] = region_labels
        mixed_map = self.merge_patches_to_map(
            mixed_patches_list, positions, (c, h, w), padding_mask
        )
        return (mixed_map, target_grid_labels_flat, target_pixel_labels)

    def generate_batch_mix(self, batch_orig_maps, batch_mod_maps, change_masks=None):
        batch_size = batch_orig_maps.shape[0]
        mixed_maps_list = []
        target_grid_labels_list = []
        target_pixel_labels_list = []
        shapes_match = True
        base_shape = batch_orig_maps[0].shape
        for i in range(batch_size):
            if (
                batch_orig_maps[i].shape != base_shape
                or batch_mod_maps[i].shape != base_shape
            ):
                shapes_match = False
                break
        for i in range(batch_size):
            mixed_map, target_grid_labels, target_pixel_labels = self.create_patch_mix(
                batch_orig_maps[i], batch_mod_maps[i]
            )
            mixed_maps_list.append(mixed_map)
            target_grid_labels_list.append(target_grid_labels)
            target_pixel_labels_list.append(target_pixel_labels)
        if not shapes_match:
            max_h = max([m.shape[1] for m in mixed_maps_list])
            max_w = max([m.shape[2] for m in mixed_maps_list])
            final_mixed_maps = []
            final_pixel_labels = []
            for mixed_map, pixel_labels in zip(
                mixed_maps_list, target_pixel_labels_list
            ):
                if mixed_map.shape[1] < max_h or mixed_map.shape[2] < max_w:
                    c = mixed_map.shape[0]
                    padded_map = torch.zeros(
                        (c, max_h, max_w),
                        device=mixed_map.device,
                        dtype=mixed_map.dtype,
                    )
                    padded_map[:, : mixed_map.shape[1], : mixed_map.shape[2]] = (
                        mixed_map
                    )
                    final_mixed_maps.append(padded_map)
                    padded_pixel_labels = torch.zeros(
                        (max_h, max_w),
                        device=pixel_labels.device,
                        dtype=pixel_labels.dtype,
                    )
                    padded_pixel_labels[
                        : pixel_labels.shape[0], : pixel_labels.shape[1]
                    ] = pixel_labels
                    final_pixel_labels.append(padded_pixel_labels)
                else:
                    final_mixed_maps.append(mixed_map)
                    final_pixel_labels.append(pixel_labels)
            stacked_mixed_maps = torch.stack(final_mixed_maps)
            stacked_pixel_labels = torch.stack(final_pixel_labels)
        else:
            stacked_mixed_maps = torch.stack(mixed_maps_list)
            stacked_pixel_labels = torch.stack(target_pixel_labels_list)
        stacked_target_labels = torch.stack(target_grid_labels_list)
        return (stacked_mixed_maps, stacked_target_labels, stacked_pixel_labels)
