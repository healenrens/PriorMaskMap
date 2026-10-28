# Generating Biased Map Priors

## Framework and Paper

We follow [Syn-D-maps](https://github.com/healenrens/Syn-D-maps), the map-transformation framework introduced in:

> Haoming Xu, Yiyang Xiao, Wei Li, and Yu Hu. **Generating Synthetic Deviation Maps for Prior-Enhanced Vectorized HD Map Construction.** 2025 IEEE Intelligent Vehicles Symposium (IV), pp. 419-426, 2025. DOI: [10.1109/IV64158.2025.11097394](https://doi.org/10.1109/IV64158.2025.11097394).

The toolkit supplies geometric, topological, and semantic map transformations. PriorMaskMap uses its topology-changing operations to construct biased priors. The coordinate conversion, offline raster/vector export, fixed mixed-prior stores, and training-time patch mixing described below form the input preparation for our model.

## Main Data Setting

| Item | Setting |
|---|---|
| Base dataset | nuScenes `v1.0-trainval` |
| Map classes | `divider`, `ped_crossing`, `boundary` |
| Prior modifications | lane addition, reduction, widening, narrowing |
| Requested modification ratio | `0.6` |
| Model map range | `x: [-15, 15] m`, `y: [-30, 30] m` |
| Raster size | `[3, 200, 100]` |
| Vector sampling | 20 points per instance |
| Model vector input | pad or truncate to 35 instances |
| Training raster patches | `10 x 10` pixels |
| Training patch-mix ratio | `0.5` |

The modification ratio is the transformation's requested lane-group budget. For example, the public lane-addition and lane-deletion implementations derive the budget from `int(max_changes * number_of_lane_groups)`. Their returned `change_rate` records the realized affected-instance fraction. Thus, `0.6` is not a fixed proportion of corrupted image pixels, a geometric displacement, or a probability of choosing a clean map during training.

The model's clean mapping targets remain unchanged. Only the prior-map copy receives the synthetic modification.

## 1. Extract Local Map Geometry

For each nuScenes frame token:

1. Read its scene, map location, and ego pose through the nuScenes SDK.
2. Extract the local lane geometry, lane and road dividers, road boundaries, and pedestrian crossings.
3. Transform the geometry from global coordinates into an ego-aligned local patch.
4. Retain an unmodified copy and generate modified copies from the same source geometry.

The toolkit's [example.py](https://github.com/healenrens/Syn-D-maps/blob/main/example.py) demonstrates map extraction through `NuScenesMap`, `NuScenesMapExplorer`, `get_map_geom`, and `new_get_boundary`. Its mini-dataset scene, token, and patch size are demonstration settings; the model uses the trainval frames and map range specified above.

## 2. Apply Topological Modifications

| Prior type | Public toolkit operation | Legacy export label |
|---|---|---|
| Lane addition | `top_utils_lib.add_lane.add_lane` | `add_road` |
| Lane reduction | `top_utils_lib.delete_lane.delete_lane` | `partial_closure` |
| Lane widening | `top_utils_lib.lane_widening.lane_widening` | `road_expansion` |
| Lane narrowing | `top_utils_lib.lane_narrowing.lane_narrowing` | `road_contraction` |

The last column identifies the categories used by our offline mixing process. These labels are not additional callable functions in the public toolkit.

Each operation receives the extracted geometry and the requested modification ratio. It operates on lane groups and their associated map elements, including dividers and boundaries. Retain the returned modified geometry and realized change rate. Keep repeated source variants for each type so that the offline export can select among them.

Our main mixed-prior setting uses these four topological types. Geometric operations such as global rotation and translation are separate settings; they are not additional categories in the default four-type mixture.

## 3. Export Aligned Raster and Vector Priors

Convert each unmodified and modified map to the same model coordinate system:

1. Merge lane-divider and road-divider instances into class `0` (`divider`), use class `1` for `ped_crossing`, and class `2` for `boundary`.
2. Resample each vector instance to 20 points along its arc length.
3. Store vectors in metric coordinates over the model range. Our legacy export converts the extracted planar coordinates with `(x, y) = (-y_source, x_source)` before this stage; do not apply this rotation again to priors already stored in model coordinates.
4. Rasterize consecutive point pairs in their corresponding semantic channels.
5. Store the binary raster, its corresponding vectors, and the realized modification rate in one offline entry.

For a point already in model coordinates, the raster conversion is:

```text
column = 100 * (x + 15) / 30
row    = 200 * (y + 30) / 60
```

The rasterizer clips pixel coordinates to the valid canvas bounds. The binary channels are ordered as `divider`, `ped_crossing`, and `boundary`. The vector array has shape `[N, 20, 3]`; its last axis is `(x, y, class_id)`.

The model loader later normalizes vector coordinates and pads or truncates the instance count. Keep metric coordinates in the stored vector array. Likewise, store a binary raster, not the model loader's internal rescaled raster values.

Each entry has the following fields:

```python
entry = {
    'rasterized_map': raster,
    'original_map': vectors,
    'change_rate': change_rate,
}
```

`original_map` is a legacy field name for the vectors belonging to that entry. For a biased entry it contains the biased vectors, not the clean ground-truth vectors.

## 4. Build Fixed Mixed-Prior Stores

The four source categories are combined offline. For each frame token, our mixing process:

1. Finds the available source categories and repeated variants.
2. Selects one category for its training-prior entry and a different category for its validation-prior entry.
3. Selects a source variant within each chosen category.
4. Writes these choices once under fixed keys. The model reads the exported choice rather than generating a fresh topological modification at every iteration.

Export the stores with the following layout:

```python
correct[sample_token]['geo_deformer']['no_change']['ED_0_level_low'] = clean_entry
mixed_train[sample_token]['mix4cats']['train']['v1'] = training_prior_entry
mixed_val[sample_token]['mix4cats']['test']['v1'] = validation_prior_entry
```

The filenames used by the released config are `correct.pkl`, `mixed_train.pkl`, and `mixed_val.pkl`. The `test` key in the validation-prior store is a retained export key. It does not refer to the hidden nuScenes test set.

The prior-store names do not determine the camera-frame split. The nuScenes info files in the config select the actual training and validation frames. When regenerating an export, retain the source-category and variant selections, the sampling seed, and the key names with that export.

## 5. Model Training and Evaluation

During training, the model reads the clean and selected biased raster for each sample token. It divides them into `10 x 10` pixel patches and selects half of the patch indices from the biased raster. Remaining patches use the clean raster. Reliability targets come from the patch source and the blank-region tests in the patch mixer.

The vector branch receives the selected biased vectors; it does not patch-mix vector instances with clean vectors.

During evaluation, the model receives the selected biased raster and biased vectors directly. It does not use the training patch-mixed raster as its prior input. The current loader also reads the clean map while constructing its auxiliary batch outputs, as described in [data.md](data.md).

The `0.6` modification ratio, four-category source selection, and `0.5` raster patch-mix ratio are separate stages with different meanings.

## Scope of This Release

This repository contains the model, the offline-prior reader, and the training-time patch mixer. The transformation library remains in [Syn-D-maps](https://github.com/healenrens/Syn-D-maps). Its example demonstrates the transformation API; it is not a batch exporter for the three PriorMaskMap pickle files by itself. The model-specific export layout is specified here and in [data.md](data.md).

To obtain the transformation toolkit over SSH:

```bash
git clone git@github.com:healenrens/Syn-D-maps.git
```

nuScenes data, generated pickle files, and source-generation experiments are not included in this model repository.

## Citation

The BibTeX entry is also available in [references.bib](references.bib).

```bibtex
@inproceedings{xu2025syntheticdeviationmaps,
  author    = {Haoming Xu and Yiyang Xiao and Wei Li and Yu Hu},
  title     = {Generating Synthetic Deviation Maps for Prior-Enhanced Vectorized HD Map Construction},
  booktitle = {2025 IEEE Intelligent Vehicles Symposium (IV)},
  year      = {2025},
  pages     = {419--426},
  doi       = {10.1109/IV64158.2025.11097394}
}
```
