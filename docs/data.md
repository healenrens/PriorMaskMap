# Data Format

The prior-generation procedure and the Syn-D-maps paper citation are documented in [data_generation.md](data_generation.md).

The nuScenes info files contain image paths, camera calibration, poses, and frame tokens. The map-annotation JSON supplies the validation ground truth for `divider`, `ped_crossing`, and `boundary`.

## Offline Priors

Each prior pickle contains a dictionary with four key levels:

```python
priors[sample_token][group][split][variant] = {
    'rasterized_map': raster,
    'original_map': vectors,
    'change_rate': change_rate,
}
```

- `rasterized_map`: three-channel binary tensor with shape `[3, 200, 100]`.
- `original_map`: vector array with shape `[N, 20, 3]`; the final axis contains `x`, `y`, and the class ID.
- `change_rate`: stored modification-rate value.

The `original_map` field stores the vectors belonging to that entry. In a modified-prior file, these are modified vectors.

The default configuration selects:

| File | Key |
|---|---|
| `correct.pkl` | `geo_deformer / no_change / ED_0_level_low` |
| `mixed_train.pkl` | `mix4cats / train / v1` |
| `mixed_val.pkl` | `mix4cats / test / v1` |

Coordinates cover `x ∈ [-15, 15]` and `y ∈ [-30, 30]` metres. Vectors are padded or truncated to 35 instances before encoding. Each instance contains 20 points.

## Loading

On the first use of a pickle pair, the loader writes per-entry files and an index under `.cache_data/`. Later runs read the index and individual entries. Each loader retains at most 1000 entries in its LRU cache. Keep `.cache_data/` on a writable disk with sufficient capacity.

Training constructs patch-mixed raster priors and reliability labels from the correct and modified maps. The vector input uses the modified vectors. Evaluation passes the selected modified raster and vectors to the model; its loader also reads the correct file while constructing the batch.

The repository does not include nuScenes files, generated priors, map-generation experiments, or checkpoints.
