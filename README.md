# PriorMaskMap

Official implementation of PriorMaskMap (ECCV 2026).

Camera-based vector mapping with offline raster and vector priors.

## Contents

- `configs/priormaskmap_r50_24e.py`: ResNet-50 configuration.
- `priormaskmap/detectors/prior_mask_map.py`: model entry point.
- `priormaskmap/models/prior_head.py`: prior queries, reliability prediction, and prediction heads.
- `priormaskmap/models/vector_encoder.py`: vector-prior encoder.
- `priormaskmap/layers/prior_transformer.py`: BEV fusion and decoder inputs.
- `priormaskmap/priors/`: offline-map loading and patch mixing.
- `priormaskmap/datasets/`: nuScenes data pipelines and Chamfer evaluation.
- `tools/`: training and evaluation entry points.

This repository contains one camera model and one training configuration. It does not include experiment archives, diagnostic tools, alternative architectures, pretrained weights, or datasets.

## Installation

Use Python 3.8 with the CUDA-enabled framework versions in [docs/installation.md](docs/installation.md). The framework installation must include its compiled operators.

Install this package in that environment:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

## Data

Arrange the files as described in [docs/data.md](docs/data.md):

```text
data/
  nuscenes/
    samples/
    sweeps/
    maps/
    nuscenes_map_infos_temporal_train.pkl
    nuscenes_map_infos_temporal_val.pkl
    nuscenes_map_anns_val.json
  priors/
    correct.pkl
    mixed_train.pkl
    mixed_val.pkl
checkpoints/
  resnet50.pth
```

The config uses relative paths. Override them with `--cfg-options` when using another layout.

## Prior Generation

Our biased priors follow [Syn-D-maps](https://github.com/healenrens/Syn-D-maps), introduced in **Generating Synthetic Deviation Maps for Prior-Enhanced Vectorized HD Map Construction**, IEEE IV 2025.

The main setting uses lane addition, lane reduction, lane widening, and lane narrowing with a modification ratio of `0.6`. We export each modified map as aligned raster and vector priors, build fixed training and validation prior stores, and apply raster patch mixing during model training.

See [docs/data_generation.md](docs/data_generation.md) for the source paper, transformation mapping, coordinate conversion, offline export format, and the distinction between corruption ratio and training-time patch mixing.

## Training

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29531 \
  bash tools/dist_train.sh configs/priormaskmap_r50_24e.py 8 \
  --work-dir work_dirs/priormaskmap_r50_24e
```

The configuration uses four samples per GPU, 24 epochs, and an AdamW learning rate of `6e-4`. Resume with `--resume-from work_dirs/priormaskmap_r50_24e/latest.pth`.

## Evaluation

Single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/priormaskmap_r50_24e.py checkpoints/priormaskmap.pth \
  --work-dir work_dirs/evaluation
```

Distributed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29541 \
  bash tools/dist_test.sh configs/priormaskmap_r50_24e.py \
  checkpoints/priormaskmap.pth 4 --work-dir work_dirs/evaluation
```

Evaluation writes predictions and `metrics.json` to the selected work directory. Add `--out work_dirs/predictions.pkl` to save raw predictions.

## License

See [LICENSE](LICENSE), [THIRD_PARTY.md](THIRD_PARTY.md), and the notices in the source files.
