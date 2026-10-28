# Installation

The implementation uses the MMCV 1.x and MMDetection 2.x APIs.

Our base environment follows [HRMapNet](https://github.com/HXMap/HRMapNet). Refer to its [step-by-step installation guide](https://github.com/HXMap/HRMapNet/blob/master/docs/install.md) for setting up PyTorch, MMCV, MMDetection, MMDetection3D, and compiled framework operators. Use the reference versions below for this release, then install the PriorMaskMap package with the command at the end of this page.

| Component | Reference environment |
|---|---|
| Python | 3.8 |
| CUDA runtime | 11.1 |
| PyTorch | 1.9.1+cu111 |
| torchvision | 0.10.1+cu111 |
| mmcv-full | 1.4.0 |
| mmdet | 2.14.0 |
| mmdet3d | 0.17.2 |
| numpy | 1.21.0 |
| scipy | 1.10.1 |
| nuscenes-devkit | 1.1.11 |
| Shapely | 1.8.5 |
| pyquaternion | 0.9.9 |
| prettytable | 3.11.0 |

Install these framework components before installing the package. `mmcv-full` and `mmdet3d` must be built for the selected PyTorch and CUDA versions. Required operators include multi-scale deformable attention and BEV pooling.

The repository does not vendor the framework source or compiled libraries. An existing compatible environment can be used directly:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

Additional runtime packages include OpenCV, Pillow, matplotlib, tqdm, terminaltables, and tensorboard. The model package does not build extra rasterization extensions.
