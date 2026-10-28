# Third-Party Notices

This repository is an incremental derivative of the camera mapping code in
[HRMapNet](https://github.com/HXMap/HRMapNet). It includes adapted model, data,
attention, loss, and evaluation code from that codebase and its dependencies.

The inherited mapping implementation is based on
[MapTR / MapTRv2](https://github.com/hustvl/MapTR),
[BEVFormer](https://github.com/fundamentalvision/BEVFormer), and
[MMDetection3D](https://github.com/open-mmlab/mmdetection3d).

The upstream MIT license and copyright notice are retained in `LICENSE`.
The bundled source retains its existing copyright notices.
The MMDetection3D license is retained in `licenses/mmdetection3d_LICENSE.txt`.
MMCV, MMDetection, MMDetection3D, PyTorch, and the nuScenes SDK are external
dependencies and retain their own licenses.

Public package paths and registry names have been renamed for this project.
These namespace changes do not transfer ownership of upstream contributions.
The adapted files cover the model head and detector, vector encoding, prior
loading and patch mixing, BEV fusion, training/evaluation entry points, and
the required data and attention support modules.

## Prior-Generation Framework

The biased-map generation procedure follows
[Syn-D-maps](https://github.com/healenrens/Syn-D-maps), the toolkit associated
with **Generating Synthetic Deviation Maps for Prior-Enhanced Vectorized HD Map
Construction**, IEEE IV 2025, DOI: `10.1109/IV64158.2025.11097394`.
The toolkit is maintained in its own repository under the Apache 2.0 license.
Its transformation library is not vendored in this model repository. See
[docs/data_generation.md](docs/data_generation.md) for the PriorMaskMap data recipe.
