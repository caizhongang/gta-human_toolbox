# Toolbox for GTA-Human Datasets

Please visit our [Homepage](https://caizhongang.github.io/projects/GTA-Human/) for more details.
The toolchain based on GTA-V is released [here](https://github.com/Wei-Chen-hub/GTA-Human-tools).

## Installation

To use our visulization tools, relevant python packages need to be installed.
```bash
conda create -n gta-human python=3.9 -y
conda activate gta-human
pip install torch==1.12.1 opencv-python==4.10.0.84 smplx==0.1.28 chumpy==0.70 trimesh==4.4.3 tqdm==4.66.4 numpy==1.23.1 pyrender==0.1.45
```

It is also highly recommended to install `openxlab` package to facilitate file downloading.
```bash
pip install openxlab
```

If you'd like to use the `visualizer_3d.py` for GTA-Human II, please also install:
```bash
pip install open3d==0.14.1
```


## Datasets

Please click on the dataset name for download links and visualization instructions.

| Features | [GTA-Human](./gta-human/README.md) | [GTA-Human II](./gta-human_v2/README.md) |
| :------------------------ | :-------: | :-------: |
| Num of Scenes             | 20,005    | 10,224    |
| Num of Person Sequences   | 20,005    | 35,352    |
| Color Images              | Yes       | Yes       |
| 3D BBox & Point Cloud     | No        | Yes       |
| Parametric Model          | SMPL      | SMPL-X    |
| Num of Persons per Scene  | 1         | 1-6       |


## Citation
```text
@ARTICLE{10652891,
  author={Cai, Zhongang and Zhang, Mingyuan and Ren, Jiawei and Wei, Chen and Ren, Daxuan and Lin, Zhengyu and Zhao, Haiyu and Yang, Lei and Loy, Chen Change and Liu, Ziwei},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Playing for 3D Human Recovery}, 
  year={2024},
  volume={},
  number={},
  pages={1-12},
  keywords={Three-dimensional displays;Annotations;Synthetic data;Shape;Training;Parametric statistics;Solid modeling;Human Pose and Shape Estimation;3D Human Recovery;Parametric Humans;Synthetic Data;Dataset},
  doi={10.1109/TPAMI.2024.3450537}
}
```
