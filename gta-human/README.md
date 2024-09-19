# GTA-Human

This is the first version of our datasets, which are built upon GTA-V for human pose and shape estimation.
It features single-person color image sequences with SMPL annotations.
Models pretrained on GTA-Human can be found at [MMHuman3D](https://github.com/open-mmlab/mmhuman3d/tree/main/configs/gta_human).


## Downloads

A small sample can be downloaded from [here](https://drive.google.com/file/d/1N-zsQvWd3uJ6P5oSGIjhqcQQfMqrwUxF/view?usp=sharing). To download the full dataset, please see below.

### Option 1: OpenXLab

GTA-Human is currently hosted on [OpenXLab](https://openxlab.org.cn/datasets/OpenXDLab/GTA-Human/tree/main/gta-human_release).
We recommend download files using [CLI tools](https://openxlab.org.cn/datasets/OpenXDLab/GTA-Human/cli/main):
```bash
openxlab dataset download --dataset-repo OpenXDLab/GTA-Human --source-path /gta-human_release --target-path /home/user/
```

You can selectively download files that you need, for example:
```bash
openxlab dataset download --dataset-repo OpenXDLab/GTA-Human --source-path /gta-human_release/images000.zip --target-path /home/user/gta-human_release/
```

### Option 2: OneDrive

We have backed-up all files on [OneDrive](https://pjlab-my.sharepoint.cn/:f:/g/personal/openmmlab_pjlab_org_cn/EjT3W_PHhApGvDB0geyC_g0BoBPK0tZfLVATnecU_bJl1A?e=rm3tSe).

## Data Structure

Please download the .zip files and place in the same directory.
```text
gta-human_release/   
├── image000.zip
├── image001.zip
├── ...
├── image030.zip
└── annotations.zip
```
Then decompress them:
```bash
unzip "*.zip"
```
The file structure should look like this:
```text
gta-human_release/   
├── images/  
│   └── seq_xxxxxxx/
│       ├── 00000000.jpeg
│       ├── 00000001.jpeg
│       └── ...
│
└── annotations/ 
        ├── seq_xxxxxx.pkl
        └── ...
```

## Data Loading
To read the images:
```python
import cv2
color_bgr = cv2.imread('/path/to/xxxxxxx.jpeg')
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)  # if RGB images are used
```

To read the annotations:
```python
import numpy as np
annot = dict(np.load('/path/to/xxxxxxx.pkl', allow_pickle=True))
```
Each .pkl consists of the following:
```text
{
    'is_male': bool,
    'fov': float, 
    'keypoints_2d': np.array of shape (n, 100, 3), 
    'keypoints_3d': np.array of shape (n, 100, 4),
    'occ': np.array of shape (n, 100),
    'self_occ': np.array of shape (n, 100),
    'num_frames': int,
    'weather': str, 
    'daytime': tuple, 
    'location_tag': str,
    'betas': np.array of shape (n, 10),
    'body_pose': np.array of shape (n, 69),
    'global_orient': np.array of shape (n, 3),
    'transl': np.array of shape (n, 3),
    'bbox_xywh': np.array of shape (n, 4),
}
```
Notes:
- `fov` has a constant value of 50.0.
- keypoints
    - `keypoints_3d` are 3D keypoints provided by the games' API, format is (x, y, z, 1.0).
    - `keypoints_2d` are projeced 3D keypoints on the image plane, format is (u, v, 1.0).
    - Definition of the 100 keypoints can be found in [MMHuman3D](https://github.com/open-mmlab/mmhuman3d/blob/main/mmhuman3d/core/conventions/keypoints_mapping/gta.py#L104-L205).
- `occ` indicates if a keypoint is occluded.
- `self_occ` indicates of a keypoint is occluded by the person's own body parts.
- `daytime` uses a (hour, minute, second) convention.


## Visualization

We provide a pyrender-based visualization tool to overlay 3D SMPL annotations on 2D images.
A small sample can be downloaded from [here](https://drive.google.com/file/d/1N-zsQvWd3uJ6P5oSGIjhqcQQfMqrwUxF/view?usp=sharing).

```
python visualizer.py <--root_dir> <--seq_name> <--body_model_path> <--save_path>
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'seq_xxxxxxxx'.
- body_model_path (str): directory in which SMPL body models are stored.
- save_path (str): path to save the visualization video.

Example:
```
python visualizer.py --root_dir /home/user/gta-human_release --seq_name seq_00000012 --body_model_path /home/user/body_models/ --save_path /home/user/seq_00000012_visual.mp4
```

Note that the SMPL model path should consist the following structure:
```text
body_models/   
└── smpl/  
    └── SMPL_NEUTRAL.npz
```
The body models may be downloaded from the [official website](https://smpl.is.tue.mpg.de/index.html).