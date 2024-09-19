# GTA-Human II

This is the latest version of our datasets, which are built upon GTA-V for expressive human pose and shape estimation.
It features multi-person scenes with SMPL-X annotations.
In addition to color image sequences, 3D bounding boxes and cropped point clouds (generated from synthetic depth images) are also provided.

## Downloads

A small sample of GTA-Human II can be downloaded from [here](https://drive.google.com/file/d/1N0-JDP6iktPC6-lqpBTARB2mqwO7cts2/view?usp=sharing). 
The full set is currently hosted on [OpenXLab](https://openxlab.org.cn/datasets/OpenXDLab/GTA-Human/tree/main/gta-human_v2_release).
We recommend download files using [CLI tools](https://openxlab.org.cn/datasets/OpenXDLab/GTA-Human/cli/main):
```bash
openxlab dataset download --dataset-repo OpenXDLab/GTA-Human --source-path /gta-human_v2_release --target-path /home/user/
```

You can selectively download files that you need, for example:
```bash
openxlab dataset download --dataset-repo OpenXDLab/GTA-Human --source-path /gta-human_v2_release/images_part_1.7z --target-path /home/user/gta-human_v2_release/
```

## Data Structure

Please download the .7z files and place in the same directory. 
Note that you may not need the point clouds if you are working on image- or video-based methods.
```text
gta-human_v2_release/   
├── images_part_1.7z
├── images_part_2.7z
├── images_part_3.7z
├── images_part_4.7z
├── images_part_5.7z
├── point_clouds_1.7z
├── point_clouds_2.7z
├── point_clouds_3.7z
├── point_clouds_4.7z
└── annotations.7z
```
Then decompress them:
```bash
7z x *.7z
```
The file structure should look like this:
```text
gta-human_v2_release/   
├── images/  
│   └── seq_xxxxxxx/
│       ├── 00000000.jpeg
│       ├── 00000001.jpeg
│       └── ...
│
├── point_clouds/  
│   └── seq_xxxxxxx/
│       ├── bbox_aaaaaa_0000.ply  # (bbox of person ID aaaaaa at frame 0)
│       ├── bbox_aaaaaa_0001.ply  # (bbox of person ID aaaaaa at frame 1)
│       ├── ...
│       ├── bbox_bbbbbb_0000.ply  # (bbox of person ID bbbbbb at frame 0)
│       ├── bbox_bbbbbb_0001.ply  # (bbox of person ID bbbbbb at frame 1)
│       ├── ...
│       ├── pcd_aaaaaa_0000.pcd  # (point cloud of person ID aaaaaa at frame 0)
│       ├── pcd_aaaaaa_0001.pcd  # (point cloud of person ID aaaaaa at frame 1)
│       ├── ...
│       ├── pcd_bbbbbb_0000.pcd  # (point cloud of person ID bbbbbb at frame 0)
│       ├── pcd_bbbbbb_0001.pcd  # (point cloud of person ID bbbbbb at frame 1)
│       └── ...
│
└── annotations/ 
    └── seq_xxxxxxx/
        ├── aaaaaa.npz  
        ├── bbbbbb.npz
        └── ...
```

## Data Loading
To read the images:
```python
import cv2
color_bgr = cv2.imread('/path/to/xxxxxxx.jpeg')
color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)  # if RGB images are used
```

To read the 3D bounding boxes and cropped point clouds:
```python
import open3d as o3d
import numpy as np
point_cloud_o3d = o3d.io.read_point_cloud('/path/to/xxxxxx.pcd')  # geometry::PointCloud with n points.
point_cloud = np.array(point_cloud_o3d.points)  # np.ndarray of (n, 3)
bounding_box = o3d.io.read_line_set('/path/to/xxxxxx.ply')  # geometry::LineSet with 12 lines
```
Notes:
- The point clouds are cropped from the scene point cloud (generated from a synthetic depth image) using the 3D bounding boxes. The original depth image or scene point cloud are very large and are hence excluded from the dataset release.
- Only subjects with valid SMPL-X annotation have their point clouds released.
- We truncate the point clouds more than 10 m away from the camera as the typical maximum range of commercial depth sensors does not exceed 10 m. This means if subjects are more than 10 m away, their bounding boxes and point clouds are not recorded.

To read the annotations:
```python
import numpy as np
annot = dict(np.load('/path/to/xxxxxxx.npz'))
for key in annot:
    if isinstance(annot[key], np.ndarray) and person[key].ndim == 0:
        annot[key] = annot[key].item()
```
Each .npz consists of the following:
```text
{
    'is_male': bool,
    'ped_action': str,
    'fov': float, 
    'keypoints_2d': np.array of shape (n, 100, 3), 
    'keypoints_3d': np.array of shape (n, 100, 4),
    'occ': np.array of shape (n, 100),
    'self_occ': np.array of shape (n, 100),
    'num_frames': int,
    'weather': str, 
    'daytime': tuple, 
    'location_tag': str,
    'bbox_xywh': np.array of shape (n, 4),
    'is_valid_smplx': bool,
    'betas': np.array of shape (n, 10),
    'body_pose': np.array of shape (n, 69),
    'global_orient': np.array of shape (n, 3),
    'transl': np.array of shape (n, 3),
    'left_hand_pose': np.array of shape (n, 24),
    'right_hand_pose': np.array of shape (n, 24),
}
```
Notes:
- `is_valid_smplx` indicates if the subject's annotation has valid SMPL-X parameters.
    - Valid SMPL-X annotations are those with sufficient movement and high-quality fitting.
    - If invalid, SMPL-X parameters are not provided, but other annotations are still available.
    - 3D bounding boxes and cropped point clouds are only available for subjects with valid SMPL-X.
    - There are 35,352 valid sequences and 13,168 invalid sequences.
- `fov` has a constant value of 50.0.
- keypoints
    - `keypoints_3d` are 3D keypoints provided by the games' API, format is (x, y, z, 1.0).
    - `keypoints_2d` are projeced 3D keypoints on the image plane, format is (u, v, 1.0).
    - Definition of the 100 keypoints can be found in [MMHuman3D](https://github.com/open-mmlab/mmhuman3d/blob/main/mmhuman3d/core/conventions/keypoints_mapping/gta.py#L104-L205).
- `occ` indicates if a keypoint is occluded.
- `self_occ` indicates of a keypoint is occluded by the person's own body parts.
- `daytime` uses a (hour, minute, second) convention.


## Visualization

### Run 2D Visualizer

We provide a pyrender-based visualization tool to overlay 3D SMPL-X annotations on 2D images.
A small sample of GTA-Human II can be downloaded from [here](https://drive.google.com/file/d/1N0-JDP6iktPC6-lqpBTARB2mqwO7cts2/view?usp=sharing). 

```
python visualizer_2d.py <--root_dir> <--seq_name> <--body_model_path> <--save_path>
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'seq_xxxxxxxx'.
- body_model_path (str): directory in which SMPL body models are stored.
- save_path (str): path to save the visualization video.

Example:
```
python visualizer_2d.py --root_dir /home/user/gta-human_v2_release --seq_name seq_00087011 --body_model_path /home/user/body_models/ --save_path /home/user/seq_00087011_2dvisual.mp4
```

### Run 3D Visualizer

We also provide a visualization tool for 3D bounding boxes and cropped point clouds.

```
python visualizer_3d.py <--root_dir> <--seq_name> <--save_path> [--virtual_cam] [--visualize_smplx] [--body_model_path]
```
- root_dir (str): root directory in which data is stored.
- seq_name (str): sequence name, in the format 'seq_xxxxxxxx'.
- save_path (str): path to save the visualization video.
- virtual_cam (str, optional): path to load virtual camera pose config. Defaults to assets/virtual_cam.json.
- visualize_smplx (flag, optional): whether to visualize SMPL 3D mesh model.
- body_model_path (str, optional): directory in which SMPL body models are stored.

Example:
```
python visualizer_3d.py --root_dir /home/user/gta-human_v2_release --seq_name seq_00087011 --save_path /home/user/seq_00087011_3dvisual.mp4
```

Note that the SMPL-X model path should consist the following structure:
```text
body_models/   
└── smplx/  
    └── SMPLX_NEUTRAL.npz
```
The body models may be downloaded from the [official website](https://smpl-x.is.tue.mpg.de/).
