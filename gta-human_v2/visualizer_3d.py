import glob
import os
import os.path as osp
import numpy as np
import argparse
import cv2
import tqdm
import trimesh
import open3d as o3d
import smplx
import torch


def load_annot(root_dir, seq_name):

    person_paths = sorted(glob.glob(osp.join(root_dir, 'annotations', seq_name, '*.npz')))
    persons = {}
    for p in person_paths:
        person_id = osp.splitext(osp.basename(p))[0]
        person = dict(np.load(p))
        for annot in person.keys():
            if isinstance(person[annot], np.ndarray) and person[annot].ndim == 0:
                person[annot] = person[annot].item()
        persons[person_id] = person

    num_frames = person['num_frames']
    return num_frames, persons


def load_3d_data(root_dir, seq_name, person_id, frame_idx):
    # load bbox
    bbox_path = osp.join(root_dir, 'point_clouds', seq_name, f'bbox_{person_id}_{frame_idx:04d}.ply')
    bbox = o3d.io.read_line_set(bbox_path)

    # load point cloud
    point_cloud_path = osp.join(root_dir, 'point_clouds', seq_name, f'pcd_{person_id}_{frame_idx:04d}.pcd')
    point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    return bbox, point_cloud


def visualize_3d(root_dir, seq_name, virtual_cam, save_path, visualize_smplx=False, body_model_path=None):
    
    if visualize_smplx:

        # Set device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize body model
        body_model = smplx.create(
            body_model_path, 
            model_type='smplx',
            flat_hand_mean=True, 
            use_face_contour=True, 
            use_pca=True, 
            num_betas=10, 
            num_pca_comps=24
        ).to(device)

    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(virtual_cam)

    # Set point size
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  

    # Add axis
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # Load annotations
    num_frames, persons = load_annot(root_dir, seq_name)

    save_images = []
    for frame_idx in tqdm.tqdm(range(num_frames)):

        geometries = []
        for person_id, person in persons.items():            

            # Prepare person meshes to visualize
            if visualize_smplx and person['is_valid_smplx']:

                model_output = body_model(
                    global_orient=torch.tensor(person['global_orient'][[frame_idx]], device=device),
                    body_pose=torch.tensor(person['body_pose'][[frame_idx]], device=device),
                    transl=torch.tensor(person['transl'][[frame_idx]], device=device),
                    betas=torch.tensor(person['betas'][[frame_idx]], device=device),
                    left_hand_pose=torch.tensor(person['left_hand_pose'][[frame_idx]], device=device),
                    right_hand_pose=torch.tensor(person['right_hand_pose'][[frame_idx]], device=device),
                    return_verts=True, 
                )
                vertices = model_output.vertices.detach().cpu().numpy().squeeze()
                faces = body_model.faces
                trimesh_mesh = trimesh.Trimesh(vertices, faces, process=False)
                open3d_mesh = trimesh_mesh.as_open3d
                open3d_mesh.compute_vertex_normals()
                geometries.append(open3d_mesh)
        
            # Prepare 3D bounding boxes and point clouds to visualize
            bbox, point_cloud = load_3d_data(root_dir, seq_name, person_id, frame_idx) 
            geometries.append(bbox)
            geometries.append(point_cloud)

        for geometry in geometries:
            vis.add_geometry(geometry)

        # Render the frame
        ctr.convert_from_pinhole_camera_parameters(parameters)
        vis.poll_events()
        vis.update_renderer()
        
        save_image = np.array(vis.capture_screen_float_buffer())
        save_images.append(save_image)

        for geometry in geometries:
            vis.remove_geometry(geometry)

    vis.destroy_window()

    # Save visualization video
    if osp.dirname(save_path):
        os.makedirs(osp.dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps=15, frameSize=(1920, 1080))
    for image in save_images:
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video.write(image)
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory in which data is stored.')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='sequence name, in the format \'seq_xxxxxxxx\'.')
    parser.add_argument('--virtual_cam', type=str, default='assets/virtual_cam.json',
                        help='virtual camera pose.')
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save the visualization video.')
    parser.add_argument('--visualize_smplx', action='store_true',
                    help='whether to visualize SMPL 3D mesh model.')
    parser.add_argument('--body_model_path', type=str, default='/home/user/body_models',
                    help="directory in which SMPL body models are stored.")
    args = parser.parse_args()

    if args.visualize_smplx:
        assert osp.isdir(args.body_model_path), f"Body model path {args.body_model_path} does not exist."
        
    visualize_3d(args.root_dir, args.seq_name, args.virtual_cam, args.save_path, args.visualize_smplx, args.body_model_path)
