import argparse
import os
import os.path as osp
import glob
import tqdm
import numpy as np
import torch
import cv2
import trimesh
import smplx
import pyrender
from pyrender.viewer import DirectionalLight, Node


def load_data(root_dir, seq_name):

    # load images
    frame_paths = sorted(glob.glob(osp.join(root_dir, 'images', seq_name, '*.jpeg')))
    images = [cv2.imread(p) for p in frame_paths]

    # load parameters
    person_paths = sorted(glob.glob(osp.join(root_dir, 'annotations', seq_name, '*.npz')))
    persons = []
    for p in person_paths:
        person = dict(np.load(p))
        for annot in person.keys():
            if isinstance(person[annot], np.ndarray) and person[annot].ndim == 0:
                person[annot] = person[annot].item()
        persons.append(person)
    
    return images, persons


def compute_camera_pose(camera_pose):
    # Convert OpenCV cam pose to OpenGL cam pose
    
    # x,-y,-z -> x,y,z
    R_convention = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    camera_pose = R_convention @ camera_pose

    return camera_pose


def create_raymond_lights():
    # set directional light at axis origin, with -z direction align with +z direction of camera/world frame
    matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    return [Node(light=DirectionalLight(color=np.ones(3), intensity=2.0),
                 matrix=matrix)]


def draw_overlay(img, camera, camera_pose, meshes):
    
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

    for i, mesh in enumerate(meshes):
        scene.add(mesh, f'mesh_{i}')

    # Defination of cam_pose: transformation from cam coord to world coord
    scene.add(camera, pose=camera_pose)

    light_nodes = create_raymond_lights()
    for node in light_nodes:
        scene.add_node(node)

    r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080, point_size=1)
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
    img = img / 255
    output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * img)
    img = (output_img * 255).astype(np.uint8)

    return img


def visualize_2d(root_dir, seq_name, body_model_path, save_path):

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

    # Initialize components for rendering
    camera = pyrender.camera.IntrinsicsCamera(fx=1158.0337, fy=1158.0337, cx=960, cy=540)
    camera_pose = compute_camera_pose(np.eye(4))  # visualize in camera coord
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))  

    # Load data
    images, persons = load_data(root_dir, seq_name)

    # Draw overlay
    save_images = []
    for frame_idx, image in enumerate(tqdm.tqdm(images)):

        # Prepare meshes to visualize
        meshes = []
        for person in persons:            

            if person['is_valid_smplx']:

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

                out_mesh = trimesh.Trimesh(vertices, faces, process=False)
                mesh = pyrender.Mesh.from_trimesh(out_mesh, material=material)
                meshes.append(mesh)

        image = draw_overlay(image, camera, camera_pose, meshes)
        save_images.append(image)
    
    # Save visualization video
    if osp.dirname(save_path):
        os.makedirs(osp.dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps=15, frameSize=(1920, 1080))
    for image in save_images:
        video.write(image)
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory in which data is stored.')
    parser.add_argument('--seq_name', type=str, required=True,
                        help='sequence name, in the format \'seq_xxxxxxxx\'.')
    parser.add_argument('--body_model_path', type=str, required=True,
                        help="directory in which SMPL body models are stored.")
    parser.add_argument('--save_path', type=str, required=True,
                        help='path to save the visualization video.')
    args = parser.parse_args()

    visualize_2d(args.root_dir, args.seq_name, args.body_model_path, args.save_path)
