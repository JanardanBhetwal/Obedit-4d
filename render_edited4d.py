#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import math
import imageio
import numpy as np
import torch
from scene import Scene
import os
from tqdm import tqdm
from utils.general_utils import safe_state
from utils.graphics_utils import getProjectionMatrix
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def to8b(x):
    return (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

def render_edited(gaussians, viewpoint_camera, zoom_factor=1.05, mask=None):
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    zoom_factor = max(1.0, float(zoom_factor))
    # Set up rasterization configuration
    zoomed_fovx = viewpoint_camera.FoVx / zoom_factor
    zoomed_fovy = viewpoint_camera.FoVy / zoom_factor
    tanfovx = math.tan(zoomed_fovx * 0.5)
    tanfovy = math.tan(zoomed_fovy * 0.5)
    world_view_transform = viewpoint_camera.world_view_transform.cuda()
    znear = getattr(viewpoint_camera, "znear", 0.01)
    zfar = getattr(viewpoint_camera, "zfar", 100.0)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=zoomed_fovx, fovY=zoomed_fovy).transpose(0, 1).to(world_view_transform.device)
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0)).squeeze(0)
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0   

    raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=False,
        )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    ############################################################################################
    means3D = gaussians.get_xyz
    means2D = screenspace_points
    opacity = gaussians._opacity
    scales = gaussians._scaling
    rotations = gaussians._rotation  
    shs = gaussians.get_features      
    ############################################################################################
    #scene.getTrainCameras()[300].time
    time_cond = torch.tensor(viewpoint_camera.time).to(means3D.device).repeat(means3D.shape[0],1)
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = gaussians._deformation(means3D, scales, 
                                                                 rotations, opacity, shs,
                                                                 time_cond)
        
    scales_final = gaussians.scaling_activation(scales_final)
    rotations_final = gaussians.rotation_activation(rotations_final)
    opacity_final = gaussians.opacity_activation(opacity_final)
    
    if (mask is None):
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final,
            means2D = means2D,
            shs = shs_final,
            colors_precomp = None,
            opacities = opacity_final,
            scales = scales_final,
            rotations = rotations_final,
            cov3D_precomp = None)
    else:
        #mask = ~mask
        rendered_image, radii, depth = rasterizer(
            means3D = means3D_final[mask],
            means2D = means2D,
            shs = shs_final[mask],
            colors_precomp = None,
            opacities = opacity_final[mask],
            scales = scales_final[mask],
            rotations = rotations_final[mask],
            cov3D_precomp = None)
    
    return rendered_image

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int) #
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip_video", action="store_true")
    parser.add_argument("--configs", type=str)
    parser.add_argument("--ply_path", type=str)
    parser.add_argument("--render_zoom", type=float, default=1.05, help="Virtual zoom factor for PLY rendering. Values > 1.0 narrow the FoV and crop edge noise.")
    parser.add_argument("--prompt", type=str, default=None, help="Edit prompt (used for output filename)")
    args = get_combined_args(parser)
    print("Rendering " , args.ply_path)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    dataset= model.extract(args)
    iteration = args.iteration
    hyperparam = hyperparam.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, hyperparam)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    before_xyz = gaussians.get_xyz
    
    print("before edit: ", gaussians.get_xyz.shape)
    
    cam_type=scene.dataset_type
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    gaussians.load_ply(args.ply_path)
    
    # Automatically find and load the latest iteration available
    point_cloud_dir = os.path.join(args.model_path, "point_cloud")
    
    # Find all iteration folders
    iterations = []
    if os.path.exists(point_cloud_dir):
        for name in os.listdir(point_cloud_dir):
            if name.startswith("iteration_"):
                try:
                    iterations.append(int(name.split("_")[1]))
                except ValueError:
                    pass  # ignore invalid folder names
    else:
        raise FileNotFoundError(f"Point cloud folder not found: {point_cloud_dir}")
    
    if not iterations:
        raise FileNotFoundError(f"No iteration folders found in {point_cloud_dir}")
    
    latest_iter = max(iterations)
    print(f"Loading latest checkpoint: iteration_{latest_iter}")
    gaussians.load_model(os.path.join(point_cloud_dir, f"iteration_{latest_iter}"))
    
    after_xyz = gaussians.get_xyz
    print("after edit: ", gaussians.get_xyz.shape)

    ## VideoCameras: moving camera / TestCameras: fixed camera (for DyNeRF)
    cameras = scene.getVideoCameras()
    imgs = []
    for idx, viewpoint_camera in enumerate(tqdm(cameras, desc="Rendering progress")):
        rendered_img = render_edited(gaussians, viewpoint_camera, zoom_factor=args.render_zoom)
        imgs.append(to8b(rendered_img.detach().cpu()).transpose(1,2,0))

    ## Determine a prompt-based filename so different prompts don't overwrite each other
    import re
    prompt_tag = getattr(args, "prompt", None)
    if not prompt_tag:
        # Try to extract prompt from ply_path  (e.g. .../point_cloud_3dedit/<prompt>/iteration_X/...)
        ply_parts = os.path.normpath(args.ply_path).split(os.sep)
        for i, part in enumerate(ply_parts):
            if part in ("point_cloud_3dedit", "point_cloud_refine") and i + 1 < len(ply_parts):
                prompt_tag = ply_parts[i + 1]
                break
    if prompt_tag:
        # Sanitize: replace non-alphanumeric characters (except spaces) with nothing, spaces with underscores
        safe_prompt = re.sub(r'[^\w\s-]', '', prompt_tag).strip()
        safe_prompt = re.sub(r'[\s]+', '_', safe_prompt)
    else:
        safe_prompt = "unknown_prompt"
    scene_name = os.path.splitext(os.path.basename(args.configs))[0]
    zoom_tag = "" if abs(args.render_zoom - 1.0) < 1e-6 else f"_zoom{str(args.render_zoom).replace('.', 'p')}"
    video_path = os.path.join(args.model_path, f"edited_{scene_name}_{safe_prompt}{zoom_tag}.mp4")
    imageio.mimwrite(video_path, imgs, fps=30)
    print(f"Video Saved: {video_path}")