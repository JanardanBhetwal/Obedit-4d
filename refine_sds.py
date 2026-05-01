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
import sys
sys.stdout.isatty = lambda: False

import numpy as np
import random
import os, sys
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, ModelHiddenParams, OptimizationParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
    )
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer
)
from ip2p_models.models.ip2p_pipeline import InstructPix2PixPipeline
from ip2p_models.models.ip2p_unet import UNet3DConditionModel
import configargparse as argparse
import torch
import torchvision
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F

from PIL import Image

from einops import rearrange
from tqdm import tqdm
import math
import os

from pytorch_lightning import seed_everything   

def encode_fast(ip2p, x):
    return ip2p.vae.encode(2 * x - 1).latent_dist.sample() * 0.18215

def encode_1(ip2p, input, encode_batch_size=1):
    """Encode images to latents with batch processing to reduce memory usage."""
    latents_list = []
    for i in range(0, input.shape[0], encode_batch_size):
        batch = input[i:i+encode_batch_size]
        latent = ip2p.vae.encode(2*batch-1).latent_dist.sample() * 0.18215
        latents_list.append(latent.cpu())  # Move to CPU immediately
        del batch, latent
        torch.cuda.empty_cache()
    latents = torch.cat(latents_list, dim=0).to(device=input.device)  # Move back to GPU
    return latents

def encode_2(ip2p, input, encode_batch_size=1):
    """Encode images to latents with batch processing to reduce memory usage."""
    latents_list = []
    for i in range(0, input.shape[0], encode_batch_size):
        batch = input[i:i+encode_batch_size]
        latent = ip2p.vae.encode(2*batch-1).latent_dist.mode()
        latents_list.append(latent.cpu())  # Move to CPU immediately
        del batch, latent
        torch.cuda.empty_cache()
    image_latents = torch.cat(latents_list, dim=0).to(device=input.device)  # Move back to GPU
    return image_latents
    
def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer, ip2p, prompt, guidance_scale, image_guidance_scale):
    
    torch_dtype = torch.float16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- SPEED & MEMORY EFFICIENCY SETTINGS ---
    sequence_length = 4   # 4 views batched into a single 3D UNet pass
    patch_size = 256      # High-res random crop; avoids downscaling entire images

    diffusion_step = 20
    num_train_timesteps = 1000
    ###
    
    first_iter = 0
    gaussians.training_only3dgs_setup(opt)

    # FREEZE XYZ to prevent 4DGS motion misalignment
    for param_group in gaussians.optimizer.param_groups:
        if param_group["name"] == "xyz":
            param_group["lr"] = 0.0
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0

    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1

    video_cams = scene.getVideoCameras()
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("data loading done")
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, sampler=sampler, num_workers=1, collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False

    # Pre-encode the prompt once — reused every iteration
    prompt_embeds = ip2p._encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )  # [3, 77, 768]

    ip2p.scheduler.config.num_train_timesteps = num_train_timesteps
    ip2p.scheduler.set_timesteps(diffusion_step)

    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    count += 1
                    viewpoint_index = (count) % len(video_cams)
                    if (count // (len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # -----------------------------------------------------------------------
        # BATCHED FAST LOOP (High Speed, Minimal Memory)
        # -----------------------------------------------------------------------
        gaussians.optimizer.zero_grad(set_to_none=True)

        images_patch_list = []
        gt_patch_list = []
        viewspace_point_tensor_list = []
        visibility_filter_list = []
        radii_list = []
        
        psnr_log = 0.0

        # 1. FAST RENDER & CROP LOOP
        for seq_idx in range(sequence_length):
            if opt.dataloader and not load_in_memory:
                try:
                    viewpoint_cam = next(loader)[0]
                except StopIteration:
                    if not random_loader:
                        viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size, shuffle=True, num_workers=1, collate_fn=list)
                        random_loader = True
                    loader = iter(viewpoint_stack_loader)
                    torch.cuda.empty_cache()
                    try:
                        viewpoint_cam = next(loader)[0]
                    except StopIteration:
                        break
            else:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()

            # Render
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, cam_type=scene.dataset_type)
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

            gt_image = viewpoint_cam.original_image.cuda() if scene.dataset_type != "PanopticSports" else viewpoint_cam['image'].cuda()
            
            # Calculate PSNR on full resolution to log
            with torch.no_grad():
                psnr_log += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()

            _, H, W = image.shape

            # Crop 256x256
            if H >= patch_size and W >= patch_size:
                y0 = random.randint(0, H - patch_size)
                x0 = random.randint(0, W - patch_size)
                image_patch = image[:, y0:y0+patch_size, x0:x0+patch_size].unsqueeze(0)
                gt_patch    = gt_image[:, y0:y0+patch_size, x0:x0+patch_size].unsqueeze(0)
            else:
                image_patch = F.interpolate(image.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                gt_patch    = F.interpolate(gt_image.unsqueeze(0), size=(patch_size, patch_size), mode='bilinear', align_corners=False)

            images_patch_list.append(image_patch)
            gt_patch_list.append(gt_patch)
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            
            # CRITICAL: Free full resolution images immediately to preserve VRAM!
            del image, gt_image, render_pkg
            
        # 2. BATCH TENSORS
        image_tensor = torch.cat(images_patch_list, 0).to(device=device, dtype=torch_dtype)
        gt_image_tensor = torch.cat(gt_patch_list, 0).to(device=device, dtype=torch_dtype)
        radii_batch = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter_batch = torch.cat(visibility_filter_list).any(dim=0)
        
        # 3. VAE ENCODE (Batched)
        latents = encode_fast(ip2p, image_tensor)
        image_latents = encode_fast(ip2p, gt_image_tensor)
        
        # Reshape for 3D UNet: b=1, c=4, f=sequence_length
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length).to(device=device, dtype=torch_dtype)
        image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length).to(device=device, dtype=torch_dtype)
        uncond_image_latents = torch.zeros_like(image_latents)

        # 4. TIMESTEP ANNEALING (Linear decay from 0.6 -> 0.2)
        progress_ratio = iteration / float(opt.iterations)
        max_t = 0.6 - (0.4 * progress_ratio) 
        min_t = 0.1
        
        noise = torch.randn_like(latents)
        t = torch.randint(int(1000 * min_t), int(1000 * max_t), [1], dtype=torch.long, device=device)
        latents_noisy = ip2p.scheduler.add_noise(latents, noise, t)
        
        image_latents_cfg = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
        latent_model_input = torch.cat([latents_noisy] * 3)
        latent_model_input = torch.cat([latent_model_input, image_latents_cfg], dim=1)

        # 5. UNET FORWARD PASS (Single batched run)
        with torch.no_grad():
            noise_pred = ip2p.unet(latent_model_input, t, prompt_embeds, None, None, False)[0]
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            
            guidance_scale_adjusted = min(guidance_scale, 8.0) # Limit over-saturation
            noise_pred = (
                noise_pred_uncond
                + guidance_scale_adjusted * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

        alphas = ip2p.scheduler.alphas_cumprod.to(device)
        w = (1 - alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        
        target = (latents - grad).detach().to(dtype=torch_dtype)
        
        # 6. LOSS & BACKWARD
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / sequence_length
        l1_reg = F.l1_loss(image_tensor, gt_image_tensor) * 0.1 # Anchors geometry
        
        loss = loss_sds + l1_reg
        loss.backward()
        
        # Accumulate viewspace gradients for densification
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor_list[0])
        for idx in range(len(viewspace_point_tensor_list)):
            if viewspace_point_tensor_list[idx].grad is not None:
                viewspace_point_tensor_grad += viewspace_point_tensor_list[idx].grad
                
        iter_end.record()
        
        # CLEANUP (Crucial for next iteration)
        del images_patch_list, gt_patch_list, viewspace_point_tensor_list
        del image_tensor, gt_image_tensor, latents, image_latents, uncond_image_latents
        del noise, latents_noisy, latent_model_input, noise_pred, grad, target
        torch.cuda.empty_cache()

        psnr_ = psnr_log / sequence_length

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * float(psnr_) + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss":  f"{ema_loss_for_log:.{7}f}",
                    "psnr":  f"{float(ema_psnr_for_log):.{2}f}",
                    "point": f"{total_point}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            timer.pause()
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save_refine(iteration, stage, prompt)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration % 100 == 99):
                    render_training_image(scene, gaussians, [test_cams[iteration % len(test_cams)]],  render, pipe, background, stage+"test",  iteration, timer.get_elapsed_time(), scene.dataset_type)
                    render_training_image(scene, gaussians, [train_cams[iteration % len(train_cams)]], render, pipe, background, stage+"train", iteration, timer.get_elapsed_time(), scene.dataset_type)
            timer.start()

            # 7. DENSIFICATION & OPTIMIZER STEP
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter_batch] = torch.max(
                    gaussians.max_radii2D[visibility_filter_batch],
                    radii_batch[visibility_filter_batch]
                )
                gaussians.add_densification_stats(viewspace_point_tensor_grad * 0.000001, visibility_filter_batch)

                opacity_threshold  = opt.opacity_threshold_fine_init  - iteration * (opt.opacity_threshold_fine_init  - opt.opacity_threshold_fine_after)  / opt.densify_until_iter
                densify_threshold  = opt.densify_grad_threshold_fine_init - iteration * (opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after) / opt.densify_until_iter

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 30000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                    torch.cuda.empty_cache()

                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0] > 10000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    torch.cuda.empty_cache()

                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0] < 30000 and opt.add_point:
                    gaussians.grow(5, 5, scene.model_path, iteration, stage)

                if iteration % opt.opacity_reset_interval == 0:
                    print("reset opacity")
                    gaussians.reset_opacity()

        if iteration < opt.iterations:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if (iteration in checkpoint_iterations):
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + f"_{stage}_" + str(iteration) + ".pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, prompt, guidance_scale, image_guidance_scale):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    gaussians.load_ply(args.ply_path)
    print(f"Loaded ply from {args.ply_path}")
    
    _ply_iter_dir = os.path.dirname(args.ply_path)
    if os.path.exists(os.path.join(_ply_iter_dir, "deformation.pth")):
        print(f"Loading deformation from ply directory: {_ply_iter_dir}")
        gaussians.load_model(_ply_iter_dir)
    else:
        print(f"No deformation.pth in {_ply_iter_dir}, falling back to original training checkpoint")
        point_cloud_dir = os.path.join(args.model_path, "point_cloud")
        iterations = []
        if os.path.exists(point_cloud_dir):
            for name in os.listdir(point_cloud_dir):
                if name.startswith("iteration_"):
                    try:
                        iterations.append(int(name.split("_")[1]))
                    except ValueError:
                        pass
        if not iterations:
            raise FileNotFoundError(f"No iteration folders found in {point_cloud_dir}")
        latest_iter = max(iterations)
        print(f"Loading deformation from: {point_cloud_dir}/iteration_{latest_iter}")
        gaussians.load_model(os.path.join(point_cloud_dir, f"iteration_{latest_iter}"))

    gaussians._deformation_table = torch.gt(torch.ones((gaussians.get_xyz.shape[0]), device="cuda"), 0)
    print("Loaded deformation field")
    timer.start()
    
    seed_everything(20211202)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16

    DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
    IP2P_SOURCE = "timbrooks/instruct-pix2pix"
    tokenizer    = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
    sys.stdout.isatty = lambda: False
    text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae")
    unet         = UNet3DConditionModel.from_pretrained_2d(IP2P_SOURCE, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae          = vae.to(device, dtype=torch_dtype)
    text_encoder = text_encoder.to(device, dtype=torch_dtype)
    unet         = unet.to(device, dtype=torch_dtype)
            
    ip2p = InstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler"),
    )
    print("Ready IP2P")

    scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, 800, timer, ip2p, prompt, guidance_scale, image_guidance_scale)


def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/sds_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',  'cameras': [scene.getTestCameras()[idx  % len(scene.getTestCameras())]  for idx in range(10, 5000, 299)]},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(10, 5000, 299)]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test   = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(stage + "/" + config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    except:
                        pass
                    l1_test   += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test   /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test,   iteration)
                    tb_writer.add_scalar(stage + "/" + config['name'] + '/loss_viewpoint - psnr',    psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points',    scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum() / scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram", scene.gaussians._deformation_accum.mean(dim=-1) / 100, iteration, max_bins=500)
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip',   type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations",       nargs="+", type=int, default=[3000, 5000])
    parser.add_argument("--save_iterations",       nargs="+", type=int, default=[100, 300, 500, 800, 1000, 1500, 2000, 3000, 5000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint",      type=str,  default=None)
    parser.add_argument("--expname",   type=str, default="")
    parser.add_argument("--configs",   type=str, default="")
    parser.add_argument("--ply_path",  type=str, default="")
    parser.add_argument("--prompt",    type=str, default="")
    parser.add_argument('--guidance_scale',       type=float, default=10.5)
    parser.add_argument('--image_guidance_scale', type=float, default=1.2)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, args.expname, args.prompt,
             args.guidance_scale, args.image_guidance_scale)

    print("\nEditing complete.")