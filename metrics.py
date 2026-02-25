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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from pytorch_msssim import ms_ssim

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    # Get sorted lists of files in renders and gt directories
    render_files = sorted(os.listdir(renders_dir))
    gt_files = sorted(os.listdir(gt_dir))

    # Ensure the number of files in both directories is the same
    if len(render_files) != len(gt_files):
        raise ValueError(f"Mismatch in number of files: {len(render_files)} renders vs {len(gt_files)} ground truths")

    for render_file, gt_file in zip(render_files, gt_files):
        # Ensure the filenames match (or adjust this logic if needed)
        if not render_file.startswith("render_") or not gt_file.startswith("gt_"):
            raise ValueError(f"Filename mismatch: {render_file} and {gt_file}")

        # Open the images
        render = Image.open(renders_dir / render_file)
        gt = Image.open(gt_dir / gt_file)

        # Convert images to tensors and append to lists
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(render_file)  # Use render filenames as image names

    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            # Check if test directory exists
            if not test_dir.exists():
                raise FileNotFoundError(f"Test directory not found: {test_dir}")

            # Assuming renders and gt are directly inside the test folder
            renders_dir = test_dir / "renders"
            gt_dir = test_dir / "gt"

            # Check if renders and gt directories exist
            if not renders_dir.exists() or not gt_dir.exists():
                raise FileNotFoundError(f"Renders or GT directory not found in: {test_dir}")

            renders, gts, image_names = readImages(renders_dir, gt_dir)

            ssims = []
            psnrs = []
            lpipss = []
            lpipsa = []
            ms_ssims = []
            Dssims = []
            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                ssims.append(ssim(renders[idx], gts[idx]))
                psnrs.append(psnr(renders[idx], gts[idx]))
                lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                ms_ssims.append(ms_ssim(renders[idx], gts[idx], data_range=1, size_average=True))
                lpipsa.append(lpips(renders[idx], gts[idx], net_type='alex'))
                Dssims.append((1 - ms_ssims[-1]) / 2)

            print("Scene: ", scene_dir,  "SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("Scene: ", scene_dir,  "PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("Scene: ", scene_dir,  "LPIPS-vgg: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
            print("Scene: ", scene_dir,  "LPIPS-alex: {:>12.7f}".format(torch.tensor(lpipsa).mean(), ".5"))
            print("Scene: ", scene_dir,  "MS-SSIM: {:>12.7f}".format(torch.tensor(ms_ssims).mean(), ".5"))
            print("Scene: ", scene_dir,  "D-SSIM: {:>12.7f}".format(torch.tensor(Dssims).mean(), ".5"))

            full_dict[scene_dir].update({"SSIM": torch.tensor(ssims).mean().item(),
                                         "PSNR": torch.tensor(psnrs).mean().item(),
                                         "LPIPS-vgg": torch.tensor(lpipss).mean().item(),
                                         "LPIPS-alex": torch.tensor(lpipsa).mean().item(),
                                         "MS-SSIM": torch.tensor(ms_ssims).mean().item(),
                                         "D-SSIM": torch.tensor(Dssims).mean().item()})

            per_view_dict[scene_dir].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                             "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                             "LPIPS-vgg": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                             "LPIPS-alex": {name: lp for lp, name in zip(torch.tensor(lpipsa).tolist(), image_names)},
                                             "MS-SSIM": {name: lp for lp, name in zip(torch.tensor(ms_ssims).tolist(), image_names)},
                                             "D-SSIM": {name: lp for lp, name in zip(torch.tensor(Dssims).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir)
            raise e

# ...existing code...

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
