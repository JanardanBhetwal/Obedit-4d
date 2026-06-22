# OBEdit-4D: Object-Centric Editing of Dynamic Scenes via Masked Diffusion Guidance

## Overview

Existing instruction-guided 4D editing methods edit the _entire_ rendered image at every optimization step. When a user asks to change only one object, "put the man in armour", "turn the man into a cyborg", the diffusion model still touches the surrounding scene, causing the edit to gradually bleed into the background, flicker across frames, and erode the identity of everything that wasn't supposed to change.

OBEdit-4D fixes this by constraining every diffusion edit to the target object before it ever reaches the Gaussian update. A binary object mask is generated automatically from the same text prompt driving the edit, and the diffusion output is blended back against the original render using that mask — so background pixels are never touched, no matter what the diffusion model predicts for them. We then apply a short score-distillation refinement pass to clean up any residual cross-frame misalignment introduced once the edited canonical Gaussians are pushed back through the (frozen) deformation field.

The result is an edit that stays where you pointed it, across every camera view and every timestep, without retraining anything.

---

## Getting Started

### Installation

1. **Install base dependencies**

   Follow the installation guide in **[4D Gaussian Splatting (4DGS)](https://github.com/hustvl/4DGaussians)** to set up the core environment (PyTorch, CUDA rasterizer, `simple-knn`, etc.).

2. **Install editing dependencies**

   ```bash
   conda activate Gaussians4D
   pip install -r requirements.txt
   ```

3. **Install Grounded-SAM**

   Mask generation uses [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) (Grounding DINO + SAM). Follow their setup instructions and download the Grounding DINO and SAM checkpoints, the mask generation script expects their paths to be set before running.

### Data Preparation

Follow the official 4DGS guide to process your multi-view capture and obtain a trained 4DGS checkpoint. As with Instruct-4DGS, you'll need:

1. **Initial point cloud**
   - File: `points3D_downsample2.ply`
   - Destination: `./data/<dataset>/<scene>/`

2. **Trained 4DGS output**
   - Directory: `point_cloud/`
   - Destination: `./output/<dataset>/<scene>/`

---

<!-- ## Editing

Editing happens in two stages: mask generation + masked canonical editing, followed by temporal refinement.

### 1. Generate the object mask

Given the same text prompt you intend to edit with, generate a binary mask of the target object at the canonical timestep:

```bash
python edit_3d_mv.py \
  --configs ./arguments/dynerf/cook_spinach.py \
  -s ./data/dynerf/cook_spinach \
  --model_path ./output/dynerf/cook_spinach \
  --prompt "turn the man into a cyborg" \
  --box_threshold 0.3 \
  --text_threshold 0.25
```

This produces per-view masks and immediately performs the masked canonical edit — only pixels inside the mask are updated; everything else is restored from the original render before the Gaussian update.

> If the mask looks incomplete or noisy (thin limbs, motion blur, heavy occlusion), try lowering `--box_threshold` slightly or rephrasing the prompt to be more specific about the target (e.g. "the man's jacket" instead of "the man"). -->

### Full Editing Pipeline

![Pipeline](frame1.png)

Run the editing pipeline using the shell script below:

```bash
# run_instruct_4dgs.sh [dataset] [scene_name] [prompt] [guidance_scale] [image_guidance_scale]
bash run_instruct_4dgs.sh dynerf cook_spinach "turn the man into a cyborg" 10.5 1.2
```

### Rendering results

```bash
python render_edited4d_mv.py \
  --configs ./arguments/dynerf/cook_spinach.py \
  --ply_path "./output/dynerf/cook_spinach/point_cloud_refine/turn the man into a cyborg/iteration_800/point_cloud.ply" \
  -s ./data/dynerf/cook_spinach \
  --model_path ./output/dynerf/cook_spinach
```

---

## Acknowledgement

This work is built on top of **[Instruct-4DGS](https://hanbyelcho.info/instruct-4dgs/)** (Kwon, Cho, and Kim, CVPR 2025), which we gratefully use as our base 4D editing architecture, along with **[4D Gaussian Splatting](https://github.com/hustvl/4DGaussians)**, **[Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)**, and **[Instruct-4D-to-4D](https://github.com/Friedrich-M/Instruct-4D-to-4D)**. We are grateful to all of these projects for their excellent work and for making their code available.

We also thank the Department of Electronics and Computer Engineering, Pulchowk Campus, for institutional support, and the project committee for guidance throughout this work.

## 🔗 Citation

If you find this work helpful, please cite this work as:

```bibtex
@misc{bhetwal2025obedit4d,
  title  = {OBEdit-4D: Object-Centric Editing of 3D Dynamic Scenes via Text Prompts},
  author = {Bhetwal, Janardan and Singh, Baibhav and Shrestha, Bhraman and Prasad, Prakash Chandra and Jaiswal, Anku},
  year   = {2026}
}

```
