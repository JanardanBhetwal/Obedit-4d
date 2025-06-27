# Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation
<em>CVPR 2025</em>
[[Paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Kwon_Efficient_Dynamic_Scene_Editing_via_4D_Gaussian-based_Static-Dynamic_Separation_CVPR_2025_paper.html) | [[arXiv]](https://arxiv.org/abs/2502.02091) | [[Project Page]](https://hanbyelcho.info/instruct-4dgs/)

**This is the official PyTorch implementation of the approach described in the following paper:**
> [Instruct-4DGS: Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation](https://arxiv.org/abs/2502.02091)\
> [Joohyun Kwon*](https://scholar.google.com/citations?user=WilZkLEAAAAJ&hl=en), [Hanbyel Cho*](https://hanbyelcho.info/) and [Junmo Kim†](https://scholar.google.com/citations?hl=en&user=GdQtWNQAAAAJ) (*Equal contribution, †Corresponding author)\
> IEEE/CVF Conference on Computer Vision and Pattern Recognition ([CVPR](https://cvpr.thecvf.com/Conferences/2025)), 2025

## 🏠 Overview
Recent 4D dynamic scene editing methods require editing thousands of 2D images used for dynamic scene synthesis and updating the entire scene with additional training loops, resulting in several hours of processing to edit a single dynamic scene. Therefore, these methods are not scalable with respect to the temporal dimension of the dynamic scene (i.e., the number of timesteps). In this work, we propose Instruct-4DGS, an efficient dynamic scene editing method that is more scalable in terms of temporal dimension. To achieve computational efficiency, we leverage a 4D Gaussian representation that models a 4D dynamic scene by combining static 3D Gaussians with a Hexplane-based deformation field, which captures dynamic information. We then perform editing solely on the static 3D Gaussians, which is the minimal but sufficient component required for visual editing. To resolve the misalignment between the edited 3D Gaussians and the deformation field, which may arise from the editing process, we introduce a refinement stage using a score distillation mechanism. Extensive editing results demonstrate that Instruct-4DGS is efficient, reducing editing time by more than half compared to existing methods while achieving high-quality edits that better follow user instructions.

![overall_framework](figs/instruct4dgs_overall.jpg)



## 📝 TODO List
- [ ] Refactor editing pipeline
- [ ] Polish the codes & update the doc



> **🔥 Heads-up! 🔥**
> We’re still cleaning the codebase and writing docs.  
> **Full release is scheduled for August-September 2025**



## 📖 Getting Started
### Installation
```bash
```


## 🔗 Citation
If you find our work helpful, please cite:
```bibtex
@InProceedings{Kwon_2025_CVPR,
    author    = {Kwon, Joohyun and Cho, Hanbyel and Kim, Junmo},
    title     = {Efficient Dynamic Scene Editing via 4D Gaussian-based Static-Dynamic Separation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {26855-26865}
}
```
