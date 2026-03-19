# Quick Reference: 4D Gaussian Splatting Evaluation Pipeline

## Installation

### 1. Validate Environment (Optional)

```bash
python setup_evaluation.py
```

### 2. Install Dependencies

```bash
# Option A: From requirements file (recommended)
pip install -r requirements_evaluation.txt

# Option B: Individual packages
pip install torch>=2.0.0 torchvision>=0.15.0 torchmetrics>=0.11.0 \
            lpips>=0.1.4 Pillow>=9.0.0 transformers>=4.30.0 scipy>=1.9.0
```

## Usage

### Basic Evaluation

```bash
python evaluate_4dgs.py \
  --render_path output/exp/test/renders \
  --gt_path output/exp/test/gt
```

### With Background Masks

```bash
python evaluate_4dgs.py \
  --render_path output/exp/test/renders \
  --gt_path output/exp/test/gt \
  --mask_path output/exp/test/masks
```

### Specify Device and Output

```bash
python evaluate_4dgs.py \
  --render_path output/exp/test/renders \
  --gt_path output/exp/test/gt \
  --device cuda \
  --output_dir output/exp/test
```

## Output Format

### Terminal Output

```
============================================================
4D Gaussian Splatting Evaluation Results
============================================================
PSNR............................ 25.123456
SSIM............................ 0.875432
LPIPS........................... 0.089234
Temporal_SSIM................... 0.923456
BG_PSNR......................... 28.456789
CLIP_Similarity................. 0.956234
============================================================
```

### JSON Output (metrics.json)

```json
{
  "PSNR": 25.123456,
  "SSIM": 0.875432,
  "LPIPS": 0.089234,
  "Temporal_SSIM": 0.923456,
  "BG_PSNR": 28.456789,
  "CLIP_Similarity": 0.956234
}
```

## Metrics Summary

| Metric          | Range  | Better | Meaning                                 |
| --------------- | ------ | ------ | --------------------------------------- |
| PSNR            | 0-∞ dB | Higher | Pixel-level similarity (typical: 20-40) |
| SSIM            | [0, 1] | Higher | Structural similarity (1=perfect)       |
| LPIPS           | [0, ∞] | Lower  | Perceptual distance (AlexNet)           |
| Temporal SSIM   | [0, 1] | Higher | Frame-to-frame consistency              |
| BG-PSNR         | 0-∞ dB | Higher | Background region quality               |
| CLIP Similarity | [0, 1] | Higher | Semantic consistency                    |

## Python API

### Simple Usage

```python
from evaluate_4dgs import Evaluator4DGS

evaluator = Evaluator4DGS(
    render_path='output/exp/test/renders',
    gt_path='output/exp/test/gt',
    mask_path='output/exp/test/masks'
)

results = evaluator.evaluate()
evaluator.print_results(results)
evaluator.save_results(results, 'output/exp/test')
```

### Individual Metrics

```python
from evaluate_4dgs import MetricsCompute, ImageLoader

metrics = MetricsCompute(device='cuda')
loader = ImageLoader()

render = loader.load_image('path/to/render.png')
gt = loader.load_image('path/to/gt.png')
mask = loader.load_mask('path/to/mask.png')

psnr = metrics.compute_psnr(render, gt)
ssim = metrics.compute_ssim(render, gt)
lpips_val = metrics.compute_lpips(render, gt)
t_ssim = metrics.compute_temporal_ssim(render, gt)
bg_psnr = metrics.compute_bg_psnr(render, gt, mask)
clip_sim = metrics.compute_clip_similarity(render, gt)
```

## Directory Structure

```
output/
└── experiment_name/
    └── test/
        ├── renders/          # Edited 4D Gaussian renders (.png/.jpg)
        ├── gt/               # Ground truth / original frames
        ├── masks/            # Binary masks (optional, 1=bg, 0=fg)
        └── metrics.json      # Output metrics (created by script)
```

### Image File Naming

- Must be naturally sortable: `frame_0.png`, `frame_1.png`, ..., `frame_10.png`
- Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`
- Must match across renders/, gt/, and masks/ directories

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip install -r requirements_evaluation.txt
```

### "CUDA out of memory"

```bash
python evaluate_4dgs.py ... --device cpu
```

### "Mismatch in number of images"

- Ensure renders/ and gt/ have same number of files
- Check all filenames match

### Slow evaluation

- Use GPU: `--device cuda` instead of `cpu`
- GPU is 50-100x faster than CPU

## Performance

| Component | Time per Frame |
| --------- | -------------- |
| PSNR/SSIM | ~1-2 ms        |
| LPIPS     | ~50-100 ms     |
| CLIP      | ~200-500 ms    |
| BG-PSNR   | <1 ms          |
| Total     | ~250-600 ms    |

For 300-frame sequence: **2-5 minutes on GPU, 15-20 minutes on CPU**

## Files Provided

| File                          | Purpose                               |
| ----------------------------- | ------------------------------------- |
| `evaluate_4dgs.py`            | Main evaluation pipeline (900+ lines) |
| `requirements_evaluation.txt` | Dependencies for evaluation           |
| `EVALUATION_GUIDE.md`         | Comprehensive documentation           |
| `example_evaluation.py`       | Usage examples and custom pipelines   |
| `setup_evaluation.py`         | Environment validation script         |
| `QUICK_REFERENCE.md`          | This file                             |

## Common Workflows

### Single Experiment

```bash
python evaluate_4dgs.py \
  --render_path output/exp1/test/renders \
  --gt_path output/exp1/test/gt
```

### Batch Evaluation

```python
from evaluate_4dgs import Evaluator4DGS
import json

experiments = ['exp1', 'exp2', 'exp3']
results = {}

for exp in experiments:
    evaluator = Evaluator4DGS(
        render_path=f'output/{exp}/test/renders',
        gt_path=f'output/{exp}/test/gt'
    )
    results[exp] = evaluator.evaluate()

with open('results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Focus on Specific Metrics

See `example_evaluation.py` for custom pipeline examples.

## Support

For issues or questions:

1. Check EVALUATION_GUIDE.md for detailed documentation
2. Review example_evaluation.py for usage patterns
3. Run setup_evaluation.py to validate environment
4. Check console output for detailed error messages

## Citation

If you use this evaluation pipeline in your work:

```bibtex
@misc{instruct4dgs_evaluation,
  title={Evaluation Pipeline for Text-Guided 4D Gaussian Splatting Editing},
  note={Part of Instruct-4DGS project}
}
```

And cite the relevant papers for individual metrics:

- LPIPS: Zhang et al., CVPR 2018
- CLIP: Radford et al., OpenAI
- PyTorchMetrics: Contributors
