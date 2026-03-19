# Evaluation Pipeline for 4D Gaussian Splatting Editing

## Overview

`evaluate_4dgs.py` is a comprehensive end-to-end evaluation pipeline for measuring the quality and temporal consistency of edited 4D Gaussian Splatting scenes. It computes multiple complementary metrics to assess editing quality from different perspectives.

## Metrics Computed

### 1. **PSNR** (Peak Signal-to-Noise Ratio)

- Measures pixel-level similarity between rendered and ground truth frames
- Higher is better (typically 20-40 dB for good quality)
- Uses `torchmetrics` implementation with data range [0, 1]

### 2. **SSIM** (Structural Similarity Index Measure)

- Evaluates structural and perceptual similarity
- Range: [0, 1], where 1 is perfect similarity
- More aligned with human visual perception than PSNR
- Uses `torchmetrics` implementation

### 3. **LPIPS** (Learned Perceptual Image Patch Similarity)

- Computes learned perceptual distance using AlexNet backbone
- Range: [0, 1], lower is better
- Highly correlated with human perceptual judgments
- Uses official `lpips` library

### 4. **Temporal SSIM (t-SSIM)**

- Computes SSIM between consecutive rendered frames
- Measures temporal consistency and flickering
- High values indicate smooth, temporally coherent editing
- Averaged across all consecutive frame pairs

### 5. **Background PSNR (BG-PSNR)**

- Isolates background pixels using binary masks (1=background, 0=foreground)
- Computes PSNR only on background region
- Ensures editing artifacts didn't corrupt the static background
- Requires mask images in the specified mask directory

### 6. **CLIP Image-to-Image Similarity**

- Uses OpenAI's CLIP vision encoder to compute semantic similarity
- Compares feature-space embeddings between rendered and GT frames
- Range: typically [0, 1], higher is better
- Captures high-level semantic consistency

## Installation

### Option 1: Install from requirements file

```bash
pip install -r requirements_evaluation.txt
```

### Option 2: Install individual packages

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 torchmetrics>=0.11.0 lpips>=0.1.4 Pillow>=9.0.0 transformers>=4.30.0 scipy>=1.9.0
```

### Dependencies

- **PyTorch**: For tensor operations and model inference
- **torchmetrics**: For PSNR and SSIM computation
- **lpips**: For perceptual distance metrics
- **transformers**: For CLIP model and processor
- **Pillow**: For image loading and conversion
- **NumPy**: For numerical operations
- **SciPy**: Required dependency of lpips

## Expected Directory Structure

```
output/
└── <experiment_name>/
    ├── test/
    │   ├── renders/      # Rendered frames from edited 4D Gaussians (.png/.jpg)
    │   ├── gt/           # Corresponding ground truth / original frames
    │   └── masks/        # (Optional) Binary masks for BG-PSNR computation
    └── metrics.json      # Output file with computed metrics
```

### Naming Convention

- Image files must be naturally sortable (e.g., `frame_0.png`, `frame_1.png`, ..., `frame_10.png`)
- The script automatically sorts files to ensure temporal order
- All three directories (renders, gt, and optionally masks) must have the same number of files
- Corresponding files must have the same filename

## Usage

### Basic Usage

```bash
python evaluate_4dgs.py \
  --render_path output/experiment_name/test/renders \
  --gt_path output/experiment_name/test/gt
```

### With Masks (for BG-PSNR)

```bash
python evaluate_4dgs.py \
  --render_path output/experiment_name/test/renders \
  --gt_path output/experiment_name/test/gt \
  --mask_path output/experiment_name/test/masks
```

### Specifying Output Directory

```bash
python evaluate_4dgs.py \
  --render_path output/experiment_name/test/renders \
  --gt_path output/experiment_name/test/gt \
  --output_dir output/experiment_name/test
```

### Using CPU (if CUDA unavailable)

```bash
python evaluate_4dgs.py \
  --render_path output/experiment_name/test/renders \
  --gt_path output/experiment_name/test/gt \
  --device cpu
```

## Output

The script produces two types of output:

### 1. Terminal Output

A formatted table displayed to the console:

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

### 2. JSON Output

Saved as `metrics.json` in the output directory:

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

## Implementation Details

### Image Loading and Normalization

- Images are loaded as RGB using Pillow
- Pixel values are normalized to [0, 1] float range
- Images are converted to (C, H, W) tensor format
- Natural sorting ensures temporal order (e.g., frame_1 < frame_2 < frame_10)

### Metric Computation

- **PSNR/SSIM**: Computed using `torchmetrics` with data range [0, 1]
- **LPIPS**: Input images are scaled to [-1, 1] range as required by the AlexNet backbone
- **Temporal SSIM**: Computed between consecutive rendered frames only
- **BG-PSNR**: Masked pixel-wise MSE computation
- **CLIP Similarity**: Normalized cosine similarity in embedding space

### Device Handling

- Automatically detects CUDA availability
- Falls back to CPU if CUDA unavailable
- All tensors and models moved to specified device
- Supports both GPU and CPU inference

### Error Handling

- Validates path existence before processing
- Checks for matching number of render and GT images
- Provides informative error messages
- Gracefully skips BG-PSNR if masks unavailable
- Logs progress at 10 intervals during evaluation

## Performance Considerations

### Memory Usage

- Stores one image pair in memory at a time
- CLIP model inference is the most memory-intensive operation (~4 GB VRAM)
- Background PSNR computation is very efficient

### Speed

- For a 300-frame sequence:
  - Typical runtime on V100 GPU: ~2-3 minutes
  - CPU runtime: ~15-20 minutes (depends on system)
- Bottleneck: CLIP model inference (~0.3-0.5s per image)

### Optimization Tips

- Use GPU if available (50-100x speedup over CPU)
- Pre-process images to smaller resolution if full resolution not needed
- Consider running on subset of frames for quick validation

## API Reference

### `ImageLoader` Class

```python
@staticmethod
def load_image(image_path: str) -> torch.Tensor
    """Load image and return normalized (3, H, W) tensor in [0, 1] range"""

@staticmethod
def load_mask(mask_path: str) -> torch.Tensor
    """Load binary mask and return (1, H, W) tensor with values {0, 1}"""

@staticmethod
def get_sorted_image_paths(directory: str) -> List[str]
    """Get naturally sorted list of image paths from directory"""
```

### `MetricsCompute` Class

```python
def compute_psnr(render: torch.Tensor, gt: torch.Tensor) -> float
    """Compute PSNR value (dB)"""

def compute_ssim(render: torch.Tensor, gt: torch.Tensor) -> float
    """Compute SSIM value in [0, 1]"""

def compute_lpips(render: torch.Tensor, gt: torch.Tensor) -> float
    """Compute LPIPS value"""

def compute_temporal_ssim(frame_t: torch.Tensor, frame_t1: torch.Tensor) -> float
    """Compute temporal consistency between consecutive frames"""

def compute_bg_psnr(render: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> float
    """Compute PSNR on background pixels only"""

def compute_clip_similarity(render: torch.Tensor, gt: torch.Tensor) -> float
    """Compute CLIP embedding cosine similarity"""
```

### `Evaluator4DGS` Class

```python
def __init__(render_path, gt_path, mask_path=None, device='cuda')
    """Initialize evaluator with image paths"""

def evaluate() -> Dict[str, float]
    """Run complete evaluation pipeline, return averaged metrics"""

def print_results(results: Dict[str, float]) -> None
    """Print formatted results table"""

def save_results(results: Dict[str, float], output_dir: str) -> None
    """Save metrics.json to output directory"""
```

## Troubleshooting

### Issue: "CUDA out of memory"

- **Solution**: Use `--device cpu` or process fewer frames at a time
- Reduce image resolution before evaluation if possible

### Issue: "No images found in directory"

- **Solution**: Verify that image files have valid extensions (.png, .jpg, .jpeg, .bmp)
- Check that the directory path is correct

### Issue: "Mismatch in number of images"

- **Solution**: Ensure render and GT directories have the same number of images
- Verify all filenames match between directories

### Issue: Slow evaluation

- **Solution**: Use GPU instead of CPU
- CLIP inference is the bottleneck; consider using CPU for other metrics only

### Issue: BG-PSNR not computed

- **Solution**: Ensure mask directory is provided and has matching filenames
- Check that mask values are normalized (0-255 or 0.0-1.0)

## Citation

If you use this evaluation pipeline, please cite the relevant papers:

- **PSNR/SSIM**: Standard image quality metrics
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018
- **CLIP**: Radford et al., "Learning Transferable Models for Unsupervised Domain Adaptation", OpenAI
- **Torchmetrics**: Torchmetrics contributors

## License

This evaluation pipeline is provided as part of the Instruct-4DGS project.

## Authors

Evaluation pipeline designed for the Instruct-4DGS project.
