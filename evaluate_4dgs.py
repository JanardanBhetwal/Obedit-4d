#!/usr/bin/env python3
"""
Comprehensive Evaluation Pipeline for 4D Gaussian Splatting Editing
Computes PSNR, SSIM, LPIPS, Temporal SSIM, Background PSNR, and CLIP Image Similarity.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
import lpips
from transformers import CLIPProcessor, CLIPModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageLoader:
    """Handles loading and normalizing images from disk."""
    
    @staticmethod
    def load_image(image_path: str) -> torch.Tensor:
        """
        Load a single image and convert to [0, 1] float tensor.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tensor of shape (3, H, W) in [0, 1] range
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
        return img_tensor
    
    @staticmethod
    def load_mask(mask_path: str) -> torch.Tensor:
        """
        Load a binary mask image.
        
        Args:
            mask_path: Path to mask file
            
        Returns:
            Tensor of shape (1, H, W) with binary values {0, 1}
        """
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        # Threshold to ensure binary: 1=background, 0=foreground
        mask_array = (mask_array > 0.5).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # (1, H, W)
        return mask_tensor
    
    @staticmethod
    def natural_sort_key(filename: str) -> Tuple:
        """
        Generate a sort key for natural sorting (e.g., frame_1, frame_2, ..., frame_10).
        
        Args:
            filename: Filename to extract sort key from
            
        Returns:
            Tuple of (string_parts, integer_parts) for sorting
        """
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        
        return tuple(convert(c) for c in re.split(r'([0-9]+)', filename))
    
    @staticmethod
    def get_sorted_image_paths(directory: str) -> List[str]:
        """
        Get sorted list of image paths from directory.
        
        Args:
            directory: Path to directory containing images
            
        Returns:
            List of absolute paths to images, sorted naturally
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        images = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        
        if not images:
            raise FileNotFoundError(f"No images found in {directory}")
        
        images.sort(key=lambda x: ImageLoader.natural_sort_key(os.path.basename(x)))
        return images


class MetricsCompute:
    """Computes various evaluation metrics."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize metric computation modules.
        
        Args:
            device: Device to use for computation ('cuda' or 'cpu')
        """
        self.device = device
        logger.info(f"Using device: {device}")
        
        # PSNR and SSIM
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        
        # LPIPS (AlexNet)
        self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
        
        # CLIP for image-to-image similarity
        logger.info("Loading CLIP model (openai/clip-vit-base-patch32)...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def compute_psnr(self, render: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Compute PSNR between rendered and ground truth images.
        
        Args:
            render: Rendered image (C, H, W)
            gt: Ground truth image (C, H, W)
            
        Returns:
            PSNR value in dB
        """
        render = render.unsqueeze(0).to(self.device)  # (1, C, H, W)
        gt = gt.unsqueeze(0).to(self.device)
        
        psnr_val = self.psnr(render, gt).item()
        return psnr_val
    
    def compute_ssim(self, render: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Compute SSIM between rendered and ground truth images.
        
        Args:
            render: Rendered image (C, H, W)
            gt: Ground truth image (C, H, W)
            
        Returns:
            SSIM value in [0, 1]
        """
        render = render.unsqueeze(0).to(self.device)  # (1, C, H, W)
        gt = gt.unsqueeze(0).to(self.device)
        
        ssim_val = self.ssim(render, gt).item()
        return ssim_val
    
    def compute_lpips(self, render: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Compute LPIPS (learned perceptual image patch similarity).
        
        Args:
            render: Rendered image (C, H, W)
            gt: Ground truth image (C, H, W)
            
        Returns:
            LPIPS value
        """
        render = render.unsqueeze(0).to(self.device) * 2 - 1  # Normalize to [-1, 1]
        gt = gt.unsqueeze(0).to(self.device) * 2 - 1
        
        with torch.no_grad():
            lpips_val = self.lpips_fn(render, gt).item()
        
        return lpips_val
    
    def compute_temporal_ssim(self, frame_t: torch.Tensor, frame_t1: torch.Tensor) -> float:
        """
        Compute temporal SSIM between consecutive frames (temporal consistency).
        
        Args:
            frame_t: Frame at time t (C, H, W)
            frame_t1: Frame at time t+1 (C, H, W)
            
        Returns:
            Temporal SSIM value in [0, 1]
        """
        frame_t = frame_t.unsqueeze(0).to(self.device)  # (1, C, H, W)
        frame_t1 = frame_t1.unsqueeze(0).to(self.device)
        
        t_ssim = self.ssim(frame_t, frame_t1).item()
        return t_ssim
    
    def compute_bg_psnr(
        self,
        render: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor
    ) -> float:
        """
        Compute PSNR on background pixels only (where mask == 1).
        
        Args:
            render: Rendered image (C, H, W)
            gt: Ground truth image (C, H, W)
            mask: Binary mask (1, H, W) where 1=background, 0=foreground
            
        Returns:
            Background PSNR value in dB
        """
        mask = mask.to(self.device)
        render = render.to(self.device)
        gt = gt.to(self.device)
        
        # Apply mask: keep only background pixels
        bg_render = render * mask
        bg_gt = gt * mask
        
        # Compute PSNR on masked region
        mse = torch.mean((bg_render - bg_gt) ** 2)
        
        if mse == 0:
            return float('inf')
        
        psnr_val = 20 * torch.log10(torch.tensor(1.0, device=self.device) / torch.sqrt(mse))
        return psnr_val.item()
    
    def compute_clip_similarity(self, render: torch.Tensor, gt: torch.Tensor) -> float:
        """
        Compute CLIP image-to-image cosine similarity.
        
        Args:
            render: Rendered image (C, H, W) in [0, 1]
            gt: Ground truth image (C, H, W) in [0, 1]
            
        Returns:
            Cosine similarity in [-1, 1], typically [0, 1]
        """
        # Convert tensor back to PIL image for CLIP processor
        render_pil = transforms.ToPILImage()(render.cpu())
        gt_pil = transforms.ToPILImage()(gt.cpu())
        
        # Process images with CLIP processor
        render_inputs = self.clip_processor(
            images=render_pil,
            return_tensors="pt"
        ).to(self.device)
        gt_inputs = self.clip_processor(
            images=gt_pil,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            render_features = self.clip_model.get_image_features(**render_inputs)
            gt_features = self.clip_model.get_image_features(**gt_inputs)
            
            # Normalize features
            render_features = F.normalize(render_features, dim=-1)
            gt_features = F.normalize(gt_features, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.mm(render_features, gt_features.t()).item()
        
        return similarity


class Evaluator4DGS:
    """Main evaluation pipeline for 4D Gaussian Splatting."""
    
    def __init__(
        self,
        render_path: str,
        gt_path: str,
        mask_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the evaluator.
        
        Args:
            render_path: Directory containing rendered frames
            gt_path: Directory containing ground truth frames
            mask_path: Optional directory containing binary masks
            device: Device for computation
        """
        self.render_path = Path(render_path)
        self.gt_path = Path(gt_path)
        self.mask_path = Path(mask_path) if mask_path else None
        self.device = device
        
        # Validate paths
        if not self.render_path.exists():
            raise FileNotFoundError(f"Render path not found: {render_path}")
        if not self.gt_path.exists():
            raise FileNotFoundError(f"GT path not found: {gt_path}")
        if self.mask_path and not self.mask_path.exists():
            logger.warning(f"Mask path not found: {mask_path}. BG-PSNR will be skipped.")
            self.mask_path = None
        
        self.image_loader = ImageLoader()
        self.metrics = MetricsCompute(device=device)
        
        # Load image lists
        logger.info("Loading image paths...")
        self.render_images = self.image_loader.get_sorted_image_paths(str(self.render_path))
        self.gt_images = self.image_loader.get_sorted_image_paths(str(self.gt_path))
        
        if len(self.render_images) != len(self.gt_images):
            raise ValueError(
                f"Mismatch in number of images: "
                f"{len(self.render_images)} renders vs {len(self.gt_images)} GT"
            )
        
        logger.info(f"Loaded {len(self.render_images)} image pairs for evaluation")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Dictionary of averaged metrics
        """
        logger.info("Starting evaluation pipeline...")
        
        psnr_values = []
        ssim_values = []
        lpips_values = []
        temporal_ssim_values = []
        bg_psnr_values = []
        clip_values = []
        
        prev_render = None
        
        for idx, (render_path, gt_path) in enumerate(
            zip(self.render_images, self.gt_images)
        ):
            if (idx + 1) % max(1, len(self.render_images) // 10) == 0:
                logger.info(f"Processing frame {idx + 1}/{len(self.render_images)}")
            
            # Load images
            render = self.image_loader.load_image(render_path)
            gt = self.image_loader.load_image(gt_path)
            
            # PSNR
            psnr_val = self.metrics.compute_psnr(render, gt)
            psnr_values.append(psnr_val)
            
            # SSIM
            ssim_val = self.metrics.compute_ssim(render, gt)
            ssim_values.append(ssim_val)
            
            # LPIPS
            lpips_val = self.metrics.compute_lpips(render, gt)
            lpips_values.append(lpips_val)
            
            # CLIP Similarity
            clip_val = self.metrics.compute_clip_similarity(render, gt)
            clip_values.append(clip_val)
            
            # Temporal SSIM
            if prev_render is not None:
                t_ssim = self.metrics.compute_temporal_ssim(prev_render, render)
                temporal_ssim_values.append(t_ssim)
            
            # Background PSNR
            if self.mask_path:
                mask_filename = os.path.basename(render_path)
                mask_path = os.path.join(str(self.mask_path), mask_filename)
                if os.path.exists(mask_path):
                    mask = self.image_loader.load_mask(mask_path)
                    bg_psnr = self.metrics.compute_bg_psnr(render, gt, mask)
                    bg_psnr_values.append(bg_psnr)
            
            prev_render = render
        
        # Compute averages
        results = {
            'PSNR': np.mean(psnr_values),
            'SSIM': np.mean(ssim_values),
            'LPIPS': np.mean(lpips_values),
            'CLIP_Similarity': np.mean(clip_values),
            'Temporal_SSIM': np.mean(temporal_ssim_values) if temporal_ssim_values else None,
            'BG_PSNR': np.mean(bg_psnr_values) if bg_psnr_values else None,
        }
        
        # Remove None values
        results = {k: v for k, v in results.items() if v is not None}
        
        logger.info("Evaluation complete!")
        return results
    
    def print_results(self, results: Dict[str, float]) -> None:
        """Print results in a formatted table."""
        print("\n" + "="*60)
        print("4D Gaussian Splatting Evaluation Results")
        print("="*60)
        
        # Sort results for consistent display
        results_ordered = OrderedDict([
            ('PSNR', results.get('PSNR')),
            ('SSIM', results.get('SSIM')),
            ('LPIPS', results.get('LPIPS')),
            ('Temporal_SSIM', results.get('Temporal_SSIM')),
            ('BG_PSNR', results.get('BG_PSNR')),
            ('CLIP_Similarity', results.get('CLIP_Similarity')),
        ])
        
        for metric, value in results_ordered.items():
            if value is not None:
                print(f"{metric:.<40} {value:.6f}")
        
        print("="*60 + "\n")
    
    def save_results(self, results: Dict[str, float], output_dir: str) -> None:
        """
        Save results to JSON file.
        
        Args:
            results: Dictionary of metrics
            output_dir: Directory to save metrics.json
        """
        output_path = Path(output_dir) / "metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any numpy types to Python native types for JSON serialization
        results_serializable = {
            k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v
            for k, v in results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate 4D Gaussian Splatting Editing Quality"
    )
    parser.add_argument(
        '--render_path',
        type=str,
        required=True,
        help='Path to directory containing rendered frames'
    )
    parser.add_argument(
        '--gt_path',
        type=str,
        required=True,
        help='Path to directory containing ground truth frames'
    )
    parser.add_argument(
        '--mask_path',
        type=str,
        default=None,
        help='Path to directory containing binary masks for BG-PSNR (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for computation (cuda or cpu)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save metrics.json. Defaults to parent of render_path.'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.render_path).parent)
    
    try:
        # Initialize and run evaluator
        evaluator = Evaluator4DGS(
            render_path=args.render_path,
            gt_path=args.gt_path,
            mask_path=args.mask_path,
            device=args.device
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Display results
        evaluator.print_results(results)
        
        # Save results
        evaluator.save_results(results, args.output_dir)
        
        logger.info("Evaluation pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
