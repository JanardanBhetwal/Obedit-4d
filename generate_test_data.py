#!/usr/bin/env python3
"""
Test Data Generator for Evaluation Pipeline
Helps create sample data structure for testing the evaluate_4dgs.py script.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


class TestDataGenerator:
    """Generates synthetic test data for evaluation pipeline."""
    
    @staticmethod
    def create_directories(base_path: str, experiment_name: str = 'test_experiment') -> dict:
        """
        Create the required directory structure.
        
        Args:
            base_path: Base output directory
            experiment_name: Name of experiment
            
        Returns:
            Dictionary with paths to created directories
        """
        dirs = {
            'experiment': Path(base_path) / experiment_name / 'test',
            'renders': Path(base_path) / experiment_name / 'test' / 'renders',
            'gt': Path(base_path) / experiment_name / 'test' / 'gt',
            'masks': Path(base_path) / experiment_name / 'test' / 'masks',
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {dir_path}")
        
        return dirs
    
    @staticmethod
    def generate_synthetic_image(
        width: int = 512,
        height: int = 512,
        image_type: str = 'random'
    ) -> Image.Image:
        """
        Generate a synthetic image.
        
        Args:
            width: Image width
            height: Image height
            image_type: Type of image ('random', 'gradient', 'noise')
            
        Returns:
            PIL Image object
        """
        if image_type == 'random':
            data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        
        elif image_type == 'gradient':
            # Create a gradient pattern
            x = np.linspace(0, 255, width)
            y = np.linspace(0, 255, height)
            xx, yy = np.meshgrid(x, y)
            
            r = xx.astype(np.uint8)
            g = yy.astype(np.uint8)
            b = ((xx + yy) / 2).astype(np.uint8)
            
            data = np.stack([r, g, b], axis=2)
        
        elif image_type == 'noise':
            # Perlin-like noise
            from PIL import Image as PILImage
            noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            data = noise
        
        else:
            raise ValueError(f"Unknown image type: {image_type}")
        
        return Image.fromarray(data, 'RGB')
    
    @staticmethod
    def generate_binary_mask(
        width: int = 512,
        height: int = 512,
        fg_ratio: float = 0.3
    ) -> Image.Image:
        """
        Generate a binary mask (1=background, 0=foreground).
        
        Args:
            width: Mask width
            height: Mask height
            fg_ratio: Ratio of foreground to background (0-1)
            
        Returns:
            PIL Image object (grayscale, 0-255)
        """
        # Create a simple circular foreground
        y, x = np.ogrid[:height, :width]
        center_x, center_y = width // 2, height // 2
        
        # Calculate radius based on fg_ratio
        radius = int(np.sqrt(fg_ratio * width * height / np.pi))
        
        # Create circular mask
        mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
        
        # Invert: 1=background, 0=foreground
        mask = (~mask).astype(np.uint8) * 255
        
        return Image.fromarray(mask, 'L')
    
    @staticmethod
    def generate_sequence(
        output_dir: str,
        num_frames: int = 30,
        width: int = 512,
        height: int = 512,
        image_type: str = 'random',
        create_masks: bool = True,
        create_offset_renders: bool = True
    ) -> None:
        """
        Generate a complete test sequence.
        
        Args:
            output_dir: Output directory path
            num_frames: Number of frames to generate
            width: Image width
            height: Image height
            image_type: Type of synthetic images
            create_masks: Whether to create mask images
            create_offset_renders: Whether to create slightly different renders (more realistic)
        """
        dirs = TestDataGenerator.create_directories(output_dir)
        
        print(f"\nGenerating {num_frames} frames ({width}x{height})...")
        
        for frame_idx in range(num_frames):
            # Generate ground truth
            gt_img = TestDataGenerator.generate_synthetic_image(width, height, image_type)
            gt_path = dirs['gt'] / f'frame_{frame_idx:04d}.png'
            gt_img.save(gt_path)
            
            # Generate render (slightly different if offset_renders enabled)
            if create_offset_renders:
                # Add slight noise to simulate editing imperfection
                gt_array = np.array(gt_img, dtype=np.float32)
                noise = np.random.normal(0, 10, gt_array.shape)
                render_array = np.clip(gt_array + noise, 0, 255).astype(np.uint8)
                render_img = Image.fromarray(render_array, 'RGB')
            else:
                render_img = gt_img.copy()
            
            render_path = dirs['renders'] / f'frame_{frame_idx:04d}.png'
            render_img.save(render_path)
            
            # Generate mask
            if create_masks:
                mask_img = TestDataGenerator.generate_binary_mask(width, height, fg_ratio=0.3)
                mask_path = dirs['masks'] / f'frame_{frame_idx:04d}.png'
                mask_img.save(mask_path)
            
            if (frame_idx + 1) % max(1, num_frames // 10) == 0:
                print(f"  Generated {frame_idx + 1}/{num_frames} frames")
        
        print(f"\n✓ Test data generated successfully!")
        print(f"  Location: {dirs['experiment']}")
        print(f"  Frames: {num_frames}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Directories: renders/, gt/, masks/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic test data for evaluation pipeline'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='Base output directory (default: output)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='test_experiment',
        help='Experiment name (default: test_experiment)'
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=30,
        help='Number of frames to generate (default: 30)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=512,
        help='Image width (default: 512)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=512,
        help='Image height (default: 512)'
    )
    parser.add_argument(
        '--image_type',
        type=str,
        choices=['random', 'gradient', 'noise'],
        default='random',
        help='Type of synthetic images (default: random)'
    )
    parser.add_argument(
        '--no_masks',
        action='store_true',
        help='Do not generate mask images'
    )
    parser.add_argument(
        '--no_offset',
        action='store_true',
        help='Do not add noise to renders (makes them identical to GT)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Test Data Generator for 4D Gaussian Splatting Evaluation")
    print("="*70)
    
    TestDataGenerator.generate_sequence(
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_type=args.image_type,
        create_masks=not args.no_masks,
        create_offset_renders=not args.no_offset
    )
    
    print("\nNext steps:")
    print(f"1. Run evaluation: python evaluate_4dgs.py \\")
    print(f"   --render_path {args.output_dir}/{args.experiment_name}/test/renders \\")
    print(f"   --gt_path {args.output_dir}/{args.experiment_name}/test/gt \\")
    print(f"   --mask_path {args.output_dir}/{args.experiment_name}/test/masks")
    print()


if __name__ == '__main__':
    main()
