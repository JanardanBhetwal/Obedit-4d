#!/usr/bin/env python3
"""
Example usage of the evaluate_4dgs.py module.
Demonstrates both CLI usage and programmatic API usage.
"""

import sys
from pathlib import Path

# Example 1: Using the CLI
def example_cli():
    """Example of using the evaluation pipeline via CLI."""
    print("="*60)
    print("Example 1: CLI Usage")
    print("="*60)
    print("""
# Basic evaluation
python evaluate_4dgs.py \\
  --render_path output/experiment_name/test/renders \\
  --gt_path output/experiment_name/test/gt

# With background masks for BG-PSNR
python evaluate_4dgs.py \\
  --render_path output/experiment_name/test/renders \\
  --gt_path output/experiment_name/test/gt \\
  --mask_path output/experiment_name/test/masks

# Specify output directory and device
python evaluate_4dgs.py \\
  --render_path output/experiment_name/test/renders \\
  --gt_path output/experiment_name/test/gt \\
  --output_dir output/experiment_name/test \\
  --device cuda
    """)


# Example 2: Using the Python API
def example_api():
    """Example of using the evaluation pipeline via Python API."""
    print("\n" + "="*60)
    print("Example 2: Python API Usage")
    print("="*60)
    
    try:
        from evaluate_4dgs import Evaluator4DGS
        
        # Initialize evaluator
        evaluator = Evaluator4DGS(
            render_path='output/experiment_name/test/renders',
            gt_path='output/experiment_name/test/gt',
            mask_path='output/experiment_name/test/masks',
            device='cuda'
        )
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Display results
        evaluator.print_results(results)
        
        # Save results
        evaluator.save_results(results, 'output/experiment_name/test')
        
        return results
        
    except Exception as e:
        print(f"Error in API example: {e}")
        return None


# Example 3: Using individual metric computation
def example_individual_metrics():
    """Example of computing individual metrics."""
    print("\n" + "="*60)
    print("Example 3: Individual Metric Computation")
    print("="*60)
    
    try:
        import torch
        from evaluate_4dgs import MetricsCompute, ImageLoader
        
        # Initialize
        metrics = MetricsCompute(device='cuda')
        loader = ImageLoader()
        
        # Load images
        render = loader.load_image('path/to/render.png')
        gt = loader.load_image('path/to/gt.png')
        mask = loader.load_mask('path/to/mask.png')
        
        # Compute individual metrics
        psnr = metrics.compute_psnr(render, gt)
        ssim = metrics.compute_ssim(render, gt)
        lpips_val = metrics.compute_lpips(render, gt)
        bg_psnr = metrics.compute_bg_psnr(render, gt, mask)
        clip_sim = metrics.compute_clip_similarity(render, gt)
        
        print(f"PSNR: {psnr:.6f}")
        print(f"SSIM: {ssim:.6f}")
        print(f"LPIPS: {lpips_val:.6f}")
        print(f"BG-PSNR: {bg_psnr:.6f}")
        print(f"CLIP Similarity: {clip_sim:.6f}")
        
    except Exception as e:
        print(f"Error in individual metrics example: {e}")


# Example 4: Batch evaluation of multiple experiments
def example_batch_evaluation():
    """Example of evaluating multiple experiments."""
    print("\n" + "="*60)
    print("Example 4: Batch Evaluation of Multiple Experiments")
    print("="*60)
    
    try:
        from evaluate_4dgs import Evaluator4DGS
        import json
        
        experiments = {
            'experiment_1': 'output/exp1/test',
            'experiment_2': 'output/exp2/test',
            'experiment_3': 'output/exp3/test',
        }
        
        results_summary = {}
        
        for exp_name, exp_path in experiments.items():
            print(f"\nEvaluating {exp_name}...")
            
            try:
                evaluator = Evaluator4DGS(
                    render_path=f'{exp_path}/renders',
                    gt_path=f'{exp_path}/gt',
                    mask_path=f'{exp_path}/masks'
                )
                
                results = evaluator.evaluate()
                results_summary[exp_name] = results
                evaluator.print_results(results)
                evaluator.save_results(results, exp_path)
                
            except Exception as e:
                print(f"Failed to evaluate {exp_name}: {e}")
        
        # Save summary
        summary_path = 'evaluation_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
        
    except Exception as e:
        print(f"Error in batch evaluation: {e}")


# Example 5: Custom metric computation pipeline
def example_custom_pipeline():
    """Example of creating a custom evaluation pipeline."""
    print("\n" + "="*60)
    print("Example 5: Custom Metric Computation Pipeline")
    print("="*60)
    
    print("""
from evaluate_4dgs import Evaluator4DGS, MetricsCompute, ImageLoader
import torch

# Custom pipeline: compute only SSIM and temporal consistency
class CustomEvaluator(Evaluator4DGS):
    def evaluate_fast(self):
        '''Fast evaluation: SSIM and temporal consistency only'''
        ssim_values = []
        temporal_ssim_values = []
        prev_render = None
        
        for idx, (render_path, gt_path) in enumerate(
            zip(self.render_images, self.gt_images)
        ):
            render = self.image_loader.load_image(render_path)
            gt = self.image_loader.load_image(gt_path)
            
            ssim = self.metrics.compute_ssim(render, gt)
            ssim_values.append(ssim)
            
            if prev_render is not None:
                t_ssim = self.metrics.compute_temporal_ssim(prev_render, render)
                temporal_ssim_values.append(t_ssim)
            
            prev_render = render
        
        return {
            'SSIM': sum(ssim_values) / len(ssim_values),
            'Temporal_SSIM': sum(temporal_ssim_values) / len(temporal_ssim_values)
        }

# Usage
evaluator = CustomEvaluator(
    render_path='output/exp/test/renders',
    gt_path='output/exp/test/gt'
)
results = evaluator.evaluate_fast()
    """)


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("Evaluation Pipeline for 4D Gaussian Splatting - Usage Examples")
    print("="*70 + "\n")
    
    example_cli()
    example_api()
    example_individual_metrics()
    example_batch_evaluation()
    example_custom_pipeline()
    
    print("\n" + "="*70)
    print("For detailed documentation, see EVALUATION_GUIDE.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
