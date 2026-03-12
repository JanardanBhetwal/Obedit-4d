# filepath: /media/JB/1.Pulchowk_Campus/7th Sem/8. Major Project/Major_Project/Instruct-4DGS/generate_test_images.py
import os
from pathlib import Path
from PIL import Image
import argparse

def generate_test_images(dataset, scene, prompt, output_dir):
    # Define paths
    test_dir = Path(output_dir) / "test"
    renders_dir = test_dir / "renders"
    gt_dir = test_dir / "gt"

    # Create directories if they don't exist
    renders_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Generate test images (dummy example, replace with your actual rendering logic)
    print(f"Generating test images for dataset: {dataset}, scene: {scene}, prompt: {prompt}")
    for i in range(5):  # Generate 5 test images as an example
        # Replace this with your actual rendering logic
        render_image = Image.new("RGB", (256, 256), (i * 50, i * 50, i * 50))  # Dummy render
        gt_image = Image.new("RGB", (256, 256), (255 - i * 50, 255 - i * 50, 255 - i * 50))  # Dummy ground truth

        # Save images
        render_image.save(renders_dir / f"render_{i}.png")
        gt_image.save(gt_dir / f"gt_{i}.png")

    print(f"Test images saved to {test_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test images for metrics evaluation")
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name")
    parser.add_argument("--scene", required=True, type=str, help="Scene name")
    parser.add_argument("--prompt", required=True, type=str, help="Editing prompt")
    parser.add_argument("--output_dir", required=True, type=str, help="Output directory for test images")
    args = parser.parse_args()

    generate_test_images(args.dataset, args.scene, args.prompt, args.output_dir)