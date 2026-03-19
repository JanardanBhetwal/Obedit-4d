#!/usr/bin/env python3
"""
Setup and validation script for the evaluation pipeline.
Checks dependencies and validates the evaluation pipeline is ready to use.
"""

import sys
import subprocess
import importlib.util
from pathlib import Path


class EnvironmentValidator:
    """Validates the evaluation environment."""
    
    REQUIRED_PACKAGES = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'torchmetrics': 'torchmetrics',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'transformers': 'transformers',
        'lpips': 'lpips',
        'scipy': 'scipy',
    }
    
    OPTIONAL_PACKAGES = {
        'cv2': 'opencv-python',
        'skimage': 'scikit-image',
    }
    
    def __init__(self):
        """Initialize validator."""
        self.missing_required = []
        self.missing_optional = []
        self.gpu_available = False
        
    def check_package(self, import_name: str, pip_name: str, optional: bool = False) -> bool:
        """
        Check if a package is installed.
        
        Args:
            import_name: Name to use in import statement
            pip_name: Name used in pip install command
            optional: Whether this is an optional dependency
            
        Returns:
            True if installed, False otherwise
        """
        spec = importlib.util.find_spec(import_name)
        is_available = spec is not None
        
        if not is_available:
            if optional:
                self.missing_optional.append((import_name, pip_name))
            else:
                self.missing_required.append((import_name, pip_name))
        
        return is_available
    
    def check_gpu(self) -> bool:
        """Check if CUDA/GPU is available."""
        try:
            import torch
            available = torch.cuda.is_available()
            if available:
                self.gpu_available = True
                print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                print("⚠ CUDA not available. Will use CPU for evaluation.")
            return available
        except Exception as e:
            print(f"⚠ Could not check CUDA: {e}")
            return False
    
    def validate(self) -> bool:
        """
        Validate the environment.
        
        Returns:
            True if all required packages are available, False otherwise
        """
        print("="*70)
        print("Validation: 4D Gaussian Splatting Evaluation Pipeline")
        print("="*70 + "\n")
        
        print("Checking required packages...")
        for import_name, pip_name in self.REQUIRED_PACKAGES.items():
            if self.check_package(import_name, pip_name):
                print(f"✓ {pip_name}")
            else:
                print(f"✗ {pip_name} (missing)")
        
        print("\nChecking optional packages...")
        for import_name, pip_name in self.OPTIONAL_PACKAGES.items():
            if self.check_package(import_name, pip_name, optional=True):
                print(f"✓ {pip_name}")
            else:
                print(f"○ {pip_name} (optional, not installed)")
        
        print("\nChecking GPU/CUDA...")
        self.check_gpu()
        
        print("\n" + "="*70)
        
        if self.missing_required:
            print("Missing Required Packages:")
            print("-"*70)
            for import_name, pip_name in self.missing_required:
                print(f"  • {pip_name}")
            
            print("\nInstall with:")
            install_cmd = " ".join([pip_name for _, pip_name in self.missing_required])
            print(f"  pip install {install_cmd}")
            
            print("\nOr install all evaluation dependencies with:")
            print("  pip install -r requirements_evaluation.txt")
            
            print("="*70 + "\n")
            return False
        else:
            print("✓ All required packages are installed!")
            if self.missing_optional:
                print(f"⚠ {len(self.missing_optional)} optional package(s) not installed.")
                print("  (Evaluation will still work, but some features may be unavailable)")
            print("="*70 + "\n")
            return True
    
    def recommend_gpu(self) -> str:
        """Get GPU recommendation based on availability."""
        if self.gpu_available:
            return "cuda"
        else:
            return "cpu"


def check_evaluate_4dgs_script() -> bool:
    """Check if evaluate_4dgs.py exists and is valid."""
    script_path = Path(__file__).parent / 'evaluate_4dgs.py'
    
    if not script_path.exists():
        print(f"✗ evaluate_4dgs.py not found at {script_path}")
        return False
    
    try:
        with open(script_path) as f:
            content = f.read()
            required_classes = ['Evaluator4DGS', 'MetricsCompute', 'ImageLoader']
            for cls in required_classes:
                if f'class {cls}' not in content:
                    print(f"✗ Required class {cls} not found in evaluate_4dgs.py")
                    return False
        
        print(f"✓ evaluate_4dgs.py exists and contains all required components")
        return True
    
    except Exception as e:
        print(f"✗ Error validating evaluate_4dgs.py: {e}")
        return False


def print_quick_start():
    """Print quick start guide."""
    print("\n" + "="*70)
    print("Quick Start Guide")
    print("="*70 + "\n")
    
    print("1. Prepare your data:")
    print("""
   output/
   └── experiment_name/
       └── test/
           ├── renders/    # Edited 4D Gaussian renders
           ├── gt/         # Ground truth frames
           └── masks/      # (Optional) Binary masks for BG-PSNR
    """)
    
    print("2. Run evaluation:")
    print("""
   python evaluate_4dgs.py \\
     --render_path output/experiment_name/test/renders \\
     --gt_path output/experiment_name/test/gt \\
     --mask_path output/experiment_name/test/masks
    """)
    
    print("3. Check results:")
    print("""
   • Terminal: Formatted metrics table printed to console
   • File: metrics.json saved in test/ directory
    """)
    
    print("For detailed documentation, see EVALUATION_GUIDE.md")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    validator = EnvironmentValidator()
    
    # Validate environment
    environment_ok = validator.validate()
    
    # Check evaluate_4dgs.py
    script_ok = check_evaluate_4dgs_script()
    
    print()
    
    if environment_ok and script_ok:
        print("✓ Environment is ready for evaluation!")
        print_quick_start()
        return 0
    else:
        print("✗ Environment setup incomplete. Please install missing dependencies.")
        if not environment_ok:
            print("  Run: pip install -r requirements_evaluation.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
