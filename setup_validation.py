"""
Setup and validation script for Phi-3 fine-tuning pipeline.

Checks system requirements, validates data, and provides quick setup.
"""

import sys
import subprocess
from pathlib import Path


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_python_version():
    """Check Python version."""
    print("\n1. Python Version")
    version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(f"   Current: Python {version}")

    if sys.version_info >= (3, 8):
        print("   ✓ Version OK (3.8+)")
        return True
    else:
        print("   ✗ Python 3.8+ required")
        return False


def check_gpu():
    """Check GPU availability."""
    print("\n2. GPU Configuration")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✓ GPU Available: {device_name}")
            print(f"   ✓ Memory: {device_memory:.2f} GB")

            if device_memory >= 16:
                print("   ✓ Memory OK (16GB+)")
                return True
            elif device_memory >= 8:
                print("   ⚠ Memory low (8GB) - may need optimization")
                return True
            else:
                print("   ✗ Memory too low (<8GB)")
                return False
        else:
            print("   ⚠ No GPU detected - training will be very slow")
            print("   ℹ CPU training possible but not recommended")
            return True

    except ImportError:
        print("   ✗ PyTorch not installed")
        return False


def check_libraries():
    """Check required libraries."""
    print("\n3. Required Libraries")

    libraries = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "datasets": "Datasets",
        "peft": "PEFT",
        "accelerate": "Accelerate",
    }

    all_installed = True

    for lib_name, display_name in libraries.items():
        try:
            __import__(lib_name)
            print(f"   ✓ {display_name}")
        except ImportError:
            print(f"   ✗ {display_name} - NOT INSTALLED")
            all_installed = False

    return all_installed


def check_data_files():
    """Check required data files."""
    print("\n4. Training Data")

    data_dir = Path("data")
    required_files = [
        "phi3_enhanced_training_data.jsonl",
        "phi3_training_data.csv",
    ]

    all_exist = True

    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / 1e6
            print(f"   ✓ {filename} ({size_mb:.1f} MB)")
        else:
            print(f"   ✗ {filename} - NOT FOUND")
            all_exist = False

    if not all_exist:
        print(
            "\n   ℹ Generate data files with: python phi3_financial_recommendation.py"
        )

    return all_exist


def check_scripts():
    """Check fine-tuning scripts."""
    print("\n5. Fine-tuning Scripts")

    scripts = [
        ("finetune_phi3_simple.py", "Simple fine-tuning"),
        ("finetune_phi3.py", "Advanced fine-tuning"),
        ("inference_phi3.py", "Inference and testing"),
    ]

    all_exist = True

    for script_name, description in scripts:
        if Path(script_name).exists():
            print(f"   ✓ {script_name}")
            print(f"      → {description}")
        else:
            print(f"   ✗ {script_name} - NOT FOUND")
            all_exist = False

    return all_exist


def validate_jsonl_format():
    """Validate JSONL format."""
    print("\n6. Training Data Format Validation")

    jsonl_path = Path("data/phi3_enhanced_training_data.jsonl")

    if not jsonl_path.exists():
        print(f"   ✗ File not found: {jsonl_path}")
        return False

    try:
        import json

        valid_count = 0
        invalid_count = 0

        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 10:  # Check first 10 lines
                    break
                if line.strip():
                    try:
                        example = json.loads(line)
                        if "messages" in example:
                            valid_count += 1
                        else:
                            invalid_count += 1
                    except json.JSONDecodeError:
                        invalid_count += 1

        if invalid_count == 0:
            print(f"   ✓ JSONL format valid")
            print(f"   ✓ Checked {valid_count} examples - all valid")
            return True
        else:
            print(f"   ✗ Invalid examples found: {invalid_count}")
            return False

    except Exception as e:
        print(f"   ✗ Error validating: {e}")
        return False


def get_system_info():
    """Get system information."""
    print("\n7. System Information")

    try:
        import platform

        print(f"   OS: {platform.system()} {platform.release()}")
        print(f"   Processor: {platform.processor()}")
    except:
        pass


def print_summary(checks: dict):
    """Print summary of checks."""
    print_header("VALIDATION SUMMARY")

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {check_name}")

    print(f"\n  Result: {passed}/{total} checks passed")

    if passed == total:
        print("\n  ✓ All checks passed! Ready to start fine-tuning.\n")
        return True
    else:
        print(
            f"\n  ✗ {total - passed} check(s) failed. Please fix before proceeding.\n"
        )
        return False


def print_next_steps():
    """Print next steps."""
    print_header("NEXT STEPS")

    print(
        """
Quick Start:
  1. python finetune_phi3_simple.py
  
  This will:
    • Load 5,000 training examples
    • Apply 8-bit quantization
    • Configure LoRA fine-tuning
    • Train for 3 epochs
    • Save model to phi3_finetuned_simple/

Estimated Time:
  • RTX 4090: ~30-45 minutes
  • RTX 3090: ~60-90 minutes
  • CPU: 4-8 hours (not recommended)

After Fine-tuning:
  1. Test inference: python inference_phi3.py
  2. Review FINETUNING_GUIDE.md for advanced options
  3. Deploy to production

For Help:
  • Read FINETUNING_GUIDE.md for detailed instructions
  • Check training logs in ./logs/
  • Use tensorboard to monitor training:
    tensorboard --logdir ./logs
"""
    )


def install_missing_packages():
    """Offer to install missing packages."""
    try:
        print_header("INSTALLING MISSING PACKAGES")

        packages = [
            "torch",
            "transformers",
            "datasets",
            "peft",
            "accelerate",
            "bitsandbytes",
        ]

        print(f"\nInstalling: {', '.join(packages)}")
        print("This may take several minutes...")

        for package in packages:
            try:
                __import__(package)
            except ImportError:
                print(f"\nInstalling {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        print("\n✓ All packages installed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error during installation: {e}")
        return False


def main():
    """Main validation function."""

    print_header("PHI-3 FINE-TUNING - SYSTEM VALIDATION")

    checks = {}

    # Run checks
    checks["Python Version"] = check_python_version()
    checks["GPU Configuration"] = check_gpu()
    checks["Required Libraries"] = check_libraries()
    checks["Training Data Files"] = check_data_files()
    checks["Fine-tuning Scripts"] = check_scripts()

    if checks["Training Data Files"]:
        checks["JSONL Format"] = validate_jsonl_format()

    get_system_info()

    # Print summary
    all_passed = print_summary(checks)

    # Handle missing libraries
    if not checks["Required Libraries"]:
        print("\nWould you like to install missing packages? (y/n): ", end="")
        response = input().strip().lower()
        if response in ("y", "yes"):
            if install_missing_packages():
                print("\n✓ Please run this script again to verify installation.")
                return

    # Print next steps if all checks passed
    if all_passed:
        print_next_steps()
    else:
        print(
            """
Please fix the failed checks before proceeding.

Common fixes:
  1. Install missing packages:
     pip install torch transformers datasets peft accelerate bitsandbytes
     
  2. Generate training data:
     python phi3_financial_recommendation.py
     
  3. Update GPU drivers for CUDA support
"""
        )


if __name__ == "__main__":
    main()
