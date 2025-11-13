#!/usr/bin/env python
"""
Verification script to check if everything is set up correctly
Run this before training: python verify_setup.py
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("="*60)
    print("1. Checking Dependencies...")
    print("="*60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'skimage': 'scikit-image',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'tensorboard': 'TensorBoard',
        'rasterio': 'Rasterio',
        'scipy': 'SciPy'
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name} installed")
        except ImportError:
            print(f"  ✗ {name} NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True

def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("2. Checking CUDA/GPU...")
    print("="*60)
    
    try:
        import torch
        
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available!")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("  ⚠ CUDA not available - will use CPU (very slow)")
            print("  Consider installing PyTorch with CUDA support")
            return True  # Not critical, can train on CPU
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

def check_data_structure():
    """Check if data is properly structured"""
    print("\n" + "="*60)
    print("3. Checking Data Structure...")
    print("="*60)
    
    data_root = Path('./data')
    
    if not data_root.exists():
        print(f"  ✗ Data directory not found: {data_root.absolute()}")
        return False
    
    print(f"  ✓ Data directory exists: {data_root.absolute()}")
    
    # Look for S2 directories
    s2_dirs = list(data_root.glob('**/S2'))
    
    if not s2_dirs:
        print("  ✗ No S2 (Sentinel-2) directories found!")
        print("  Expected structure: data/*/ROIs*/*/S2/")
        return False
    
    print(f"  ✓ Found {len(s2_dirs)} S2 directories")
    
    # Check TIF files
    total_tifs = 0
    total_subdirs = 0
    
    for s2_dir in s2_dirs[:3]:  # Show first 3
        print(f"\n  Checking: {s2_dir.relative_to(data_root)}")
        subdirs = [d for d in s2_dir.glob('*') if d.is_dir()]
        total_subdirs += len(subdirs)
        print(f"    Subdirectories: {len(subdirs)}")
        
        tif_files = list(s2_dir.glob('*/*.tif'))
        total_tifs += len(tif_files)
        print(f"    TIF files: {len(tif_files)}")
        
        if tif_files:
            sample_tif = tif_files[0]
            print(f"    Sample: {sample_tif.name}")
            
            # Try to load it
            try:
                import rasterio
                with rasterio.open(sample_tif) as src:
                    print(f"    Bands: {src.count}, Size: {src.width}x{src.height}")
            except Exception as e:
                print(f"    ⚠ Could not read TIF: {e}")
    
    if len(s2_dirs) > 3:
        print(f"\n  ... and {len(s2_dirs) - 3} more S2 directories")
    
    print(f"\n  Total TIF files found: {total_tifs}")
    
    if total_tifs < 10:
        print("  ⚠ Very few TIF files found. Is the dataset complete?")
    else:
        print("  ✓ Dataset looks good!")
    
    return True

def check_project_structure():
    """Check if all required files exist"""
    print("\n" + "="*60)
    print("4. Checking Project Structure...")
    print("="*60)
    
    required_files = [
        'config.py',
        'train.py',
        'test.py',
        'models/__init__.py',
        'models/generator.py',
        'models/discriminator.py',
        'models/losses.py',
        'utils/__init__.py',
        'utils/data_loader.py',
        'utils/cloud_synthesis.py',
        'utils/metrics.py'
    ]
    
    required_dirs = [
        'checkpoints',
        'logs',
        'test_results'
    ]
    
    all_good = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING!")
            all_good = False
    
    print()
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ⚠ {dir_path}/ - creating...")
            path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {dir_path}/ - created")
    
    if not all_good:
        print("\n❌ Some required files are missing!")
        return False
    
    print("\n✓ All required files present!")
    return True

def test_imports():
    """Test if model imports work"""
    print("\n" + "="*60)
    print("5. Testing Model Imports...")
    print("="*60)
    
    try:
        print("  Testing config...")
        from config import Config
        config = Config()
        print(f"  ✓ Config loaded (batch_size={config.BATCH_SIZE}, img_size={config.IMG_SIZE})")
        
        print("  Testing models...")
        from models.generator import CloudRemovalGenerator
        from models.discriminator import Discriminator
        from models.losses import CloudRemovalLoss, LSGANLoss
        print("  ✓ Models imported")
        
        print("  Testing utils...")
        from utils.data_loader import SEN12MSCRDataset, create_dataloaders
        from utils.metrics import calculate_psnr, calculate_ssim
        print("  ✓ Utils imported")
        
        print("\n✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def estimate_dataset_size():
    """Estimate how many samples we have"""
    print("\n" + "="*60)
    print("6. Estimating Dataset Size...")
    print("="*60)
    
    try:
        from config import Config
        from utils.data_loader import SEN12MSCRDataset
        
        config = Config()
        
        print("  Loading train dataset...")
        train_dataset = SEN12MSCRDataset(config.DATASET_ROOT, split='train', img_size=128)
        print(f"  ✓ Train: {len(train_dataset)} samples")
        
        print("  Loading val dataset...")
        val_dataset = SEN12MSCRDataset(config.DATASET_ROOT, split='val', img_size=128)
        print(f"  ✓ Val: {len(val_dataset)} samples")
        
        print("  Loading test dataset...")
        test_dataset = SEN12MSCRDataset(config.DATASET_ROOT, split='test', img_size=128)
        print(f"  ✓ Test: {len(test_dataset)} samples")
        
        total = len(train_dataset) + len(val_dataset) + len(test_dataset)
        print(f"\n  Total: {total} samples")
        
        if total < 50:
            print("  ⚠ Dataset seems small. Consider adding more data.")
        else:
            print("  ✓ Dataset size looks good!")
        
        # Test loading one sample
        print("\n  Testing sample loading...")
        sample = train_dataset[0]
        print(f"  ✓ Sample loaded successfully")
        print(f"    Cloudy image shape: {sample['cloudy_img'].shape}")
        print(f"    Clean image shape: {sample['clean_img'].shape}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("CLOUD REMOVAL SETUP VERIFICATION")
    print("="*60 + "\n")
    
    checks = [
        ("Dependencies", check_dependencies),
        ("CUDA/GPU", check_cuda),
        ("Data Structure", check_data_structure),
        ("Project Structure", check_project_structure),
        ("Imports", test_imports),
        ("Dataset", estimate_dataset_size)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED! Ready to train!")
        print("="*60)
        print("\nNext step:")
        print("  python train.py")
    else:
        print("✗ SOME CHECKS FAILED!")
        print("="*60)
        print("\nPlease fix the issues above before training.")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)