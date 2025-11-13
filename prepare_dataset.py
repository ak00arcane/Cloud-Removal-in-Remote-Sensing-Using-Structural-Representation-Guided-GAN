import os
import tarfile
from pathlib import Path
from tqdm import tqdm

def extract_tar_files(source_dir, target_dir):
    """
    Extract all .tar.gz files from source_dir to target_dir
    
    Args:
        source_dir: Directory containing .tar.gz files
        target_dir: Directory where files will be extracted
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .tar.gz files
    tar_files = list(source_path.glob('*.tar.gz'))
    
    if not tar_files:
        print(f"No .tar.gz files found in {source_dir}")
        return
    
    print(f"Found {len(tar_files)} tar.gz files to extract")
    
    # Extract each file
    for tar_file in tqdm(tar_files, desc="Extracting files"):
        # Only extract s2 (Sentinel-2 optical) files for this implementation
        if '_s2.tar.gz' not in str(tar_file):
            print(f"Skipping {tar_file.name} (not Sentinel-2 data)")
            continue
        
        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                # Extract to target directory
                tar.extractall(path=target_path)
                print(f"Extracted: {tar_file.name}")
        except Exception as e:
            print(f"Error extracting {tar_file.name}: {e}")
    
    print(f"\nExtraction complete! Files extracted to: {target_path}")
    
    # Show directory structure
    show_structure(target_path)

def show_structure(directory, max_depth=3, current_depth=0):
    """Display directory structure"""
    if current_depth >= max_depth:
        return
    
    path = Path(directory)
    indent = "  " * current_depth
    
    if current_depth == 0:
        print(f"\nDirectory structure:")
        print(f"{path}/")
    
    try:
        items = sorted(path.iterdir())
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # Show directories
        for d in dirs[:5]:  # Limit to first 5
            print(f"{indent}├── {d.name}/")
            if current_depth < max_depth - 1:
                show_structure(d, max_depth, current_depth + 1)
        
        if len(dirs) > 5:
            print(f"{indent}└── ... ({len(dirs) - 5} more directories)")
        
        # Show file count
        if files:
            print(f"{indent}└── {len(files)} files")
            
    except PermissionError:
        print(f"{indent}[Permission Denied]")

def verify_dataset(data_dir):
    """
    Verify the extracted dataset structure
    """
    data_path = Path(data_dir)
    
    print("\n" + "="*60)
    print("Dataset Verification")
    print("="*60)
    
    # Find all ROI directories
    roi_dirs = list(data_path.glob('ROIs*'))
    
    if not roi_dirs:
        print("❌ No ROI directories found!")
        print(f"   Expected directories like: ROIs1868_summer, ROIs1970_fall, etc.")
        return False
    
    print(f"✓ Found {len(roi_dirs)} ROI directories:")
    for roi_dir in roi_dirs:
        print(f"  - {roi_dir.name}")
    
    # Check for s2 subdirectories
    s2_dirs = list(data_path.glob('**/s2_*'))
    
    if not s2_dirs:
        print("\n❌ No Sentinel-2 (s2) directories found!")
        return False
    
    print(f"\n✓ Found {len(s2_dirs)} Sentinel-2 scene directories")
    
    # Check for .tif files
    tif_files = list(data_path.glob('**/*.tif'))
    
    if not tif_files:
        print("\n❌ No .tif image files found!")
        return False
    
    print(f"✓ Found {len(tif_files)} .tif image files")
    
    # Sample file info
    if tif_files:
        sample_file = tif_files[0]
        print(f"\nSample file: {sample_file.name}")
        print(f"File size: {sample_file.stat().st_size / (1024*1024):.2f} MB")
    
    print("\n" + "="*60)
    print("Dataset verification complete! ✓")
    print("="*60)
    
    return True

def main():
    """Main function to prepare the dataset"""
    
    print("="*60)
    print("SEN12MS-CR Dataset Preparation")
    print("="*60)
    
    # Paths - MODIFY THESE ACCORDING TO YOUR SETUP
    DOWNLOAD_DIR = "./data/SEN12MS-CR-raw"  # Where you downloaded the .tar.gz files
    EXTRACT_DIR = "./data/SEN12MS-CR"        # Where files will be extracted
    
    print(f"\nSource directory: {DOWNLOAD_DIR}")
    print(f"Target directory: {EXTRACT_DIR}")
    
    # Check if source directory exists
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"\n❌ Error: Source directory not found: {DOWNLOAD_DIR}")
        print("\nPlease:")
        print("1. Create the directory: mkdir -p ./data/SEN12MS-CR-raw")
        print("2. Move your downloaded .tar.gz files to this directory")
        print("3. Run this script again")
        return
    
    # Ask for confirmation
    response = input("\nProceed with extraction? (y/n): ")
    if response.lower() != 'y':
        print("Extraction cancelled.")
        return
    
    # Extract files
    extract_tar_files(DOWNLOAD_DIR, EXTRACT_DIR)
    
    # Verify dataset
    verify_dataset(EXTRACT_DIR)
    
    print("\n✓ Dataset preparation complete!")
    print("\nNext steps:")
    print("1. Review the extracted data structure above")
    print("2. Update config.py with the correct DATASET_ROOT path")
    print("3. Run: python train.py")

if __name__ == '__main__':
    main()