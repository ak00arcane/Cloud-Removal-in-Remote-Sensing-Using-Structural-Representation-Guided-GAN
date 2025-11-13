import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
from pathlib import Path
import cv2
import random
from utils.cloud_synthesis import CloudMatting, compute_gradient_map, compute_structure_map
from utils.augmentation import get_training_augmentation, get_validation_augmentation

class SEN12MSCRDataset(Dataset):
    """
    Dataset for SEN12MS-CR cloud removal
    Handles the actual directory structure: data/asiaWest_n/ROIsXXXX/XXX/S2/X/*.tif
    """
    def __init__(self, root_dir, split='train', img_size=256, use_augmentation=True):
        """
        Args:
            root_dir: Root directory (e.g., './data')
            split: 'train', 'val', or 'test'
            img_size: Size to resize images to
            use_augmentation: Whether to use data augmentation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.use_augmentation = use_augmentation
        self.cloud_matting = CloudMatting()
        
        # Set up augmentations
        self.transform = get_training_augmentation(img_size) if split == 'train' and use_augmentation else \
                        get_validation_augmentation(img_size)
        
        # Find all S2 image directories
        self.image_pairs = self._load_image_pairs()
        print(f"Found {len(self.image_pairs)} image pairs for {split}")
        
    def _load_image_pairs(self):
        """Load all available S2 image pairs from the dataset"""
        image_pairs = []
        
        # Navigate through the directory structure
        # Pattern: data/asiaWest_n/ROIsXXXX/XXX/S2/X/*.tif
        
        for region_dir in self.root_dir.glob('*/'):  # asiaWest_n, etc.
            if not region_dir.is_dir():
                continue
            
            for rois_dir in region_dir.glob('ROIs*'):  # ROIs1868, ROIs1970, ROIs2017
                if not rois_dir.is_dir():
                    continue
                
                for scene_dir in rois_dir.glob('*'):  # 57, 83, 112, 127, etc.
                    if not scene_dir.is_dir():
                        continue
                    
                    # Look for S2 directory
                    s2_dir = scene_dir / 'S2'
                    if not s2_dir.exists():
                        continue
                    
                    # Get all subdirectories in S2 (0, 1, 2, 3, etc.)
                    s2_subdirs = sorted([d for d in s2_dir.glob('*') if d.is_dir()])
                    
                    if len(s2_subdirs) < 2:
                        continue  # Need at least 2 images for temporal pairs
                    
                    # For each subdirectory, get TIF files
                    for s2_subdir in s2_subdirs:
                        tif_files = sorted(list(s2_subdir.glob('*.tif')))
                        
                        if len(tif_files) > 0:
                            # Create pairs within same scene
                            for other_s2_subdir in s2_subdirs:
                                if other_s2_subdir != s2_subdir:
                                    other_tif_files = sorted(list(other_s2_subdir.glob('*.tif')))
                                    
                                    if len(other_tif_files) > 0:
                                        # Pair: (primary image dir, temporal reference dir)
                                        image_pairs.append({
                                            'primary_dir': s2_subdir,
                                            'primary_files': tif_files,
                                            'temporal_dir': other_s2_subdir,
                                            'temporal_files': other_tif_files,
                                            'scene_id': f"{rois_dir.name}_{scene_dir.name}"
                                        })
        
        # Split into train/val/test
        random.seed(42)
        random.shuffle(image_pairs)
        
        n_total = len(image_pairs)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        if self.split == 'train':
            return image_pairs[:n_train]
        elif self.split == 'val':
            return image_pairs[n_train:n_train+n_val]
        else:  # test
            return image_pairs[n_train+n_val:]
    
    def _load_tif_image(self, tif_path):
        """Load a .tif image using rasterio and handle Sentinel-2 bands"""
        try:
            with rasterio.open(tif_path) as src:
                # Read all bands
                img = src.read()  # Shape: (bands, height, width)
                
                # Sentinel-2 has 13 bands, we need RGB
                # Band 4 (Red), Band 3 (Green), Band 2 (Blue) - indices 3, 2, 1
                if img.shape[0] >= 4:
                    # Select RGB bands (B4, B3, B2)
                    rgb = np.stack([img[3], img[2], img[1]], axis=0)  # (3, H, W)
                elif img.shape[0] == 3:
                    rgb = img  # Already RGB
                else:
                    # Grayscale or single band - repeat to make RGB
                    rgb = np.stack([img[0], img[0], img[0]], axis=0)
                
                # Transpose to (H, W, C)
                rgb = np.transpose(rgb, (1, 2, 0))
                
                # Normalize to [0, 1]
                rgb = rgb.astype(np.float32)
                
                # Handle different value ranges
                if rgb.max() > 100:  # Likely 0-10000 or 0-255 range
                    if rgb.max() > 3000:
                        rgb = np.clip(rgb / 10000.0, 0, 1)  # Sentinel-2 typical range
                    else:
                        rgb = np.clip(rgb / 255.0, 0, 1)
                else:
                    rgb = np.clip(rgb / 100.0, 0, 1)
                
                return rgb
                
        except Exception as e:
            print(f"Error loading {tif_path}: {e}")
            # Return a blank image as fallback
            return np.zeros((256, 256, 3), dtype=np.float32)
    
    def _create_cloud_mask(self, img, threshold=0.7):
        """Create a cloud mask based on brightness"""
        # Convert to grayscale
        gray = np.mean(img, axis=-1)
        
        # Threshold to detect bright areas (potential clouds)
        mask = (gray > threshold).astype(np.float32)
        
        # Apply morphological operations to clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _augment(self, *images):
        """Apply random augmentation to images"""
        images = list(images)
        
        # Random horizontal flip (use copy to avoid negative strides)
        if random.random() > 0.5:
            images = [np.ascontiguousarray(np.fliplr(img)) for img in images]
        
        # Random vertical flip (use copy to avoid negative strides)
        if random.random() > 0.5:
            images = [np.ascontiguousarray(np.flipud(img)) for img in images]
        
        # Random rotation (90, 180, 270 degrees) (use copy to avoid negative strides)
        k = random.randint(0, 3)
        if k > 0:
            images = [np.ascontiguousarray(np.rot90(img, k)) for img in images]
        
        return images
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # Load primary image (will become clean ground truth)
        primary_file = random.choice(pair['primary_files'])
        clean_img = self._load_tif_image(primary_file)
        
        # Load temporal reference image
        temporal_file = random.choice(pair['temporal_files'])
        temporal_img = self._load_tif_image(temporal_file)
        
        # Resize images
        clean_img = cv2.resize(clean_img, (self.img_size, self.img_size))
        temporal_img = cv2.resize(temporal_img, (self.img_size, self.img_size))
        
        # Create cloud mask
        cloud_mask = self._create_cloud_mask(clean_img, threshold=0.7)
        
        # Synthesize cloudy image
        cloudy_img = clean_img.copy()
        
        # Add synthetic clouds
        if self.split == 'train' or random.random() > 0.3:
            # Create synthetic cloud pattern
            cloud_pattern = np.random.rand(self.img_size, self.img_size, 3) * 0.5 + 0.5
            alpha = cloud_mask[..., np.newaxis] * (0.5 + random.random() * 0.3)
            cloudy_img = alpha * cloud_pattern + (1 - alpha) * clean_img
            cloudy_img = np.clip(cloudy_img, 0, 1)
        
        # Compute gradient and structure maps
        gradient_map = compute_gradient_map(clean_img)
        structure_map = compute_structure_map(clean_img)
        
        # Apply augmentation
        if self.use_augmentation and self.split == 'train':
            cloudy_img, clean_img, temporal_img, cloud_mask, gradient_map, structure_map = \
                self._augment(cloudy_img, clean_img, temporal_img, cloud_mask, gradient_map, structure_map)
        
        # Convert to tensors
        cloudy_img = torch.from_numpy(cloudy_img).permute(2, 0, 1).float()
        clean_img = torch.from_numpy(clean_img).permute(2, 0, 1).float()
        temporal_img = torch.from_numpy(temporal_img).permute(2, 0, 1).float()
        cloud_mask = torch.from_numpy(cloud_mask).unsqueeze(0).float()
        gradient_map = torch.from_numpy(gradient_map).unsqueeze(0).float()
        structure_map = torch.from_numpy(structure_map).unsqueeze(0).float()
        
        # Normalize to [-1, 1] for GAN
        cloudy_img = cloudy_img * 2 - 1
        clean_img = clean_img * 2 - 1
        temporal_img = temporal_img * 2 - 1
        
        return {
            'cloudy_img': cloudy_img,
            'clean_img': clean_img,
            'temporal_img': temporal_img,
            'cloud_mask': cloud_mask,
            'gradient_map': gradient_map,
            'structure_map': structure_map,
            'is_real': torch.tensor(0.0)
        }

def create_dataloaders(config, num_workers=0):
    """Create train, validation, and test dataloaders"""
    
    # Check if CUDA is available for pin_memory
    import torch
    use_cuda = torch.cuda.is_available()
    
    # Use 0 workers on Windows to avoid multiprocessing issues
    if num_workers > 0:
        import platform
        if platform.system() == 'Windows':
            num_workers = 0
            print("Note: Using num_workers=0 on Windows to avoid multiprocessing issues")
    
    train_dataset = SEN12MSCRDataset(
        root_dir=config.DATASET_ROOT,
        split='train',
        img_size=config.IMG_SIZE,
        use_augmentation=config.USE_AUGMENTATION
    )
    
    val_dataset = SEN12MSCRDataset(
        root_dir=config.DATASET_ROOT,
        split='val',
        img_size=config.IMG_SIZE,
        use_augmentation=False
    )
    
    test_dataset = SEN12MSCRDataset(
        root_dir=config.DATASET_ROOT,
        split='test',
        img_size=config.IMG_SIZE,
        use_augmentation=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,  # Enable for GPU
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda
    )
    
    return train_loader, val_loader, test_loader