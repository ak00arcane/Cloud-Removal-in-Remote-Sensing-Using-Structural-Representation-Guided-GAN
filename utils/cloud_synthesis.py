import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class CloudMatting:
    """
    Cloud Matting algorithm for synthesizing realistic cloud images
    Based on equation 10: I(x) = R_c(x) + [1 - α(x)]R_b(x)
    """
    def __init__(self):
        pass
    
    def extract_cloud_pattern(self, cloud_image, cloud_mask):
        """
        Extract cloud foreground from real cloud image
        
        Args:
            cloud_image: Real cloud image (H, W, C)
            cloud_mask: Binary cloud mask (H, W)
        
        Returns:
            cloud_foreground: Extracted cloud pattern
            alpha: Attenuation factor
        """
        # Normalize inputs
        cloud_mask = cloud_mask.astype(np.float32)
        
        # Estimate alpha (attenuation factor) using cloud density
        # Alpha represents how much the ground is attenuated by clouds
        alpha = gaussian_filter(cloud_mask, sigma=5)
        alpha = np.clip(alpha, 0, 1)
        
        # Extract cloud foreground
        # For simplicity, we estimate it as the difference in bright regions
        cloud_brightness = np.mean(cloud_image, axis=-1)
        cloud_brightness_norm = (cloud_brightness - cloud_brightness.min()) / (cloud_brightness.max() - cloud_brightness.min() + 1e-8)
        
        # Cloud foreground is the bright contribution
        cloud_foreground = cloud_image * cloud_brightness_norm[..., np.newaxis]
        
        return cloud_foreground, alpha
    
    def synthesize_cloud_image(self, clean_image, cloud_pattern, alpha, cloud_mask):
        """
        Synthesize cloud image using Cloud Matting formula
        I(x) = R_c(x) + [1 - α(x)]R_b(x)
        
        Args:
            clean_image: Cloud-free ground image (H, W, C)
            cloud_pattern: Cloud foreground R_c (H, W, C)
            alpha: Attenuation factor (H, W)
            cloud_mask: Binary mask indicating cloud regions (H, W)
        
        Returns:
            synthetic_cloud_image: Synthesized cloud image
        """
        # Ensure alpha has correct shape
        if len(alpha.shape) == 2:
            alpha = alpha[..., np.newaxis]
        
        # Apply cloud matting formula
        # I(x) = R_c(x) + [1 - α(x)] * R_b(x)
        synthetic_image = cloud_pattern * cloud_mask[..., np.newaxis] + (1 - alpha) * clean_image
        
        # Clip values to valid range
        synthetic_image = np.clip(synthetic_image, 0, 255).astype(np.uint8)
        
        return synthetic_image
    
    def generate_synthetic_pair(self, clean_image, reference_cloud_image, reference_cloud_mask):
        """
        Generate synthetic cloud image from clean image and reference cloud
        
        Args:
            clean_image: Clean ground truth image (H, W, C)
            reference_cloud_image: Real cloud image for pattern extraction (H, W, C)
            reference_cloud_mask: Cloud mask for reference image (H, W)
        
        Returns:
            synthetic_cloud_image: Synthesized cloud image
            cloud_mask: Cloud mask
        """
        # Resize reference if needed
        if reference_cloud_image.shape[:2] != clean_image.shape[:2]:
            reference_cloud_image = cv2.resize(reference_cloud_image, 
                                              (clean_image.shape[1], clean_image.shape[0]))
            reference_cloud_mask = cv2.resize(reference_cloud_mask, 
                                             (clean_image.shape[1], clean_image.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)
        
        # Extract cloud pattern from reference
        cloud_pattern, alpha = self.extract_cloud_pattern(reference_cloud_image, reference_cloud_mask)
        
        # Synthesize cloud image
        synthetic_image = self.synthesize_cloud_image(clean_image, cloud_pattern, alpha, reference_cloud_mask)
        
        return synthetic_image, reference_cloud_mask

def compute_gradient_map(image):
    """
    Compute gradient magnitude map for an image
    
    Args:
        image: Input image (H, W, C) or (H, W)
    
    Returns:
        gradient_map: Gradient magnitude (H, W)
    """
    if len(image.shape) == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute gradients using Sobel operator
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to [0, 1]
    gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min() + 1e-8)
    
    return gradient_magnitude

def compute_structure_map(image, method='canny'):
    """
    Compute structure map using edge detection
    
    Args:
        image: Input image (H, W, C) or (H, W)
        method: 'canny' or 'laplacian'
    
    Returns:
        structure_map: Structure/edge map (H, W)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    
    if method == 'canny':
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        structure_map = edges.astype(np.float32) / 255.0
    else:
        # Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        structure_map = np.abs(laplacian)
        structure_map = (structure_map - structure_map.min()) / (structure_map.max() - structure_map.min() + 1e-8)
    
    return structure_map