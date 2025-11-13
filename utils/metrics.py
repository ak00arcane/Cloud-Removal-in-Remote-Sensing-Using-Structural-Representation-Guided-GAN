import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(img1.shape) == 4:
        psnr_values = []
        for i in range(img1.shape[0]):
            # Convert from (C, H, W) to (H, W, C)
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            psnr_values.append(psnr(im2, im1, data_range=data_range))
        return np.mean(psnr_values)
    else:
        if len(img1.shape) == 3 and img1.shape[0] in [1, 3]:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        return psnr(img2, img1, data_range=data_range)

def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculate Structural Similarity Index
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(img1.shape) == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            # Convert from (C, H, W) to (H, W, C)
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            ssim_values.append(ssim(im2, im1, data_range=data_range, channel_axis=2))
        return np.mean(ssim_values)
    else:
        if len(img1.shape) == 3 and img1.shape[0] in [1, 3]:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        return ssim(img2, img1, data_range=data_range, channel_axis=2)

def calculate_cc(img1, img2):
    """
    Calculate Correlation Coefficient
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Flatten images
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    # Handle degenerate cases where standard deviation is zero
    std1 = img1_flat.std()
    std2 = img2_flat.std()
    eps = 1e-8
    if std1 < eps or std2 < eps:
        # Correlation undefined if one image is constant; return 0.0 as a neutral value
        return 0.0

    # Calculate correlation coefficient safely
    cc = np.corrcoef(img1_flat, img2_flat)[0, 1]
    if not np.isfinite(cc):
        return 0.0

    return float(cc)

def calculate_rmse(img1, img2):
    """
    Calculate Root Mean Square Error
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    mse = np.mean((img1 - img2) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse

def calculate_all_metrics(pred, target, data_range=2.0):
    """
    Calculate all metrics: PSNR, SSIM, CC, RMSE
    
    Args:
        pred: Predicted image (range: [-1, 1])
        target: Target image (range: [-1, 1])
        data_range: Range of pixel values
    
    Returns:
        dict: Dictionary containing all metrics
    """
    # Convert from [-1, 1] to [0, 1]
    pred_norm = (pred + 1) / 2
    target_norm = (target + 1) / 2
    
    metrics = {
        'PSNR': calculate_psnr(pred_norm, target_norm, data_range=1.0),
        'SSIM': calculate_ssim(pred_norm, target_norm, data_range=1.0),
        'CC': calculate_cc(pred_norm, target_norm),
        'RMSE': calculate_rmse(pred_norm, target_norm)
    }
    
    return metrics