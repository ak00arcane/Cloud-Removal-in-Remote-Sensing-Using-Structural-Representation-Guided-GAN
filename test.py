import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from config import Config
from models.generator import CloudRemovalGenerator
from utils.data_loader import create_dataloaders
from utils.metrics import calculate_all_metrics

from utils.data_loader import create_dataloaders

class Tester:
    def __init__(self, config, checkpoint_path='checkpoints/best_model.pth'):
        self.config = config
        self.device = config.DEVICE
        
        # Create model
        self.generator = CloudRemovalGenerator(config).to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Create dataloaders
        _, _, self.test_loader = create_dataloaders(config)
        
        # Create output directory
        self.output_dir = './test_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f'Tester initialized. Using device: {self.device}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f'Model loaded from: {checkpoint_path}')
        else:
            print(f'Warning: No checkpoint found at {checkpoint_path}')
    
    def denormalize(self, img):
        """Convert from [-1, 1] to [0, 1]"""
        return (img + 1) / 2
    
    def tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array for visualization"""
        img = tensor.detach().cpu().numpy()
        if len(img.shape) == 4:  # Batch
            img = img[0]
        if img.shape[0] in [1, 3]:  # (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        return np.clip(img, 0, 1)
    
    def save_comparison(self, cloudy, clean, pred, idx, metrics):
        """Save comparison visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Denormalize images
        cloudy = self.denormalize(cloudy)
        clean = self.denormalize(clean)
        pred = self.denormalize(pred)
        
        # Convert to numpy
        cloudy_np = self.tensor_to_numpy(cloudy)
        clean_np = self.tensor_to_numpy(clean)
        pred_np = self.tensor_to_numpy(pred)
        
        # Plot
        axes[0].imshow(cloudy_np)
        axes[0].set_title('Cloudy Input')
        axes[0].axis('off')
        
        axes[1].imshow(pred_np)
        axes[1].set_title(f'Prediction\nPSNR: {metrics["PSNR"]:.2f}, SSIM: {metrics["SSIM"]:.4f}')
        axes[1].axis('off')
        
        axes[2].imshow(clean_np)
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'result_{idx:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save individual images
        cv2.imwrite(
            os.path.join(self.output_dir, f'pred_{idx:04d}.png'),
            (pred_np * 255).astype(np.uint8)[..., ::-1]  # RGB to BGR
        )
    
    def test(self):
        """Run testing on test set"""
        self.generator.eval()
        
        all_metrics = {'PSNR': [], 'SSIM': [], 'CC': [], 'RMSE': []}
        
        print('Running inference on test set...')
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(self.test_loader, desc='Testing')):
                cloudy_img = batch['cloudy_img'].to(self.device)
                clean_img = batch['clean_img'].to(self.device)
                temporal_img = batch['temporal_img'].to(self.device)
                cloud_mask = batch['cloud_mask'].to(self.device)
                
                # Forward pass
                pred_img, _, _ = self.generator(cloudy_img, cloud_mask, temporal_img)
                
                # Calculate metrics
                metrics = calculate_all_metrics(pred_img, clean_img)
                
                for key in metrics:
                    all_metrics[key].append(metrics[key])
                
                # Save comparison for first 50 images
                if idx < 50:
                    self.save_comparison(cloudy_img, clean_img, pred_img, idx, metrics)
        
        # Calculate average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        std_metrics = {key: np.std(values) for key, values in all_metrics.items()}
        
        # Print results
        print('\n' + '='*50)
        print('Test Results:')
        print('='*50)
        for key in avg_metrics:
            print(f'{key:6s}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}')
        print('='*50)
        
        # Save results to file
        with open(os.path.join(self.output_dir, 'test_results.txt'), 'w') as f:
            f.write('Test Results:\n')
            f.write('='*50 + '\n')
            for key in avg_metrics:
                f.write(f'{key}: {avg_metrics[key]:.4f} ± {std_metrics[key]:.4f}\n')
        
        print(f'\nResults saved to: {self.output_dir}')
        
        return avg_metrics
    
    def infer_single_image(self, cloudy_path, temporal_path, mask_path=None, save_path='output.png'):
        """
        Perform inference on a single image
        
        Args:
            cloudy_path: Path to cloudy image
            temporal_path: Path to temporal reference image
            mask_path: Path to cloud mask (optional, will be auto-generated if None)
            save_path: Path to save output
        """
        self.generator.eval()
        
        # Load images
        cloudy_img = cv2.imread(cloudy_path)
        cloudy_img = cv2.cvtColor(cloudy_img, cv2.COLOR_BGR2RGB)
        
        temporal_img = cv2.imread(temporal_path)
        temporal_img = cv2.cvtColor(temporal_img, cv2.COLOR_BGR2RGB)
        
        # Resize
        h, w = cloudy_img.shape[:2]
        cloudy_img = cv2.resize(cloudy_img, (self.config.IMG_SIZE, self.config.IMG_SIZE))
        temporal_img = cv2.resize(temporal_img, (self.config.IMG_SIZE, self.config.IMG_SIZE))
        
        # Normalize
        cloudy_img = cloudy_img.astype(np.float32) / 255.0
        temporal_img = temporal_img.astype(np.float32) / 255.0
        
        # Load or generate mask
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.config.IMG_SIZE, self.config.IMG_SIZE))
            mask = mask.astype(np.float32) / 255.0
        else:
            # Auto-generate mask based on brightness
            gray = np.mean(cloudy_img, axis=-1)
            mask = (gray > 0.6).astype(np.float32)
        
        # Convert to tensors
        cloudy_tensor = torch.from_numpy(cloudy_img).permute(2, 0, 1).unsqueeze(0).float()
        temporal_tensor = torch.from_numpy(temporal_img).permute(2, 0, 1).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        
        # Normalize to [-1, 1]
        cloudy_tensor = cloudy_tensor * 2 - 1
        temporal_tensor = temporal_tensor * 2 - 1
        
        # Move to device
        cloudy_tensor = cloudy_tensor.to(self.device)
        temporal_tensor = temporal_tensor.to(self.device)
        mask_tensor = mask_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            pred_tensor, _, _ = self.generator(cloudy_tensor, mask_tensor, temporal_tensor)
        
        # Denormalize and convert to numpy
        pred_tensor = self.denormalize(pred_tensor)
        pred_img = self.tensor_to_numpy(pred_tensor)
        
        # Resize back to original size
        pred_img = cv2.resize(pred_img, (w, h))
        
        # Save result
        pred_img_bgr = (pred_img * 255).astype(np.uint8)
        pred_img_bgr = cv2.cvtColor(pred_img_bgr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, pred_img_bgr)
        
        print(f'Result saved to: {save_path}')
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        cloudy_display = cv2.resize(cloudy_img, (w, h))
        axes[0].imshow(cloudy_display)
        axes[0].set_title('Cloudy Input')
        axes[0].axis('off')
        
        axes[1].imshow(pred_img)
        axes[1].set_title('Cloud Removed')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_comparison.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        return pred_img

def main():
    config = Config()
    tester = Tester(config)
    
    # Run testing
    tester.test()

if __name__ == '__main__':
    main()
