import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from models.generator import CloudRemovalGenerator

class CloudRemovalInference:
    def __init__(self, checkpoint_path='checkpoints/best_model.pth', device=None):
        self.config = Config()
        self.device = device if device else self.config.DEVICE
        
        # Create and load model
        self.generator = CloudRemovalGenerator(self.config).to(self.device)
        self.load_checkpoint(checkpoint_path)
        
        print(f'Model loaded from: {checkpoint_path}')
        print(f'Using device: {self.device}')
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f'Successfully loaded checkpoint from: {checkpoint_path}')
        else:
            raise FileNotFoundError(f'No checkpoint found at {checkpoint_path}')
    
    def preprocess_image(self, img_path, size=None):
        """Load and preprocess image"""
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f'Could not read image: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if size:
            img = cv2.resize(img, (size, size))
        
        # Convert to float and normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = img * 2 - 1
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img
    
    def postprocess_image(self, tensor):
        """Convert tensor to numpy image"""
        # Move to CPU and convert to numpy
        img = tensor.detach().cpu().numpy()
        
        # Take first image if batch
        if len(img.shape) == 4:
            img = img[0]
        
        # Convert from CHW to HWC
        img = np.transpose(img, (1, 2, 0))
        
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        
        # Clip values
        img = np.clip(img, 0, 1)
        
        # Convert to uint8
        img = (img * 255).astype(np.uint8)
        
        return img
    
    def generate_cloud_mask(self, img):
        """Simple cloud mask generation based on brightness"""
        # Convert to grayscale and normalize
        gray = np.mean(img.cpu().numpy()[0], axis=0)
        
        # Threshold for cloud detection (adjust if needed)
        mask = (gray > 0.6).astype(np.float32)
        
        # Convert to tensor
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
        return mask
    
    def run_inference(self, cloudy_path, temporal_path=None, output_dir='inference_results', 
                     save_comparison=True, size=None):
        """
        Run inference on a single image pair
        
        Args:
            cloudy_path: Path to cloudy image
            temporal_path: Path to temporal reference image (optional)
            output_dir: Directory to save results
            save_comparison: If True, save side-by-side comparison
            size: Size to resize images to (optional)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Load cloudy image
        cloudy_img = self.preprocess_image(cloudy_path, size).to(self.device)
        
        # Load or create temporal image
        if temporal_path:
            temporal_img = self.preprocess_image(temporal_path, size).to(self.device)
        else:
            # If no temporal image, use cloudy image (less optimal but works)
            temporal_img = cloudy_img.clone()
        
        # Generate cloud mask
        cloud_mask = self.generate_cloud_mask(cloudy_img)
        
        # Run inference
        self.generator.eval()
        with torch.no_grad():
            pred_img, _, _ = self.generator(cloudy_img, cloud_mask, temporal_img)
        
        # Convert predictions to numpy
        pred_img_np = self.postprocess_image(pred_img)
        cloudy_img_np = self.postprocess_image(cloudy_img)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(cloudy_path))[0]
        pred_path = os.path.join(output_dir, f'{base_name}_cloudless.png')
        
        # Save prediction
        cv2.imwrite(pred_path, cv2.cvtColor(pred_img_np, cv2.COLOR_RGB2BGR))
        
        # Save comparison if requested
        if save_comparison:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(cloudy_img_np)
            axes[0].set_title('Input (Cloudy)')
            axes[0].axis('off')
            
            axes[1].imshow(pred_img_np)
            axes[1].set_title('Prediction (Cloud-free)')
            axes[1].axis('off')
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, f'{base_name}_comparison.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f'Results saved to: {output_dir}')
        return pred_img_np
    
    def batch_inference(self, input_dir, temporal_dir=None, output_dir='inference_results'):
        """
        Run inference on all images in a directory
        
        Args:
            input_dir: Directory containing cloudy images
            temporal_dir: Directory containing temporal images (optional)
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all images
        image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc='Processing images'):
            cloudy_path = os.path.join(input_dir, img_file)
            
            # Find matching temporal image if directory provided
            temporal_path = None
            if temporal_dir:
                temporal_path = os.path.join(temporal_dir, img_file)
                if not os.path.exists(temporal_path):
                    print(f'Warning: No matching temporal image found for {img_file}')
                    temporal_path = None
            
            try:
                self.run_inference(cloudy_path, temporal_path, output_dir)
            except Exception as e:
                print(f'Error processing {img_file}: {str(e)}')

def main():
    parser = argparse.ArgumentParser(description='Cloud Removal Inference')
    parser.add_argument('--input', required=True, help='Path to input image or directory')
    parser.add_argument('--temporal', help='Path to temporal reference image or directory')
    parser.add_argument('--output', default='inference_results', help='Output directory')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--size', type=int, help='Size to resize images to')
    parser.add_argument('--no-comparison', action='store_true', help='Don\'t save comparison visualizations')
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = CloudRemovalInference(checkpoint_path=args.checkpoint)
    
    # Check if input is directory or single image
    if os.path.isdir(args.input):
        print(f'Processing directory: {args.input}')
        inferencer.batch_inference(args.input, args.temporal, args.output)
    else:
        print(f'Processing single image: {args.input}')
        inferencer.run_inference(
            args.input, 
            args.temporal,
            args.output,
            save_comparison=not args.no_comparison,
            size=args.size
        )

if __name__ == '__main__':
    main()