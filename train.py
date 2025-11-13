import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from config import Config
from models.generator import CloudRemovalGenerator
from models.discriminator import Discriminator
from models.losses import CloudRemovalLoss, LSGANLoss
from utils.data_loader import create_dataloaders
from utils.metrics import calculate_all_metrics

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Create models
        self.generator = CloudRemovalGenerator(config).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # Create loss functions
        self.cr_loss = CloudRemovalLoss(config).to(self.device)
        self.gan_loss = LSGANLoss().to(self.device)
        
        # Create optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2)
        )
        
        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # Tensorboard writer
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        print(f"Training on device: {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {
            'G_total': 0,
            'G_rec': 0,
            'G_perc': 0,
            'G_style': 0,
            'G_stru': 0,
            'G_grad': 0,
            'G_gan': 0,
            'D': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            cloudy_img = batch['cloudy_img'].to(self.device)
            clean_img = batch['clean_img'].to(self.device)
            temporal_img = batch['temporal_img'].to(self.device)
            cloud_mask = batch['cloud_mask'].to(self.device)
            gradient_map = batch['gradient_map'].to(self.device)
            structure_map = batch['structure_map'].to(self.device)
            
            batch_size = cloudy_img.size(0)
            
            # =================== Train Generator ===================
            self.optimizer_G.zero_grad()
            
            # Forward pass - unpack generator outputs (pred_img, pred_grad, pred_stru)
            pred_img, pred_grad, pred_stru = self.generator(cloudy_img, cloud_mask, temporal_img)

            
            # Calculate cloud removal loss
            loss_cr, losses_dict = self.cr_loss(
                pred_img, clean_img,
                pred_grad, gradient_map,
                pred_stru, structure_map
            )
            
            # Calculate GAN loss
            pred_fake = self.discriminator(pred_img)
            loss_gan = self.gan_loss.generator_loss(pred_fake)
            
            # Total generator loss (Equation 9)
            loss_G = loss_cr + self.config.LAMBDA_GAN * loss_gan
            
            # Backward pass
            loss_G.backward()
            self.optimizer_G.step()
            
            # =================== Train Discriminator ===================
            self.optimizer_D.zero_grad()
            
            # Real images
            pred_real = self.discriminator(clean_img.detach())
            
            # Fake images
            pred_fake = self.discriminator(pred_img.detach())
            
            # Calculate discriminator loss (Equation 7)
            loss_D = self.gan_loss.discriminator_loss(pred_real, pred_fake)
            
            # Backward pass
            loss_D.backward()
            self.optimizer_D.step()
            
            # =================== Logging ===================
            epoch_losses['G_total'] += loss_G.item()
            epoch_losses['G_rec'] += losses_dict['rec'].item()
            epoch_losses['G_perc'] += losses_dict['perc'].item()
            epoch_losses['G_style'] += losses_dict['style'].item()
            epoch_losses['G_stru'] += losses_dict['stru'].item()
            epoch_losses['G_grad'] += losses_dict['grad'].item()
            epoch_losses['G_gan'] += loss_gan.item()
            epoch_losses['D'] += loss_D.item()
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}'
            })
            
            # Tensorboard logging
            if self.global_step % self.config.LOG_INTERVAL == 0:
                self.writer.add_scalar('Train/G_total', loss_G.item(), self.global_step)
                self.writer.add_scalar('Train/G_rec', losses_dict['rec'].item(), self.global_step)
                self.writer.add_scalar('Train/G_perc', losses_dict['perc'].item(), self.global_step)
                self.writer.add_scalar('Train/G_style', losses_dict['style'].item(), self.global_step)
                self.writer.add_scalar('Train/G_stru', losses_dict['stru'].item(), self.global_step)
                self.writer.add_scalar('Train/G_grad', losses_dict['grad'].item(), self.global_step)
                self.writer.add_scalar('Train/G_gan', loss_gan.item(), self.global_step)
                self.writer.add_scalar('Train/D', loss_D.item(), self.global_step)
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate(self):
        """Validate the model"""
        self.generator.eval()
        
        val_losses = []
        all_metrics = {'PSNR': [], 'SSIM': [], 'CC': [], 'RMSE': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
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
        
        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        # Log to tensorboard
        for key, value in avg_metrics.items():
            self.writer.add_scalar(f'Val/{key}', value, self.current_epoch)
        
        print(f'\nValidation Metrics:')
        for key, value in avg_metrics.items():
            print(f'  {key}: {value:.4f}')
        
        return avg_metrics
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
        }
        
        filepath = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, filepath)
        print(f'Checkpoint saved: {filepath}')
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """Load model checkpoint"""
        filepath = os.path.join(self.config.CHECKPOINT_DIR, filename)
        
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            
            print(f'Checkpoint loaded: {filepath}')
            return True
        else:
            print(f'No checkpoint found: {filepath}')
            return False
    
    def train(self):
        """Main training loop"""
        print('Starting training...')
        
        best_psnr = 0
        
        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_losses = self.train_epoch()
            
            print(f'\nEpoch {epoch+1} Training Losses:')
            print(f'  G_total: {train_losses["G_total"]:.4f}')
            print(f'  D: {train_losses["D"]:.4f}')
            
            # Validate
            val_metrics = self.validate()
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
            
            # Save best model
            if val_metrics['PSNR'] > best_psnr:
                best_psnr = val_metrics['PSNR']
                self.save_checkpoint('best_model.pth')
                print(f'New best model saved! PSNR: {best_psnr:.4f}')
        
        print('Training completed!')
        self.writer.close()

def main():
    # Create config
    config = Config()
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint if exists
    trainer.load_checkpoint('checkpoint.pth')
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()