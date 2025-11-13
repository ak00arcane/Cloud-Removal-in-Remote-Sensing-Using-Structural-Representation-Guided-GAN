import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from models.generator import CloudRemovalGenerator
from models.discriminator import Discriminator
from utils.data_loader import SEN12MSCRDataset
from utils.metrics import calculate_metrics
from config import Config

def train_model(config=None):
    if config is None:
        config = Config()
    
    # Create data loaders with augmentation
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize models
    generator = CloudRemovalGenerator(config).to(config.DEVICE)
    discriminator = Discriminator(config).to(config.DEVICE)
    
    # Optimizers
    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    optimizer_D = optim.Adam(
        discriminator.parameters(),
        lr=config.LEARNING_RATE,
        betas=(config.BETA1, config.BETA2)
    )
    
    # Learning rate schedulers
    scheduler_G = optim.lr_scheduler.StepLR(
        optimizer_G,
        step_size=config.LR_DECAY_START,
        gamma=config.LR_DECAY_FACTOR
    )
    
    scheduler_D = optim.lr_scheduler.StepLR(
        optimizer_D,
        step_size=config.LR_DECAY_START,
        gamma=config.LR_DECAY_FACTOR
    )
    
    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()
    criterion_perceptual = VGGPerceptualLoss().to(config.DEVICE)
    
    # Tensorboard writer
    writer = SummaryWriter('logs/improved_model')
    
    # Training loop
    best_psnr = 0
    global_step = 0
    
    for epoch in range(config.NUM_EPOCHS):
        generator.train()
        discriminator.train()
        
        # Training phase
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        for batch in train_progress:
            # Get batch data
            cloudy_img = batch['cloudy_img'].to(config.DEVICE)
            clean_img = batch['clean_img'].to(config.DEVICE)
            temporal_img = batch['temporal_img'].to(config.DEVICE)
            cloud_mask = batch['cloud_mask'].to(config.DEVICE)
            
            # Train Generator
            optimizer_G.zero_grad()
            
            # Generate cloud-free image
            pred_img, pred_grad, pred_stru = generator(cloudy_img, cloud_mask, temporal_img)
            
            # Adversarial loss
            pred_real = discriminator(clean_img)
            pred_fake = discriminator(pred_img)
            
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            
            # Pixel-wise loss
            loss_pixel = criterion_pixel(pred_img, clean_img)
            
            # Perceptual loss
            loss_perceptual = criterion_perceptual(pred_img, clean_img)
            
            # Additional losses (gradient and structure)
            loss_gradient = criterion_pixel(pred_grad, batch['gradient_map'].to(config.DEVICE))
            loss_structure = criterion_pixel(pred_stru, batch['structure_map'].to(config.DEVICE))
            
            # Total generator loss
            loss_G = (config.LAMBDA_GAN * loss_GAN +
                     config.LAMBDA_L1 * loss_pixel +
                     config.LAMBDA_PERCEPTUAL * loss_perceptual +
                     config.LAMBDA_GRADIENT * loss_gradient +
                     config.LAMBDA_STRUCTURE * loss_structure)
            
            loss_G.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            pred_real = discriminator(clean_img)
            pred_fake = discriminator(pred_img.detach())
            
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()
            
            # Update progress bar
            train_progress.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}'
            })
            
            # Log to tensorboard
            writer.add_scalar('Loss/Generator', loss_G.item(), global_step)
            writer.add_scalar('Loss/Discriminator', loss_D.item(), global_step)
            writer.add_scalar('Loss/Pixel', loss_pixel.item(), global_step)
            writer.add_scalar('Loss/Perceptual', loss_perceptual.item(), global_step)
            
            global_step += 1
        
        # Validation phase
        generator.eval()
        val_metrics = {'psnr': [], 'ssim': [], 'cc': [], 'rmse': []}
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                cloudy_img = batch['cloudy_img'].to(config.DEVICE)
                clean_img = batch['clean_img'].to(config.DEVICE)
                temporal_img = batch['temporal_img'].to(config.DEVICE)
                cloud_mask = batch['cloud_mask'].to(config.DEVICE)
                
                pred_img, _, _ = generator(cloudy_img, cloud_mask, temporal_img)
                
                # Calculate metrics
                metrics = calculate_metrics(pred_img, clean_img)
                for k, v in metrics.items():
                    val_metrics[k].append(v)
        
        # Average validation metrics
        avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
        
        # Log validation metrics
        for k, v in avg_metrics.items():
            writer.add_scalar(f'Validation/{k}', v, epoch)
        
        # Save best model
        if avg_metrics['psnr'] > best_psnr:
            best_psnr = avg_metrics['psnr']
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'best_psnr': best_psnr,
                'global_step': global_step
            }, 'checkpoints/improved_best_model.pth')
        
        # Save periodic checkpoint
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'psnr': avg_metrics['psnr'],
                'global_step': global_step
            }, f'checkpoints/improved_checkpoint_epoch_{epoch+1}.pth')
        
        # Update learning rates
        scheduler_G.step()
        scheduler_D.step()
    
    writer.close()

if __name__ == '__main__':
    train_model()