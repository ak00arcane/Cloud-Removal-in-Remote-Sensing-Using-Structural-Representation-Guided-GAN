import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16FeatureExtractor(nn.Module):
    """VGG16 for perceptual and style loss"""
    def __init__(self):
        super().__init__()
        # Use weights parameter instead of pretrained
        from torchvision.models import vgg16, VGG16_Weights
        vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg16_model.features
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Layer indices for feature extraction
        self.layer_indices = [3, 8, 15, 22, 29]  # relu1_2, relu2_2, relu3_3, relu4_3, relu5_3
    
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features

class CloudRemovalLoss(nn.Module):
    """Combined loss for cloud removal network"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vgg = VGG16FeatureExtractor()
        self.l1_loss = nn.L1Loss()
        
    def gram_matrix(self, x):
        """Compute Gram matrix for style loss"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def reconstruction_loss(self, pred, target):
        """L1 reconstruction loss (Equation 1)"""
        return self.l1_loss(pred, target)
    
    def perceptual_loss(self, pred, target):
        """Perceptual loss using VGG features (Equation 2)"""
        # Normalize images for VGG
        pred_norm = (pred + 1) / 2  # Convert from [-1, 1] to [0, 1]
        target_norm = (target + 1) / 2
        
        # Repeat channels if grayscale
        if pred_norm.shape[1] == 1:
            pred_norm = pred_norm.repeat(1, 3, 1, 1)
            target_norm = target_norm.repeat(1, 3, 1, 1)
        
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            loss += self.l1_loss(pred_feat, target_feat)
        
        return loss / len(pred_features)
    
    def style_loss(self, pred, target):
        """Style loss using Gram matrices (Equation 3)"""
        # Normalize images for VGG
        pred_norm = (pred + 1) / 2
        target_norm = (target + 1) / 2
        
        if pred_norm.shape[1] == 1:
            pred_norm = pred_norm.repeat(1, 3, 1, 1)
            target_norm = target_norm.repeat(1, 3, 1, 1)
        
        pred_features = self.vgg(pred_norm)
        target_features = self.vgg(target_norm)
        
        loss = 0
        for pred_feat, target_feat in zip(pred_features, target_features):
            pred_gram = self.gram_matrix(pred_feat)
            target_gram = self.gram_matrix(target_feat)
            loss += self.l1_loss(pred_gram, target_gram)
        
        return loss / len(pred_features)
    
    def structure_loss(self, pred_stru, target_stru):
        """Structure loss (Equation 4)"""
        return self.l1_loss(pred_stru, target_stru)
    
    def gradient_loss(self, pred_grad, target_grad):
        """Gradient loss (Equation 5)"""
        return self.l1_loss(pred_grad, target_grad)
    
    def forward(self, pred_img, target_img, pred_grad, target_grad, pred_stru, target_stru):
        """
        Compute total cloud removal loss (Equation 6)
        """
        loss_rec = self.reconstruction_loss(pred_img, target_img)
        loss_perc = self.perceptual_loss(pred_img, target_img)
        loss_style = self.style_loss(pred_img, target_img)
        loss_stru = self.structure_loss(pred_stru, target_stru)
        loss_grad = self.gradient_loss(pred_grad, target_grad)
        
        total_loss = (
            self.config.LAMBDA_REC * loss_rec +
            self.config.LAMBDA_PERC * loss_perc +
            self.config.LAMBDA_STYLE * loss_style +
            self.config.LAMBDA_STRU * loss_stru +
            self.config.LAMBDA_GRAD * loss_grad
        )
        
        losses = {
            'total': total_loss,
            'rec': loss_rec,
            'perc': loss_perc,
            'style': loss_style,
            'stru': loss_stru,
            'grad': loss_grad
        }
        
        return total_loss, losses

class LSGANLoss(nn.Module):
    """Least Squares GAN Loss"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def discriminator_loss(self, real_pred, fake_pred):
        """
        Discriminator loss (Equation 7)
        L_D = E[(D(I_fake))^2] + E[(D(I_gt) - 1)^2]
        """
        real_loss = self.mse_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = self.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        return (real_loss + fake_loss) / 2
    
    def generator_loss(self, fake_pred):
        """
        Generator loss (Equation 8)
        L_G = E[(D(I_fake) - 1)^2]
        """
        return self.mse_loss(fake_pred, torch.ones_like(fake_pred))