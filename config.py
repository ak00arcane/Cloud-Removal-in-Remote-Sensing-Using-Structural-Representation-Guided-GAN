import os
import torch

class Config:
    # ==========================
    # Dataset Paths
    # ==========================
    DATASET_ROOT = './data'              # Root folder containing asiaWest_n
    PROCESSED_DATA_ROOT = './data/processed'
    SYNTHETIC_DATA_ROOT = './data/synthetic'

    # ==========================
    # Training Parameters
    # ==========================
    BATCH_SIZE = 4                       # Batch size for training
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0002
    BETA1 = 0.5
    BETA2 = 0.999

    # ==========================
    # Image Parameters
    # ==========================
    IMG_SIZE = 256
    INPUT_CHANNELS = 6                  # cloudy (3) + mask (1) + temporal (8)
    OUTPUT_CHANNELS = 3                  # RGB output

    # ==========================
    # Network Architecture
    # ==========================
    ENCODER_CHANNELS = [64, 128, 256, 512]
    DECODER_CHANNELS = [512, 256, 128, 64]

    # ==========================
    # Loss Weights
    # ==========================
    LAMBDA_REC = 1.0        # Reconstruction loss
    LAMBDA_PERC = 0.1       # Perceptual loss
    LAMBDA_STYLE = 250.0    # Style loss
    LAMBDA_STRU = 1.0       # Structure loss
    LAMBDA_GRAD = 1.0       # Gradient loss
    LAMBDA_GAN = 0.01       # GAN loss

    # ==========================
    # Device and Logging
    # ==========================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_DIR = './checkpoints'
    LOG_DIR = './logs'
    SAVE_INTERVAL = 5
    LOG_INTERVAL = 100

    # ==========================
    # Data Augmentation
    # ==========================
    USE_AUGMENTATION = True

    # ==========================
    # Directory Setup
    # ==========================
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)
    os.makedirs(SYNTHETIC_DATA_ROOT, exist_ok=True)
