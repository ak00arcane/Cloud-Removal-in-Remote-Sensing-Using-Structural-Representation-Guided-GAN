# Structural Representation-Guided GAN for Cloud Removal

Implementation of the IEEE GRSL 2025 paper: "Structural Representation-Guided GAN for Remote Sensing Image Cloud Removal"

## 📋 Overview

This project implements a novel cloud removal framework for optical remote sensing images using:

- **Structural representation guidance** through gradient and structure branches
- **Generative Adversarial Network (GAN)** with LSGAN loss
- **Multi-temporal auxiliary images** for reliable thick cloud removal
- **Cloud Matting synthesis** for realistic training data generation

## 🎯 Key Features

- ✅ Complete implementation of the paper's architecture
- ✅ Structural representation branches (gradient + structure)
- ✅ Error feedback fusion mechanism
- ✅ GAN-based adversarial training
- ✅ Multi-temporal image support
- ✅ Cloud Matting synthesis algorithm
- ✅ Comprehensive evaluation metrics (PSNR, SSIM, CC, RMSE)
- ✅ TensorBoard logging
- ✅ Checkpoint management

## 📁 Project Structure

```
cloud_removal_project/
├── config.py                 # Configuration settings
├── train.py                  # Training script
├── test.py                   # Testing and inference
├── prepare_dataset.py        # Dataset preparation
├── requirements.txt          # Dependencies
├── models/
│   ├── __init__.py
│   ├── generator.py         # Cloud removal network
│   ├── discriminator.py     # Discriminator network
│   └── losses.py            # Loss functions
├── utils/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset and dataloaders
│   ├── cloud_synthesis.py   # Cloud Matting synthesis
│   └── metrics.py           # Evaluation metrics
├── data/
│   ├── SEN12MS-CR-raw/      # Downloaded .tar.gz files
│   └── SEN12MS-CR/          # Extracted dataset
├── checkpoints/              # Model checkpoints
├── logs/                     # TensorBoard logs
└── test_results/             # Test outputs
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create project directory
mkdir cloud_removal_project && cd cloud_removal_project

# Create all necessary subdirectories
mkdir -p data/SEN12MS-CR-raw data/SEN12MS-CR models utils checkpoints logs test_results

# Create __init__.py files
touch models/__init__.py utils/__init__.py
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
conda create -n cloud_removal python=3.9
conda activate cloud_removal

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 3. Prepare Dataset

**You already have the dataset downloaded!** Now extract it:

```bash
# Move your downloaded .tar.gz files
mv /path/to/your/downloads/*.tar.gz ./data/SEN12MS-CR-raw/

# Extract dataset (this will take a while)
python prepare_dataset.py
```

**Note:** You can start with just 1-2 seasons if disk space is limited:

- Priority: Extract `ROIs1868_summer_s2.tar.gz` and `ROIs1158_spring_s2.tar.gz`

### 4. Train the Model

```bash
# Start training
python train.py

# Monitor with TensorBoard (in another terminal)
tensorboard --logdir=./logs
```

### 5. Test the Model

```bash
# Test on test set
python test.py

# Results will be saved in ./test_results/
```

## ⚙️ Configuration

Edit `config.py` to customize training:

```python
# Key parameters
BATCH_SIZE = 4              # Reduce to 2 or 1 if OOM
IMG_SIZE = 256              # Image size
NUM_EPOCHS = 100            # Training epochs
LEARNING_RATE = 0.0002      # Learning rate

# Loss weights
LAMBDA_REC = 1.0           # Reconstruction
LAMBDA_PERC = 0.1          # Perceptual
LAMBDA_STYLE = 250.0       # Style
LAMBDA_STRU = 1.0          # Structure
LAMBDA_GRAD = 1.0          # Gradient
LAMBDA_GAN = 0.01          # GAN
```

## 📊 Expected Results

After training on SEN12MS-CR dataset:

| Metric | Expected Value |
| ------ | -------------- |
| PSNR   | 32-35 dB       |
| SSIM   | 0.90-0.94      |
| CC     | 0.93-0.96      |
| RMSE   | 0.02-0.04      |

## 🔧 Troubleshooting

### Out of Memory (OOM)

```python
# In config.py
BATCH_SIZE = 1              # Reduce batch size
IMG_SIZE = 128              # Reduce image size
```

### Slow Training

```python
# In train.py, modify create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(config, num_workers=2)
```

### Dataset Not Loading

```bash
# Verify extraction
python prepare_dataset.py

# Check for .tif files
find data/SEN12MS-CR/ -name "*.tif" | wc -l
```

## 📝 Using Custom Images

```python
from config import Config
from test import Tester

config = Config()
tester = Tester(config, checkpoint_path='checkpoints/best_model.pth')

# Infer on custom image
tester.infer_single_image(
    cloudy_path='path/to/cloudy.png',
    temporal_path='path/to/temporal.png',
    save_path='output_clean.png'
)
```

## 🎓 Citation

If you use this code, please cite the original paper:

```bibtex
@article{yang2025structural,
  title={Structural Representation-Guided GAN for Remote Sensing Image Cloud Removal},
  author={Yang, Jiajun and Wang, Wenjing and Chen, Keyan and Liu, Liqin and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={22},
  pages={6002105},
  year={2025},
  publisher={IEEE}
}
```

## 📦 Model Architecture

### Generator (Cloud Removal Network)

- **Encoder**: 6-level encoder with multi-scale feature extraction
- **Gradient Branch**: First 3 encoder levels → gradient map prediction
- **Structure Branch**: Last 3 encoder levels → structure map prediction
- **Error Feedback**: Fuses branch features into decoder
- **Decoder**: 6-level decoder with skip connections

### Discriminator

- 5-layer convolutional network
- PatchGAN architecture
- LSGAN loss for stable training

### Loss Functions

1. **Reconstruction Loss** (L1): Pixel-level consistency
2. **Perceptual Loss**: VGG16-based feature matching
3. **Style Loss**: Gram matrix matching
4. **Structure Loss**: Edge/structure map consistency
5. **Gradient Loss**: Gradient map consistency
6. **Adversarial Loss**: LSGAN loss

## 🔍 Training Details

- **Optimizer**: Adam (lr=0.0002, β1=0.5, β2=0.999)
- **Batch Size**: 4
- **Epochs**: 100
- **Image Size**: 256×256
- **Data Augmentation**: Random flip, rotation
- **Mixed Training**: Synthetic + real cloud images

## 📈 Monitoring Training

TensorBoard logs include:

- Training losses (G_total, G_rec, G_perc, G_style, G_stru, G_grad, G_gan, D)
- Validation metrics (PSNR, SSIM, CC, RMSE)
- Sample images

```bash
tensorboard --logdir=./logs --port=6006
```

## 💾 Checkpoints

Models are saved:

- Every 5 epochs: `checkpoint_epoch_X.pth`
- Best model: `best_model.pth` (highest validation PSNR)
- Latest: `checkpoint.pth` (for resuming training)

## 🤝 Contributing

Feel free to open issues or submit pull requests for:

- Bug fixes
- Performance improvements
- Additional features
- Documentation improvements

## 📄 License

This implementation is for research purposes. Please refer to the original paper for licensing information.

## 🙏 Acknowledgments

- Original paper authors: Yang et al.
- SEN12MS-CR dataset: TUM München
- PyTorch community

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.

---

**Note**: This implementation requires significant computational resources. A GPU with at least 8GB VRAM is recommended for training.
