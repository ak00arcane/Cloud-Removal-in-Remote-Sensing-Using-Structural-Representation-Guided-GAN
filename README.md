Cloud-Removal-in-Remote-Sensing-Using-Structural-Representation-Guided-GAN

Abstract 

Clouds frequently obstruct satellite images, limiting their use in applications such as agriculture, disaster monitoring, and land-cover analysis. This project implements a Structural Representation-Guided GAN (SRG-GAN) to remove clouds and reconstruct the underlying ground surface from Sentinel-2 imagery. The model integrates multi-scale structural cues, gradient information, and perceptual losses to preserve terrain details while performing cloud removal. Using the SEN12MS-CR dataset, the GAN is trained for 100 epochs with real and synthetic cloud data. Quantitative results demonstrate strong performance with PSNR of 30.14 dB, SSIM of 0.809, CC of 0.799, and RMSE of 0.064. Qualitative analysis shows successful reconstruction for medium and thin clouds, while extremely dense cloud regions remain challenging. The implementation includes full training, evaluation, and inference pipelines. This work demonstrates the potential of deep generative models for reliable cloud removal in remote sensing.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Team Members
Hannah Cinderella - Reg No:23MIA1043
Akshaya G.K — Reg No: 23MIA1127
Kamalesh.V — Reg No: 23MIA1035

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Base Paper Reference
Yang, et al., “Structural Representation-Guided GAN for Cloud Removal,”
IEEE Geoscience and Remote Sensing Letters (GRSL), 2025.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Tools and Libraries Used
Programming Language: Python
Deep Learning Framework: PyTorch
IDE: Visual Studio Code
GPU Support: NVIDIA CUDA
Visualization: TensorBoard, Matplotlib
Image Processing: OpenCV, NumPy
Dataset Handling: TorchVision, custom loaders

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Steps to Execute the Code

Clone the Repository
git clone <your-github-repo-link>
cd cloud-removal-SRGGAN

Install Dependencies
pip install -r requirements.txt

Prepare Dataset

Place SEN12MS-CR dataset inside:
./data/SEN12MS_CR/
Dataset Link:: https://dataserv.ub.tum.de/index.php/s/m1554803

Train the Model
python train.py --config config.yaml

Run Inference
python inference.py --input cloudy.png --output result.png

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

View Training Metrics
tensorboard --logdir runs/
Description of the Dataset
Dataset Used: SEN12MS-CR (Cloud Removal Subset)
Content: Paired cloudy and cloud-free Sentinel-2 RGB images
Resolution: 256 × 256 pixel patches

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Split:
Training: 58 pairs
Validation: 12 pairs
Testing: 14 pairs

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Characteristics:
Includes thin, medium, and thick cloud variations
Provides clean reference targets for supervised learning

Result Summary
Quantitative Results:
PSNR: 30.14 dB
SSIM: 0.809
CC: 0.799
RMSE: 0.064

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Qualitative Results:
Successful cloud removal for most medium and thin cloud regions
Reconstruction is smooth but less accurate under extremely dense clouds

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Visual comparison includes:
• Cloudy Input
• Model Output
• Ground Truth

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
YouTube Demo Link
video link ::(https://youtu.be/CypFTo34CUU)
