#!/bin/bash

# Cloud Removal Training Script
# Usage: ./run.sh [prepare|train|test|tensorboard|clean]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Cloud Removal Implementation${NC}"
echo -e "${GREEN}========================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check GPU availability
check_gpu() {
    if command_exists nvidia-smi; then
        echo -e "${GREEN}✓ GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo -e "${YELLOW}⚠ No GPU detected. Training will use CPU (very slow).${NC}"
    fi
}

# Function to check Python environment
check_environment() {
    echo -e "\n${YELLOW}Checking environment...${NC}"
    
    if ! command_exists python; then
        echo -e "${RED}✗ Python not found!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Python version:${NC} $(python --version)"
    
    if ! python -c "import torch" 2>/dev/null; then
        echo -e "${RED}✗ PyTorch not installed!${NC}"
        echo "Install with: pip install torch torchvision"
        exit 1
    fi
    
    echo -e "${GREEN}✓ PyTorch installed${NC}"
    python -c "import torch; print(f'  Version: {torch.__version__}')"
    python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
}

# Function to prepare dataset
prepare_dataset() {
    echo -e "\n${YELLOW}Preparing dataset...${NC}"
    
    if [ ! -d "data/SEN12MS-CR-raw" ]; then
        echo -e "${RED}✗ data/SEN12MS-CR-raw directory not found!${NC}"
        echo "Create it with: mkdir -p data/SEN12MS-CR-raw"
        echo "Then place your .tar.gz files there"
        exit 1
    fi
    
    tar_count=$(ls data/SEN12MS-CR-raw/*.tar.gz 2>/dev/null | wc -l)
    if [ "$tar_count" -eq 0 ]; then
        echo -e "${RED}✗ No .tar.gz files found in data/SEN12MS-CR-raw/${NC}"
        echo "Please download the dataset and place .tar.gz files there"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Found $tar_count .tar.gz files${NC}"
    echo -e "${YELLOW}Extracting... This may take a while.${NC}"
    
    python prepare_dataset.py
    
    echo -e "${GREEN}✓ Dataset preparation complete!${NC}"
}

# Function to start training
start_training() {
    echo -e "\n${YELLOW}Starting training...${NC}"
    check_environment
    check_gpu
    
    if [ ! -d "data/SEN12MS-CR" ] || [ -z "$(ls -A data/SEN12MS-CR)" ]; then
        echo -e "${RED}✗ Dataset not prepared!${NC}"
        echo "Run: ./run.sh prepare"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Dataset ready${NC}"
    echo -e "${YELLOW}Starting training... Press Ctrl+C to stop${NC}"
    echo ""
    
    python train.py
}

# Function to run testing
run_testing() {
    echo -e "\n${YELLOW}Running tests...${NC}"
    
    if [ ! -f "checkpoints/best_model.pth" ]; then
        echo -e "${RED}✗ No trained model found!${NC}"
        echo "Train a model first with: ./run.sh train"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Model found${NC}"
    python test.py
    
    echo -e "${GREEN}✓ Testing complete!${NC}"
    echo -e "Results saved in: test_results/"
}

# Function to start TensorBoard
start_tensorboard() {
    echo -e "\n${YELLOW}Starting TensorBoard...${NC}"
    
    if ! command_exists tensorboard; then
        echo -e "${RED}✗ TensorBoard not installed!${NC}"
        echo "Install with: pip install tensorboard"
        exit 1
    fi
    
    if [ ! -d "logs" ] || [ -z "$(ls -A logs)" ]; then
        echo -e "${YELLOW}⚠ No logs found yet. Start training first.${NC}"
    fi
    
    echo -e "${GREEN}✓ Starting TensorBoard on http://localhost:6006${NC}"
    tensorboard --logdir=./logs --port=6006
}

# Function to clean up
clean_up() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    read -p "Remove checkpoints? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf checkpoints/*
        echo -e "${GREEN}✓ Checkpoints removed${NC}"
    fi
    
    read -p "Remove logs? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf logs/*
        echo -e "${GREEN}✓ Logs removed${NC}"
    fi
    
    read -p "Remove test results? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf test_results/*
        echo -e "${GREEN}✓ Test results removed${NC}"
    fi
    
    echo -e "${GREEN}✓ Cleanup complete!${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  prepare      - Extract and prepare the dataset"
    echo "  train        - Start model training"
    echo "  test         - Run testing on trained model"
    echo "  tensorboard  - Start TensorBoard visualization"
    echo "  clean        - Clean up checkpoints and logs"
    echo "  check        - Check environment and setup"
    echo ""
    echo "Examples:"
    echo "  ./run.sh prepare      # First time: prepare dataset"
    echo "  ./run.sh train        # Train the model"
    echo "  ./run.sh tensorboard  # Monitor training (in another terminal)"
    echo "  ./run.sh test         # Test trained model"
}

# Main script logic
case "${1:-}" in
    prepare)
        prepare_dataset
        ;;
    train)
        start_training
        ;;
    test)
        run_testing
        ;;
    tensorboard)
        start_tensorboard
        ;;
    clean)
        clean_up
        ;;
    check)
        check_environment
        check_gpu
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

echo -e "\n${GREEN}Done!${NC}"