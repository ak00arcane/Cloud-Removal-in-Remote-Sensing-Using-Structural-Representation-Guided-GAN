@echo off
REM Cloud Removal Training Script for Windows
REM Usage: run.bat [prepare|train|test|tensorboard|clean]

setlocal enabledelayedexpansion

echo ========================================
echo Cloud Removal Implementation
echo ========================================
echo.

if "%1"=="" goto usage
if "%1"=="prepare" goto prepare
if "%1"=="train" goto train
if "%1"=="test" goto test
if "%1"=="tensorboard" goto tensorboard
if "%1"=="clean" goto clean
if "%1"=="check" goto check
goto usage

:check
echo Checking environment...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    exit /b 1
)
echo [OK] Python version:
python --version

python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] PyTorch not installed!
    echo Install with: pip install torch torchvision
    exit /b 1
)
echo [OK] PyTorch installed
python -c "import torch; print(f'  Version: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"

echo.
echo Checking GPU...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
if errorlevel 1 (
    echo [WARNING] No GPU detected. Training will use CPU ^(very slow^).
) else (
    echo [OK] GPU detected
)

goto end

:prepare
echo.
echo Preparing dataset...
echo.

if not exist "data\SEN12MS-CR-raw" (
    echo [ERROR] data\SEN12MS-CR-raw directory not found!
    echo Create it with: mkdir data\SEN12MS-CR-raw
    echo Then place your .tar.gz files there
    exit /b 1
)

dir /b "data\SEN12MS-CR-raw\*.tar.gz" 2>nul | find /c /v "" > temp_count.txt
set /p tar_count=<temp_count.txt
del temp_count.txt

if "%tar_count%"=="0" (
    echo [ERROR] No .tar.gz files found in data\SEN12MS-CR-raw\
    echo Please download the dataset and place .tar.gz files there
    exit /b 1
)

echo [OK] Found %tar_count% .tar.gz files
echo Extracting... This may take a while.
echo.

python prepare_dataset.py

echo.
echo [OK] Dataset preparation complete!
goto end

:train
echo.
echo Starting training...
echo.

call :check

if not exist "data\SEN12MS-CR" (
    echo [ERROR] Dataset not prepared!
    echo Run: run.bat prepare
    exit /b 1
)

dir /b "data\SEN12MS-CR" 2>nul | find /c /v "" > temp_count.txt
set /p file_count=<temp_count.txt
del temp_count.txt

if "%file_count%"=="0" (
    echo [ERROR] Dataset directory is empty!
    echo Run: run.bat prepare
    exit /b 1
)

echo [OK] Dataset ready
echo Starting training... Press Ctrl+C to stop
echo.

python train.py

goto end

:test
echo.
echo Running tests...
echo.

if not exist "checkpoints\best_model.pth" (
    echo [ERROR] No trained model found!
    echo Train a model first with: run.bat train
    exit /b 1
)

echo [OK] Model found
python test.py

echo.
echo [OK] Testing complete!
echo Results saved in: test_results\
goto end

:tensorboard
echo.
echo Starting TensorBoard...
echo.

where tensorboard >nul 2>&1
if errorlevel 1 (
    echo [ERROR] TensorBoard not installed!
    echo Install with: pip install tensorboard
    exit /b 1
)

if not exist "logs" (
    echo [WARNING] No logs found yet. Start training first.
)

echo [OK] Starting TensorBoard on http://localhost:6006
tensorboard --logdir=.\logs --port=6006
goto end

:clean
echo.
echo Cleaning up...
echo.

set /p response="Remove checkpoints? (y/n): "
if /i "%response%"=="y" (
    if exist "checkpoints" (
        del /q "checkpoints\*.*" 2>nul
        echo [OK] Checkpoints removed
    )
)

set /p response="Remove logs? (y/n): "
if /i "%response%"=="y" (
    if exist "logs" (
        del /q "logs\*.*" 2>nul
        echo [OK] Logs removed
    )
)

set /p response="Remove test results? (y/n): "
if /i "%response%"=="y" (
    if exist "test_results" (
        del /q "test_results\*.*" 2>nul
        echo [OK] Test results removed
    )
)

echo [OK] Cleanup complete!
goto end

:usage
echo Usage: run.bat [command]
echo.
echo Commands:
echo   prepare      - Extract and prepare the dataset
echo   train        - Start model training
echo   test         - Run testing on trained model
echo   tensorboard  - Start TensorBoard visualization
echo   clean        - Clean up checkpoints and logs
echo   check        - Check environment and setup
echo.
echo Examples:
echo   run.bat prepare      # First time: prepare dataset
echo   run.bat train        # Train the model
echo   run.bat tensorboard  # Monitor training ^(in another terminal^)
echo   run.bat test         # Test trained model
goto end

:end
echo.
echo Done!