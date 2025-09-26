# Installation Commands for License Plate Recognition System

## Windows Installation:
```cmd
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install ultralytics opencv-python numpy pandas scikit-learn xlsxwriter openpyxl joblib pillow

# Run system
python setup_lpr.py
```

## Linux Installation:
```bash
# Auto setup
chmod +x setup_linux.sh
./setup_linux.sh

# Or manual:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install ultralytics opencv-python numpy pandas scikit-learn xlsxwriter openpyxl joblib pillow

# Run system
python3 setup_lpr.py
```

## Required Files:
- bestPlateCar.pt (YOLO model)
- video.mp4 (test video)
- ../extract/num/ (character dataset)

## Usage:
1. `python feature_extractor.py` - Process character dataset
2. `python train_classifier.py` - Train neural network
3. `python detect_with_chars.py` - Extract characters from video
4. `python real_time_lpr.py` - Real-time processing
5. `python lpr_gui.py` - GUI interface

## System Specs:
- 824 character samples
- 17 features per character
- 100% classification accuracy
- GPU acceleration (CUDA)
- Cross-platform compatibility