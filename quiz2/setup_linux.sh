#!/bin/bash
# Linux setup script for LPR system

echo "🐧 Setting up License Plate Recognition System for Linux..."

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ This script is designed for Linux"
    exit 1
fi

# Check for GPU
echo "🔍 Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    
    # Install CUDA PyTorch
    echo "🚀 Installing PyTorch with CUDA support..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "⚠️  No NVIDIA GPU detected, installing CPU version..."
    pip3 install torch torchvision torchaudio
fi

# Install other dependencies
echo "📦 Installing other dependencies..."
pip3 install ultralytics opencv-python numpy pandas scikit-learn xlsxwriter openpyxl joblib

# For GUI support
echo "🎨 Installing GUI dependencies..."
pip3 install pillow tk

# Check if display is available for GUI
if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
    echo "✅ Display detected - GUI will work"
else
    echo "⚠️  No display detected - GUI may not work (use SSH -X or run on desktop)"
fi

# Make scripts executable
chmod +x *.py

echo "✅ Setup complete! You can now run:"
echo "   python3 feature_extractor.py"
echo "   python3 train_classifier.py" 
echo "   python3 detect_with_chars.py"
echo "   python3 real_time_lpr.py"
echo "   python3 lpr_gui.py  # Requires desktop environment"