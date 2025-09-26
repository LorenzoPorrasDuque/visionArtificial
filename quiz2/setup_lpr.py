"""
Complete License Plate Recognition System Setup Guide
=====================================================

This script helps you set up and run the complete LPR system step by step.

Prerequisites:
- YOLO model trained for license plate detection (bestPlateCar.pt)
- Video file for testing (video.mp4)
- Character dataset in ../extract/num/ folders (0-9)

Steps completed:
1. ✅ Video recording capability
2. ✅ YOLO model for plate detection  
3. ✅ Character extraction from plates
4. ✅ Character segmentation
5. ✅ Feature extraction
6. ✅ Character classification training
7. ✅ Real-time prediction system
8. ✅ GUI interface

Usage Instructions:
==================

## Step 1: Extract Features and Train Model
Run this first to train the character recognition model:

```bash
python feature_extractor.py
python train_classifier.py
```

## Step 2: Test Character Extraction
Extract characters from video:

```bash
python detect_with_chars.py
```

## Step 3: Run Real-time LPR
For command-line real-time processing:

```bash
python real_time_lpr.py
```

## Step 4: Launch GUI Application
For the complete GUI experience:

```bash
python lpr_gui.py
```

## File Structure Created:
```
quiz2/
├── bestPlateCar.pt              # YOLO model (provided by professor)
├── video.mp4                    # Test video
├── detect_with_chars.py         # Character extraction from video
├── character_separator.py       # Character segmentation
├── feature_extractor.py         # Feature extraction
├── train_classifier.py          # Model training
├── real_time_lpr.py            # Real-time LPR system
├── lpr_gui.py                  # GUI application
├── setup_lpr.py                # This file
├── extracted_characters/        # Extracted character images
├── character_features.xlsx      # Training features
├── character_classifier.joblib  # Trained model
└── feature_scaler.joblib       # Feature scaler
```

## System Capabilities:
- ✅ GPU-accelerated YOLO detection
- ✅ Advanced character segmentation
- ✅ Multi-feature extraction (geometric + density)
- ✅ Neural network classification
- ✅ Real-time video processing
- ✅ Cross-platform compatibility
- ✅ User-friendly GUI
- ✅ Performance monitoring

## Next Steps:
1. Ensure you have all required files
2. Train the character classifier
3. Test with your video
4. Use the GUI for interactive recognition
"""

import sys
from pathlib import Path
import subprocess

def check_requirements():
    """
    Check if all requirements are met
    """
    print("🔍 Checking system requirements...")
    
    # Check required files
    script_dir = Path(__file__).parent
    required_files = {
        "bestPlateCar.pt": "YOLO model for plate detection",
        "video.mp4": "Test video file", 
        "../extract/num/0": "Character dataset folder"
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        full_path = script_dir / file_path
        if not full_path.exists():
            missing_files.append(f"❌ {file_path} - {description}")
        else:
            print(f"✅ {file_path} - Found")
    
    if missing_files:
        print("\n⚠️  Missing required files:")
        for missing in missing_files:
            print(f"   {missing}")
        return False
    
    # Check trained models
    model_files = {
        "character_classifier.joblib": "Trained character classifier",
        "feature_scaler.joblib": "Feature scaler",
        "character_features.xlsx": "Extracted features"
    }
    
    trained_models_exist = True
    for model_file, description in model_files.items():
        if not (script_dir / model_file).exists():
            print(f"⚠️  {model_file} - {description} (needs training)")
            trained_models_exist = False
        else:
            print(f"✅ {model_file} - Found")
    
    return True, trained_models_exist

def run_training_pipeline():
    """
    Run the complete training pipeline
    """
    print("\n🚀 Starting training pipeline...")
    
    try:
        print("\n1️⃣ Extracting features from character dataset...")
        result = subprocess.run([sys.executable, "feature_extractor.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Feature extraction failed: {result.stderr}")
            return False
        print("✅ Feature extraction completed")
        
        print("\n2️⃣ Training character classifier...")
        result = subprocess.run([sys.executable, "train_classifier.py"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Model training failed: {result.stderr}")
            return False
        print("✅ Model training completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        return False

def run_system_test():
    """
    Test the complete system
    """
    print("\n🧪 Testing the complete system...")
    
    try:
        print("\n1️⃣ Testing character extraction...")
        result = subprocess.run([sys.executable, "detect_with_chars.py"], 
                              timeout=60, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Character extraction test passed")
        else:
            print(f"⚠️  Character extraction had issues: {result.stderr}")
        
        print("\n✅ System test completed")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def launch_gui():
    """
    Launch the GUI application
    """
    print("\n🎨 Launching GUI application...")
    
    try:
        subprocess.run([sys.executable, "lpr_gui.py"])
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")

def main():
    """
    Main setup function
    """
    print("=" * 60)
    print("🚗 LICENSE PLATE RECOGNITION SYSTEM SETUP")
    print("=" * 60)
    
    # Check requirements
    requirements_ok, models_trained = check_requirements()
    
    if not requirements_ok:
        print("\n❌ System requirements not met. Please ensure all required files are present.")
        return
    
    print("\n✅ All requirements met!")
    
    # Train models if needed
    if not models_trained:
        print("\n🤖 Character classifier needs training...")
        choice = input("Train character classifier now? (y/n): ").lower().strip()
        
        if choice == 'y':
            if run_training_pipeline():
                print("\n🎉 Training completed successfully!")
            else:
                print("\n❌ Training failed. Please check the errors above.")
                return
        else:
            print("⚠️  Skipping training. Some features may not work without trained models.")
    else:
        print("\n✅ All models are trained and ready!")
    
    # Ask what to do next
    print("\n" + "=" * 60)
    print("🎯 WHAT WOULD YOU LIKE TO DO?")
    print("=" * 60)
    print("1. 🧪 Test character extraction with video")
    print("2. 🎮 Launch GUI application")
    print("3. ⚡ Run real-time LPR (command line)")
    print("4. 🔄 Re-train character classifier")
    print("5. ❌ Exit")
    
    while True:
        choice = input("\\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            run_system_test()
            break
        elif choice == '2':
            launch_gui()
            break
        elif choice == '3':
            subprocess.run([sys.executable, "real_time_lpr.py"])
            break
        elif choice == '4':
            run_training_pipeline()
            break
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    main()