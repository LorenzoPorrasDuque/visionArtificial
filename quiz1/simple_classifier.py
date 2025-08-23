import cv2
import numpy as np

def classify_image_simple(image_path):
    """
    Simple function to classify an image into one of 5 levels
    Returns: (level, confidence, features)
    """
    
    # Load and process image
    img = cv2.imread(image_path)
    if img is None:
        return None, 0, None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract 3 key features
    brightness = np.mean(gray)
    contrast = np.max(gray) - np.min(gray)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture = laplacian.var()
    
    features = {
        'brightness': brightness,
        'contrast': contrast,
        'texture': texture
    }
    
    # SIMPLE CLASSIFICATION RULES
    # Based on brightness (most reliable feature)
    
    if brightness < 14:
        level = "nivel_1"
        confidence = "HIGH" if contrast < 20 and texture < 2 else "MEDIUM"
        
    elif brightness < 19:
        level = "nivel_2" 
        confidence = "HIGH" if 25 <= contrast <= 35 and 2 <= texture <= 6 else "MEDIUM"
        
    elif brightness < 25:
        level = "nivel_3"
        confidence = "HIGH" if contrast > 50 and texture > 6 else "MEDIUM"
        
    elif brightness < 45:
        level = "nivel_5"
        confidence = "HIGH" if 80 <= contrast <= 100 and 7 <= texture <= 11 else "MEDIUM"
        
    else:  # brightness >= 45
        level = "nivel_4"
        confidence = "HIGH" if contrast > 150 and texture > 30 else "MEDIUM"
    
    return level, confidence, features

def print_classification_ranges():
    """Print the simple ranges for manual reference"""
    print("="*60)
    print("📊 SIMPLE IMAGE CLASSIFICATION RANGES")
    print("="*60)
    print()
    print("🎯 NIVEL 1 (Very Dark Images):")
    print("   Brightness: 13.0 - 13.7")
    print("   Contrast:   16.0 - 17.0") 
    print("   Texture:    0.9 - 1.2")
    print()
    print("🎯 NIVEL 2 (Dark Images):")
    print("   Brightness: 15.7 - 18.9")
    print("   Contrast:   25.0 - 33.0")
    print("   Texture:    2.6 - 5.9")
    print()
    print("🎯 NIVEL 3 (Medium Brightness):")
    print("   Brightness: 21.0 - 21.3")
    print("   Contrast:   54.0 - 55.0")
    print("   Texture:    6.6 - 6.8")
    print()
    print("🎯 NIVEL 4 (Bright Images):")
    print("   Brightness: 41.8 - 61.5")
    print("   Contrast:   161.0 - 232.0")
    print("   Texture:    30.4 - 65.3")
    print()
    print("🎯 NIVEL 5 (Medium-Bright Images):")
    print("   Brightness: 37.6 - 38.6")
    print("   Contrast:   90.0 - 95.0")
    print("   Texture:    7.3 - 10.1")
    print()
    print("="*60)
    print("🔑 QUICK CLASSIFICATION RULES:")
    print("="*60)
    print("1. Brightness < 14    → NIVEL 1")
    print("2. Brightness 14-19   → NIVEL 2") 
    print("3. Brightness 19-25   → NIVEL 3")
    print("4. Brightness 25-45   → NIVEL 5")
    print("5. Brightness > 45    → NIVEL 4")
    print("="*60)

# Example usage
if __name__ == "__main__":
    # Print the ranges
    print_classification_ranges()
    
    # Test with sample images
    test_images = [
        "/home/anime/Desktop/visionArtificial/quiz1/nivel_1/A10_064.bmp",
        "/home/anime/Desktop/visionArtificial/quiz1/nivel_2/A30_133.bmp",
        "/home/anime/Desktop/visionArtificial/quiz1/nivel_3/AC30_173.bmp",
        "/home/anime/Desktop/visionArtificial/quiz1/nivel_4/A50_256.bmp", 
        "/home/anime/Desktop/visionArtificial/quiz1/nivel_5/f068.bmp"
    ]
    
    print("\n🧪 TESTING SIMPLE CLASSIFIER:")
    print("-" * 40)
    
    for img_path in test_images:
        level, confidence, features = classify_image_simple(img_path)
        if level:
            img_name = img_path.split('/')[-1]
            print(f"\n📸 {img_name}")
            print(f"   Level: {level}")
            print(f"   Confidence: {confidence}")
            print(f"   Brightness: {features['brightness']:.1f}")
            print(f"   Contrast: {features['contrast']:.1f}")
            print(f"   Texture: {features['texture']:.1f}")
