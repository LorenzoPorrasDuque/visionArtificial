import cv2
import numpy as np
import os
from pathlib import Path

class SimpleImageClassifier:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.levels = ['nivel_1', 'nivel_2', 'nivel_3', 'nivel_4', 'nivel_5']
        
    def get_simple_features(self, image_path):
        """Extract only the most important features"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Only 3 key features for simple classification
        brightness = np.mean(gray)
        contrast = np.max(gray) - np.min(gray)
        
        # Simple texture measure
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture = laplacian.var()
        
        return {
            'brightness': brightness,
            'contrast': contrast, 
            'texture': texture
        }
    
    def analyze_all_levels(self):
        """Analyze all images and create simple ranges"""
        print("🔍 ANALYZING IMAGES FOR SIMPLE RANGES")
        print("="*50)
        
        ranges = {}
        
        for level in self.levels:
            level_path = self.base_path / level
            print(f"\nAnalyzing {level}...")
            
            brightness_values = []
            contrast_values = []
            texture_values = []
            
            for img_file in level_path.glob("*.bmp"):
                features = self.get_simple_features(img_file)
                if features:
                    brightness_values.append(features['brightness'])
                    contrast_values.append(features['contrast'])
                    texture_values.append(features['texture'])
            
            if brightness_values:
                ranges[level] = {
                    'brightness': {
                        'min': min(brightness_values),
                        'max': max(brightness_values),
                        'avg': sum(brightness_values) / len(brightness_values)
                    },
                    'contrast': {
                        'min': min(contrast_values),
                        'max': max(contrast_values),
                        'avg': sum(contrast_values) / len(contrast_values)
                    },
                    'texture': {
                        'min': min(texture_values),
                        'max': max(texture_values),
                        'avg': sum(texture_values) / len(texture_values)
                    },
                    'count': len(brightness_values)
                }
                print(f"✅ Processed {len(brightness_values)} images")
            else:
                print(f"❌ No images found")
        
        return ranges
    
    def print_simple_ranges(self, ranges):
        """Print easy-to-read ranges"""
        print("\n" + "="*60)
        print("📊 SIMPLE CLASSIFICATION RANGES")
        print("="*60)
        
        for level, data in ranges.items():
            print(f"\n🎯 {level.upper()}:")
            print("-" * 30)
            print(f"Brightness:  {data['brightness']['min']:6.1f} - {data['brightness']['max']:6.1f} (avg: {data['brightness']['avg']:6.1f})")
            print(f"Contrast:    {data['contrast']['min']:6.1f} - {data['contrast']['max']:6.1f} (avg: {data['contrast']['avg']:6.1f})")
            print(f"Texture:     {data['texture']['min']:6.1f} - {data['texture']['max']:6.1f} (avg: {data['texture']['avg']:6.1f})")
            print(f"Images:      {data['count']}")
    
    def classify_new_image(self, image_path, ranges):
        """Classify a new image using simple rules"""
        features = self.get_simple_features(image_path)
        if not features:
            return None
            
        brightness = features['brightness']
        contrast = features['contrast']
        texture = features['texture']
        
        print(f"\n🔍 NEW IMAGE FEATURES:")
        print(f"Brightness: {brightness:.1f}")
        print(f"Contrast:   {contrast:.1f}")
        print(f"Texture:    {texture:.1f}")
        
        # Simple classification rules based on brightness primarily
        if brightness < 14:
            return "nivel_1", "Very dark image"
        elif brightness < 19:
            return "nivel_2", "Dark image with some texture"
        elif brightness < 25:
            return "nivel_3", "Medium brightness"
        elif brightness < 45:
            return "nivel_5", "Medium-bright image"
        else:
            return "nivel_4", "Bright image with high contrast"

def main():
    # Initialize the simple classifier
    base_path = "/home/anime/Desktop/visionArtificial/quiz1"
    classifier = SimpleImageClassifier(base_path)
    
    # Analyze all levels
    ranges = classifier.analyze_all_levels()
    
    # Print the simple ranges
    classifier.print_simple_ranges(ranges)
    
    # Create simple rules
    print("\n" + "="*60)
    print("🎯 SIMPLE CLASSIFICATION RULES")
    print("="*60)
    print("""
Based on the analysis, here are SIMPLE rules to classify new images:

1. BRIGHTNESS (most important):
   • Below 14     → NIVEL 1 (very dark)
   • 14-19       → NIVEL 2 (dark) 
   • 19-25       → NIVEL 3 (medium)
   • 25-45       → NIVEL 5 (medium-bright)
   • Above 45    → NIVEL 4 (bright)

2. CONTRAST (secondary check):
   • Low (0-20)   → Probably Nivel 1-2
   • Medium (20-60) → Probably Nivel 3-5  
   • High (60+)   → Probably Nivel 4

3. TEXTURE (final check):
   • Very low (0-2)   → Nivel 1
   • Low (2-10)       → Nivel 2-3
   • High (10+)       → Nivel 4-5
    """)
    
    # Test with sample images
    print("\n" + "="*60)
    print("🧪 TESTING CLASSIFICATION")
    print("="*60)
    
    test_images = [
        ('nivel_1', 'A10_064.bmp'),
        ('nivel_2', 'A30_133.bmp'), 
        ('nivel_3', 'AC30_173.bmp'),
        ('nivel_4', 'A50_256.bmp'),
        ('nivel_5', 'f068.bmp')
    ]
    
    for true_level, img_name in test_images:
        img_path = Path(base_path) / true_level / img_name
        if img_path.exists():
            predicted_level, reason = classifier.classify_new_image(img_path, ranges)
            correct = "✅" if predicted_level == true_level else "❌"
            print(f"\n{img_name} (True: {true_level})")
            print(f"Predicted: {predicted_level} - {reason} {correct}")

if __name__ == "__main__":
    main()