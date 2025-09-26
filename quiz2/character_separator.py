import cv2
import numpy as np
from pathlib import Path
import os

class CharacterSeparator:
    """
    Advanced character segmentation for license plates
    """
    
    def __init__(self, min_char_width=8, min_char_height=15, max_char_width=50, max_char_height=80):
        self.min_char_width = min_char_width
        self.min_char_height = min_char_height  
        self.max_char_width = max_char_width
        self.max_char_height = max_char_height
    
    def preprocess_plate(self, plate_img):
        """
        Preprocess license plate image for better character segmentation
        """
        # Convert to grayscale if needed
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
            
        # Resize to standard height for consistent processing
        target_height = 60
        aspect_ratio = gray.shape[1] / gray.shape[0]
        target_width = int(target_height * aspect_ratio)
        resized = cv2.resize(gray, (target_width, target_height))
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # Apply adaptive threshold for better character separation
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Invert if characters are darker than background
        if cv2.countNonZero(binary) > binary.size // 2:
            binary = cv2.bitwise_not(binary)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return resized, binary
    
    def find_character_contours(self, binary_img):
        """
        Find and filter potential character contours
        """
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_characters = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Filter based on size and aspect ratio
            if (self.min_char_width <= w <= self.max_char_width and
                self.min_char_height <= h <= self.max_char_height and
                0.1 <= aspect_ratio <= 1.2 and  # Characters are usually taller or square
                area >= 100):  # Minimum area threshold
                
                valid_characters.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio
                })
        
        # Sort characters from left to right
        valid_characters.sort(key=lambda x: x['bbox'][0])
        
        return valid_characters
    
    def segment_characters(self, plate_img, save_path=None, save_prefix="char"):
        """
        Segment individual characters from license plate
        Returns list of character images
        """
        original, binary = self.preprocess_plate(plate_img)
        character_data = self.find_character_contours(binary)
        
        segmented_chars = []
        
        for i, char_info in enumerate(character_data):
            x, y, w, h = char_info['bbox']
            
            # Add padding around character
            padding = 3
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(binary.shape[1], x + w + padding)
            y_end = min(binary.shape[0], y + h + padding)
            
            # Extract character ROI from binary image
            char_roi = binary[y_start:y_end, x_start:x_end]
            
            if char_roi.size > 0:
                # Normalize character size
                normalized_char = cv2.resize(char_roi, (20, 40))
                segmented_chars.append(normalized_char)
                
                # Save character if path provided
                if save_path:
                    char_filename = Path(save_path) / f"{save_prefix}_{i:02d}.png"
                    cv2.imwrite(str(char_filename), normalized_char)
        
        return segmented_chars, original, binary
    
    def visualize_segmentation(self, plate_img, show_steps=True):
        """
        Visualize the segmentation process
        """
        original, binary = self.preprocess_plate(plate_img)
        character_data = self.find_character_contours(binary)
        
        # Create visualization
        vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes around detected characters
        for i, char_info in enumerate(character_data):
            x, y, w, h = char_info['bbox']
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(vis_img, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        if show_steps:
            # Show processing steps
            cv2.imshow("Original", cv2.resize(original, (300, 100)))
            cv2.imshow("Binary", cv2.resize(binary, (300, 100)))
            cv2.imshow("Segmentation", cv2.resize(vis_img, (300, 100)))
            cv2.waitKey(1)
        
        return vis_img, len(character_data)

def test_character_separator():
    """
    Test the character separator with sample images
    """
    separator = CharacterSeparator()
    
    # Test with extracted character images if they exist
    test_dir = Path("extracted_characters")
    if test_dir.exists():
        image_files = list(test_dir.glob("*.png"))[:5]  # Test with first 5 images
        
        for img_path in image_files:
            print(f"Processing: {img_path.name}")
            img = cv2.imread(str(img_path))
            
            if img is not None:
                chars, original, binary = separator.segment_characters(img)
                vis_img, char_count = separator.visualize_segmentation(img)
                
                print(f"  - Found {char_count} characters")
                
                # Show individual characters
                for i, char in enumerate(chars):
                    cv2.imshow(f"Char {i}", cv2.resize(char, (60, 120)))
                
                cv2.waitKey(2000)  # Wait 2 seconds before next image
                cv2.destroyAllWindows()
    else:
        print("No test images found. Please run detect_with_chars.py first to generate extracted characters.")

if __name__ == "__main__":
    test_character_separator()