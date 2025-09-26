import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import xlsxwriter
from glob import glob
import os

class FeatureExtractor:
    """
    Extract features from character images for machine learning training
    """
    
    def __init__(self):
        self.feature_names = [
            'area', 'perimeter', 'circularity',
            'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
            'aspect_ratio', 'extent', 'solidity', 'density_top_left',
            'density_top_right', 'density_bottom_left', 'density_bottom_right'
        ]
    
    def binarize_image(self, img_gray):
        """
        Binarize grayscale image
        """
        return cv2.inRange(img_gray, 0, 127)
    
    def extract_contours(self, img_binary):
        """
        Extract the largest contour from binary image
        """
        contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = []
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
        return cnt
    
    def extract_geometric_features(self, img_binary, contour):
        """
        Extract geometric features from contour
        """
        features = {}
        
        if len(contour) > 0:
            # Basic geometric features
            features['area'] = cv2.contourArea(contour)
            features['perimeter'] = cv2.arcLength(contour, True)
            
            # Avoid division by zero
            if features['perimeter'] > 0:
                features['circularity'] = 4 * np.pi * features['area'] / (features['perimeter'] ** 2)
            else:
                features['circularity'] = 0
            
            # Hu moments
            M = cv2.moments(contour)
            if M['m00'] != 0:  # Avoid division by zero
                Hu = cv2.HuMoments(M)
                for i in range(7):
                    features[f'hu{i+1}'] = Hu[i][0] if not np.isnan(Hu[i][0]) else 0
            else:
                for i in range(7):
                    features[f'hu{i+1}'] = 0
            
            # Bounding rectangle features
            x, y, w, h = cv2.boundingRect(contour)
            features['aspect_ratio'] = w / h if h > 0 else 0
            
            # Extent (contour area / bounding rectangle area)
            rect_area = w * h
            features['extent'] = features['area'] / rect_area if rect_area > 0 else 0
            
            # Solidity (contour area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            features['solidity'] = features['area'] / hull_area if hull_area > 0 else 0
        else:
            # Default values if no contour found
            for key in ['area', 'perimeter', 'circularity', 'aspect_ratio', 'extent', 'solidity']:
                features[key] = 0
            for i in range(7):
                features[f'hu{i+1}'] = 0
        
        return features
    
    def extract_density_features(self, img_binary):
        """
        Extract density features from different regions of the image
        """
        h, w = img_binary.shape
        half_h, half_w = h // 2, w // 2
        
        # Divide image into quadrants
        top_left = img_binary[:half_h, :half_w]
        top_right = img_binary[:half_h, half_w:]
        bottom_left = img_binary[half_h:, :half_w]
        bottom_right = img_binary[half_h:, half_w:]
        
        features = {}
        features['density_top_left'] = cv2.countNonZero(top_left) / (half_h * half_w) if (half_h * half_w) > 0 else 0
        features['density_top_right'] = cv2.countNonZero(top_right) / (half_h * (w - half_w)) if (half_h * (w - half_w)) > 0 else 0
        features['density_bottom_left'] = cv2.countNonZero(bottom_left) / ((h - half_h) * half_w) if ((h - half_h) * half_w) > 0 else 0
        features['density_bottom_right'] = cv2.countNonZero(bottom_right) / ((h - half_h) * (w - half_w)) if ((h - half_h) * (w - half_w)) > 0 else 0
        
        return features
    
    def extract_all_features(self, img_gray):
        """
        Extract all features from a grayscale character image
        """
        # Preprocess image
        img_binary = self.binarize_image(img_gray)
        
        # Resize to standard size for consistent feature extraction
        img_binary = cv2.resize(img_binary, (20, 40))
        
        # Extract contour
        contour = self.extract_contours(img_binary)
        
        # Extract features
        geometric_features = self.extract_geometric_features(img_binary, contour)
        density_features = self.extract_density_features(img_binary)
        
        # Combine all features
        all_features = {**geometric_features, **density_features}
        
        # Return features as ordered list
        feature_vector = [all_features[name] for name in self.feature_names]
        
        return feature_vector
    
    def process_dataset(self, dataset_path, output_excel_path):
        """
        Process the entire dataset and save features to Excel
        """
        dataset_path = Path(dataset_path)
        
        # Initialize data storage
        all_features = []
        all_labels = []
        
        print("Processing dataset for feature extraction...")
        
        # Process each digit folder
        for digit in range(10):
            digit_folder = dataset_path / str(digit)
            if digit_folder.exists():
                print(f"Processing digit {digit}...")
                
                # Process all images in digit folder
                image_files = list(digit_folder.glob("*.png"))
                
                for img_path in image_files:
                    try:
                        # Load image
                        img = cv2.imread(str(img_path), 0)  # Load as grayscale
                        
                        if img is not None:
                            # Extract features
                            features = self.extract_all_features(img)
                            
                            # Store features and label
                            all_features.append(features)
                            all_labels.append(digit)
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                
                print(f"  - Processed {len(image_files)} images for digit {digit}")
        
        # Save to Excel
        self.save_features_to_excel(all_features, all_labels, output_excel_path)
        
        print(f"Feature extraction completed. Saved {len(all_features)} samples to {output_excel_path}")
        
        return all_features, all_labels
    
    def save_features_to_excel(self, features, labels, output_path):
        """
        Save features and labels to Excel file
        """
        workbook = xlsxwriter.Workbook(output_path)
        worksheet = workbook.add_worksheet('features')
        
        # Write header
        worksheet.write(0, 0, 'label')
        for i, feature_name in enumerate(self.feature_names):
            worksheet.write(0, i + 1, feature_name)
        
        # Write data
        for row, (feature_vector, label) in enumerate(zip(features, labels), 1):
            worksheet.write(row, 0, label)
            for col, feature_value in enumerate(feature_vector, 1):
                worksheet.write(row, col, feature_value)
        
        workbook.close()
    
    def process_new_characters(self, characters_dir):
        """
        Process newly extracted characters and add to existing dataset
        """
        characters_path = Path(characters_dir)
        if not characters_path.exists():
            print(f"Directory {characters_dir} does not exist")
            return []
        
        features_list = []
        image_files = list(characters_path.glob("*.png"))
        
        print(f"Processing {len(image_files)} new character images...")
        
        for img_path in image_files:
            try:
                img = cv2.imread(str(img_path), 0)
                if img is not None:
                    features = self.extract_all_features(img)
                    features_list.append({
                        'filename': img_path.name,
                        'features': features
                    })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        return features_list

def main():
    """
    Main function to run feature extraction
    """
    extractor = FeatureExtractor()
    
    # Path to the existing dataset
    dataset_path = Path("../extract/num")
    output_excel = "character_features.xlsx"
    
    if dataset_path.exists():
        # Process the labeled dataset
        features, labels = extractor.process_dataset(dataset_path, output_excel)
        
        print(f"\nDataset processing complete:")
        print(f"- Total samples: {len(features)}")
        print(f"- Features per sample: {len(extractor.feature_names)}")
        print(f"- Feature names: {extractor.feature_names}")
        
        # Also process any newly extracted characters
        new_chars_dir = "extracted_characters"
        if Path(new_chars_dir).exists():
            new_features = extractor.process_new_characters(new_chars_dir)
            if new_features:
                print(f"\nProcessed {len(new_features)} newly extracted characters")
                print("These can be manually labeled and added to training data")
    else:
        print(f"Dataset not found at {dataset_path}")
        print("Please ensure the extract/num directory exists with labeled character images")

if __name__ == "__main__":
    main()