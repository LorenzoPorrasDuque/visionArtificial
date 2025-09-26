import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import cv2
from pathlib import Path
from feature_extractor import FeatureExtractor

class CharacterClassifier:
    """
    Character classification model for license plate OCR
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        
    def load_data(self, excel_path):
        """
        Load training data from Excel file
        """
        try:
            df = pd.read_excel(excel_path, header=0, engine='openpyxl')
            y = df.iloc[:, 0].values  # First column is labels
            X = df.iloc[:, 1:].values  # Remaining columns are features
            
            print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train the character classification model
        """
        print("Training character classification model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with optimized parameters
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state,
            alpha=0.001
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Training Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        self.is_trained = True
        return accuracy
    
    def hyperparameter_tuning(self, X, y):
        """
        Perform hyperparameter tuning to find best parameters
        """
        print("Performing hyperparameter tuning...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Define parameter grid
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 50, 25)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
        
        # Perform grid search
        mlp = MLPClassifier(max_iter=1000, random_state=42)
        grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Train final model with best parameters
        self.model = grid_search.best_estimator_
        
        # Evaluate on test set
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test accuracy with best parameters: {accuracy:.4f}")
        
        self.is_trained = True
        return accuracy
    
    def predict_character(self, character_image):
        """
        Predict a single character from image
        """
        if not self.is_trained:
            print("Model not trained yet!")
            return None
        
        # Extract features from image
        features = self.feature_extractor.extract_all_features(character_image)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        confidence = max(self.model.predict_proba(features_scaled)[0])
        
        return prediction, confidence
    
    def predict_multiple_characters(self, character_images):
        """
        Predict multiple characters
        """
        predictions = []
        
        for char_img in character_images:
            result = self.predict_character(char_img)
            if result:
                predictions.append(result)
        
        return predictions
    
    def save_model(self, model_path, scaler_path):
        """
        Save trained model and scaler
        """
        if self.is_trained:
            joblib.dump(self.model, model_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Model saved to {model_path}")
            print(f"Scaler saved to {scaler_path}")
        else:
            print("No trained model to save!")
    
    def load_model(self, model_path, scaler_path):
        """
        Load pre-trained model and scaler
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            print(f"Model loaded from {model_path}")
            print(f"Scaler loaded from {scaler_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def test_with_sample_images(self, test_dir):
        """
        Test the model with sample images
        """
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        test_path = Path(test_dir)
        if not test_path.exists():
            print(f"Test directory {test_dir} not found")
            return
        
        print(f"\nTesting model with images from {test_dir}")
        
        image_files = list(test_path.glob("*.png"))[:10]  # Test first 10 images
        
        for img_path in image_files:
            img = cv2.imread(str(img_path), 0)
            if img is not None:
                prediction, confidence = self.predict_character(img)
                print(f"{img_path.name}: Predicted {prediction} (confidence: {confidence:.3f})")

def main():
    """
    Main training function
    """
    # Initialize classifier
    classifier = CharacterClassifier()
    
    # First, extract features if not already done
    feature_excel = "character_features.xlsx"
    if not Path(feature_excel).exists():
        print("Feature file not found. Running feature extraction first...")
        extractor = FeatureExtractor()
        dataset_path = Path("../extract/num")
        if dataset_path.exists():
            extractor.process_dataset(dataset_path, feature_excel)
        else:
            print("Dataset not found! Please ensure ../extract/num exists with character images")
            return
    
    # Load training data
    X, y = classifier.load_data(feature_excel)
    
    if X is not None and y is not None:
        # Train model (you can choose between regular training or hyperparameter tuning)
        print("Choose training method:")
        print("1. Quick training with good default parameters")
        print("2. Hyperparameter tuning (slower but potentially better results)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            accuracy = classifier.hyperparameter_tuning(X, y)
        else:
            accuracy = classifier.train_model(X, y)
        
        # Save trained model
        model_path = "character_classifier.joblib"
        scaler_path = "feature_scaler.joblib"
        classifier.save_model(model_path, scaler_path)
        
        # Test with sample images if available
        test_dirs = ["extracted_characters", "../extract/num/0"]
        for test_dir in test_dirs:
            if Path(test_dir).exists():
                classifier.test_with_sample_images(test_dir)
                break
        
        print(f"\nTraining completed with accuracy: {accuracy:.4f}")
    else:
        print("Failed to load training data")

if __name__ == "__main__":
    main()