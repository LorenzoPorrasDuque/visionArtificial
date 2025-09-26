from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import torch
from character_separator import CharacterSeparator
from train_classifier import CharacterClassifier
import time

class RealTimeLPR:
    """
    Real-time License Plate Recognition system
    """
    
    def __init__(self, yolo_model_path, classifier_model_path, scaler_path):
        # Initialize YOLO model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        
        # Initialize character separator
        self.char_separator = CharacterSeparator()
        
        # Initialize character classifier
        self.char_classifier = CharacterClassifier()
        self.char_classifier.load_model(classifier_model_path, scaler_path)
        
        # Processing parameters
        self.conf_threshold = 0.5
        self.min_chars = 3  # Minimum characters to consider a valid plate
        
        # Statistics
        self.frame_count = 0
        self.processing_times = []
    
    def process_frame(self, frame):
        """
        Process a single frame for license plate detection and recognition
        """
        start_time = time.time()
        
        # YOLO detection
        results = self.yolo_model.predict(
            source=frame,
            conf=0.25,
            verbose=False,
            device=self.device,
            half=True if self.device == 'cuda' else False,
            imgsz=640,
            max_det=10
        )
        
        recognized_plates = []
        
        # Process each detection
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                if confidence > self.conf_threshold:
                    # Extract plate ROI
                    plate_roi = frame[y1:y2, x1:x2]
                    
                    if plate_roi.size > 0:
                        # Segment characters
                        characters, _, _ = self.char_separator.segment_characters(plate_roi)
                        
                        if len(characters) >= self.min_chars:
                            # Recognize characters
                            plate_text = ""
                            char_confidences = []
                            
                            for char_img in characters:
                                if self.char_classifier.is_trained:
                                    prediction, char_conf = self.char_classifier.predict_character(char_img)
                                    plate_text += str(prediction)
                                    char_confidences.append(char_conf)
                                else:
                                    plate_text += "?"
                            
                            # Calculate average confidence
                            avg_confidence = np.mean(char_confidences) if char_confidences else 0
                            
                            recognized_plates.append({
                                'bbox': (x1, y1, x2, y2),
                                'text': plate_text,
                                'confidence': confidence,
                                'char_confidence': avg_confidence,
                                'char_count': len(characters)
                            })
        
        # Record processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Keep only last 100 times for averaging
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        return recognized_plates
    
    def draw_results(self, frame, recognized_plates):
        """
        Draw detection and recognition results on frame
        """
        for plate in recognized_plates:
            x1, y1, x2, y2 = plate['bbox']
            text = plate['text']
            yolo_conf = plate['confidence']
            char_conf = plate['char_confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare text
            main_text = f"{text}"
            conf_text = f"YOLO: {yolo_conf:.2f} | OCR: {char_conf:.2f}"
            
            # Draw background for text
            (text_width, text_height), _ = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            (conf_width, conf_height), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            
            max_width = max(text_width, conf_width)
            total_height = text_height + conf_height + 10
            
            # Draw background rectangle
            cv2.rectangle(frame, (x1, y1 - total_height - 10), (x1 + max_width + 10, y1), (0, 0, 0), -1)
            
            # Draw texts
            cv2.putText(frame, main_text, (x1 + 5, y1 - conf_height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run_video(self, video_path=None, save_output=False, output_path="real_time_lpr_output.mp4"):
        """
        Run real-time LPR on video file or webcam
        """
        # Open video source
        if video_path:
            cap = cv2.VideoCapture(str(video_path))
            print(f"Processing video: {video_path}")
        else:
            cap = cv2.VideoCapture(0)  # Webcam
            print("Processing webcam feed")
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) if video_path else 30
        
        # Video writer for saving output
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Process frame
            recognized_plates = self.process_frame(frame)
            
            # Draw results
            self.draw_results(frame, recognized_plates)
            
            # Add performance info
            if self.processing_times:
                avg_time = np.mean(self.processing_times)
                fps_real = 1.0 / avg_time if avg_time > 0 else 0
                perf_text = f"FPS: {fps_real:.1f} | Frame: {self.frame_count} | Plates: {len(recognized_plates)}"
                cv2.putText(frame, perf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Real-time License Plate Recognition", frame)
            
            # Save frame if requested
            if save_output and out:
                out.write(frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"lpr_frame_{self.frame_count:04d}.jpg"
                cv2.imwrite(save_path, frame)
                print(f"Frame saved as {save_path}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            print(f"\nProcessing Statistics:")
            print(f"Average processing time: {avg_time:.3f}s")
            print(f"Average FPS: {1.0/avg_time:.1f}")
            print(f"Total frames processed: {self.frame_count}")
    
    def run_realtime_camera(self):
        """
        Run real-time LPR on camera feed
        """
        print("Starting real-time camera LPR...")
        self.run_video(video_path=None, save_output=False)
    
    def process_video_file(self, video_path, save_output=True):
        """
        Process a video file with LPR
        """
        output_path = Path(video_path).parent / f"lpr_{Path(video_path).name}"
        self.run_video(video_path=video_path, save_output=save_output, output_path=str(output_path))

def main():
    """
    Main function to run real-time LPR
    """
    script_dir = Path(__file__).parent
    
    # Paths to models
    yolo_model_path = script_dir / "bestPlateCar.pt"
    classifier_model_path = script_dir / "character_classifier.joblib"
    scaler_path = script_dir / "feature_scaler.joblib"
    
    # Check if models exist
    models_missing = []
    if not yolo_model_path.exists():
        models_missing.append(str(yolo_model_path))
    if not classifier_model_path.exists():
        models_missing.append(str(classifier_model_path))
    if not scaler_path.exists():
        models_missing.append(str(scaler_path))
    
    if models_missing:
        print("Missing required model files:")
        for model in models_missing:
            print(f"  - {model}")
        print("\nPlease ensure you have:")
        print("1. Trained YOLO model (bestPlateCar.pt)")
        print("2. Trained character classifier (run train_classifier.py)")
        return
    
    # Initialize LPR system
    lpr = RealTimeLPR(
        yolo_model_path=str(yolo_model_path),
        classifier_model_path=str(classifier_model_path),
        scaler_path=str(scaler_path)
    )
    
    # Choose processing mode
    print("\\nReal-time License Plate Recognition System")
    print("1. Process video file")
    print("2. Real-time camera feed")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        video_path = script_dir / "video.mp4"
        if video_path.exists():
            lpr.process_video_file(str(video_path))
        else:
            print(f"Video file not found: {video_path}")
            custom_path = input("Enter video path: ").strip()
            if Path(custom_path).exists():
                lpr.process_video_file(custom_path)
            else:
                print("Video file not found!")
    elif choice == "2":
        lpr.run_realtime_camera()
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()