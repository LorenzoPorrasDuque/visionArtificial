from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import torch

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cargar el modelo YOLOv8 (asegúrate de que el archivo 'best.pt' es el que entrenaste con 26 clases)
model = YOLO(script_dir / 'bestPlateCar.pt')
model.to(device)  # Move model to GPU

# Ruta del video de entrada y de salida
video_path = script_dir / 'video.mp4'
output_video_path = script_dir / 'predicted_video_with_chars.mp4'
characters_output_dir = script_dir / 'extracted_characters'

# Create directory for extracted characters
characters_output_dir.mkdir(exist_ok=True)

# Abrir el video
cap = cv2.VideoCapture(video_path)

# Obtener la resolución del video original
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Processing {total_frames} frames...")

# # Crear un objeto para escribir el video con las predicciones
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el video de salida
out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

def extract_characters_from_plate(plate_roi, frame_count, plate_id):
    """
    Extract individual characters from a license plate ROI
    """
    if plate_roi.size == 0:
        return []
    
    # Convert to grayscale for processing
    gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY) if len(plate_roi.shape) == 3 else plate_roi
    
    # Apply preprocessing to enhance character detection
    # 1. Resize to standard size for better processing
    plate_height = 60
    aspect_ratio = gray_plate.shape[1] / gray_plate.shape[0]
    plate_width = int(plate_height * aspect_ratio)
    gray_plate = cv2.resize(gray_plate, (plate_width, plate_height))
    
    # 2. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_plate, (3, 3), 0)
    
    # 3. Apply binary threshold to separate characters from background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Apply morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # 5. Find contours (potential characters)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    character_rois = []
    
    # Filter and sort contours
    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = cv2.contourArea(contour)
        
        # Filter contours based on size and aspect ratio (typical for characters)
        if (0.2 < aspect_ratio < 1.0 and  # Characters are usually taller than wide
            area > 50 and  # Minimum area
            h > 15 and  # Minimum height
            w > 8):  # Minimum width
            valid_contours.append((contour, x, y, w, h))
    
    # Sort contours left to right (reading order)
    valid_contours.sort(key=lambda x: x[1])  # Sort by x coordinate
    
    # Extract character ROIs
    for i, (contour, x, y, w, h) in enumerate(valid_contours):
        # Add small padding around character
        padding = 2
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(binary.shape[1], x + w + padding)
        y_end = min(binary.shape[0], y + h + padding)
        
        # Extract character ROI
        char_roi = binary[y_start:y_end, x_start:x_end]
        
        if char_roi.size > 0:
            # Resize to standard size for consistency
            char_roi = cv2.resize(char_roi, (20, 40))
            character_rois.append(char_roi)
            
            # Save character image for inspection/training
            char_filename = characters_output_dir / f"frame_{frame_count:04d}_plate_{plate_id}_char_{i:02d}.png"
            cv2.imwrite(str(char_filename), char_roi)
    
    return character_rois

# Procesar cada frame del video
frame_count = 0
plate_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Salir si no quedan más frames

    frame_count += 1
    if frame_count % 100 == 0:  # Print progress every 100 frames
        print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

    # Realizar la predicción en el frame actual
    results = model.predict(
        source=frame, 
        conf=0.25, 
        verbose=False,
        device=device,  # Explicitly use GPU
        half=True if device == 'cuda' else False,  # Use FP16 precision on GPU for speed
        imgsz=640,  # You can try smaller sizes like 320 for even faster inference
        max_det=100  # Limit max detections per image
    )

    # Extraer las cajas delimitadoras (bounding boxes) y dibujar rectángulos
    for result in results:
        for box in result.boxes:  # Para cada caja delimitadora
            # Obtener las coordenadas de la caja delimitadora (formato xyxy)
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convertir las coordenadas a enteros
            class_id = int(box.cls[0])  # ID de la clase
            confidence = box.conf[0]  # Confianza de la predicción
            
            if confidence > 0.5:
                # Dibujar el rectángulo alrededor del objeto detectado
                cv2.rectangle(frame, (x1-5, y1-3), (x2+3, y2+3), (0, 255, 0), 2)  # Color verde y grosor 2
                label = f'Class {class_id} ({confidence:.2f})'

                # Poner la etiqueta con el ID de la clase y la confianza
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Extract license plate ROI
                plate_roi = frame[y1-3:y2+3, x1-5:x2+3]
                
                if plate_roi.size > 0:
                    # Extract characters from the plate
                    characters = extract_characters_from_plate(plate_roi, frame_count, plate_count)
                    
                    if len(characters) > 0:
                        # Draw character count on the frame
                        char_info = f'Chars: {len(characters)}'
                        cv2.putText(frame, char_info, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    plate_count += 1

    # Escribir el frame anotado en el video de salida
    out.write(frame)

# Liberar los recursos
cap.release()
out.release()

print(f"Video procesado y guardado en {output_video_path}")
print(f"Caracteres extraídos guardados en {characters_output_dir}")
print(f"Total de placas procesadas: {plate_count}")