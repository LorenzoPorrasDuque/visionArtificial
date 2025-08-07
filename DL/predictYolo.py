from ultralytics import YOLO
import cv2

# Cargar el modelo YOLO entrenado
model = YOLO(r"C:\Users\Anime\Desktop\visionArtificial\DL\best.pt")  # Asegúrate de que el modelo esté en el mismo directorio o da la ruta completa

# Ruta de la imagen que deseas predecir
image_path = r"C:\Users\Anime\Desktop\visionArtificial\DL\test\images\00-20-16_002_1_scale_2.5_aug_20250806_212324.jpg"  # Cambia esto por tu imagen

# Realizar la predicción
results = model(image_path)

# Mostrar resultados
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])  # clase predicha
        conf = box.conf[0]     # confianza
        x1, y1, x2, y2 = box.xyxy[0]  # coordenadas de la caja

        print(f"Clase: {cls}, Confianza: {conf:.2f}, BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

# Mostrar la imagen con las predicciones
result_image = results[0].plot()  # dibuja las cajas en la imagen original
cv2.imshow("Predicción YOLO - Nutrition", cv2.resize(result_image, (200, 800)))
cv2.waitKey(0)
cv2.destroyAllWindows()