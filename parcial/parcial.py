import cv2
import numpy as np

# --- CLASIFICACIÓN DE PIEZAS ---
# 1. bien fabricada: color correcto, agujero presente
# 2. sin perforar: color correcto, sin agujero
# 3. mal fabricada: color incorrecto, forma incorrecta, defecto

# Colores de referencia (BGR)
COLORES_REF = {
    'roja': (40, 40, 200),
    'naranja': (0, 140, 255),
    'verde': (0, 200, 0),
    'amarilla': (0, 220, 220),
    'morada': (180, 40, 180)
}

# Tolerancia para color
TOL_COLOR = 60

# Detecta el color dominante de la pieza
def detectar_color_pieza(img, mask):
    mean_color = cv2.mean(img, mask=mask)[:3]
    for nombre, ref in COLORES_REF.items():
        if np.linalg.norm(np.array(mean_color) - np.array(ref)) < TOL_COLOR:
            return nombre
    return 'desconocido'

# Detecta si hay agujero (perforación) en el centro
def detectar_perforacion(img, main_contour):
    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
    centro_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(centro_mask, (cx, cy), 20, 255, -1)
    centro_color = cv2.mean(img, mask=centro_mask)[:3]
    # Si el centro es negro, hay perforación
    return np.all(np.array(centro_color) < 40), (cx, cy)

# Clasifica una pieza
def clasificar_pieza(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 'mal fabricada', 'desconocido', None, None
    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    color = detectar_color_pieza(img, mask)
    perforada, centro = detectar_perforacion(img, main_contour)
    if color == 'desconocido':
        return 'mal fabricada', color, main_contour, centro
    if perforada:
        return 'bien fabricada', color, main_contour, centro
    else:
        return 'sin perforar', color, main_contour, centro

# Procesa el video y cuenta cada tipo de pieza
def inspeccionar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    bien_fabricadas = 0
    sin_perforar = 0
    mal_fabricadas = 0
    total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tipo, color, main_contour, centro = clasificar_pieza(frame)
        total += 1
        if tipo == 'bien fabricada':
            bien_fabricadas += 1
        elif tipo == 'sin perforar':
            sin_perforar += 1
        else:
            mal_fabricadas += 1
        # Visualización clara
        resultado = frame.copy()
        if main_contour is not None:
            cv2.drawContours(resultado, [main_contour], -1, (0,255,0), 3)
        if centro is not None:
            cv2.circle(resultado, centro, 20, (0,0,255), 2)
            cv2.circle(resultado, centro, 3, (255,255,255), -1)
        cv2.putText(resultado, f"Tipo: {tipo}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(resultado, f"Color: {color}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(resultado, f"Total: {total}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(resultado, f"Bien: {bien_fabricadas}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(resultado, f"Sin perforar: {sin_perforar}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(resultado, f"Mal: {mal_fabricadas}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Inspección", resultado)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"Total piezas: {total}")
    print(f"Bien fabricadas: {bien_fabricadas}")
    print(f"Sin perforar: {sin_perforar}")
    print(f"Mal fabricadas: {mal_fabricadas}")

# Ejemplo de uso
if __name__ == "__main__":
    inspeccionar_video("video_2.mp4")
