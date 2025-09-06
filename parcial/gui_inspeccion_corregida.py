import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Colores de referencia mejorados para mejor diferenciación
COLORES_REF = {
    'roja': (30, 30, 180),      # Más azul para distinguir del naranja
    'naranja': (0, 120, 255),   # Más saturado naranja
    'verde': (0, 200, 0),
    'amarilla': (0, 220, 220),
    'morada': (150, 50, 150),
    'azul_verdoso': (42, 84, 89),  # Nuevo color detectado en la muestra
}
TOL_COLOR = 70  # Valor balanceado entre detección y precisión

def mejorar_imagen_para_deteccion(frame):
    """Mejora la imagen aumentando brillo y contraste para mejor detección"""
    # Convertir a float para evitar overflow
    frame_float = frame.astype(np.float32)
    
    # Aumentar brillo (+30) y contraste (x1.5)
    frame_mejorado = frame_float * 1.5 + 30
    
    # Asegurar que los valores estén en rango 0-255
    frame_mejorado = np.clip(frame_mejorado, 0, 255).astype(np.uint8)
    
    return frame_mejorado

def detectar_color_pieza(img, mask):
    """Detecta el color de una pieza con mejor manejo de objetos oscuros"""
    mean_color = cv2.mean(img, mask=mask)[:3]
    mejor_coincidencia = 'desconocido'
    menor_distancia = float('inf')
    
    # Aumentar el brillo para objetos muy oscuros antes de comparar
    if np.mean(mean_color) < 60:  # Si es muy oscuro
        mean_color = tuple(min(255, c * 1.8 + 40) for c in mean_color)  # Aumentar brillo
        print(f"Color oscuro ajustado: {mean_color}")
    
    for nombre, ref in COLORES_REF.items():
        distancia = np.linalg.norm(np.array(mean_color) - np.array(ref))
        if distancia < TOL_COLOR and distancia < menor_distancia:
            menor_distancia = distancia
            mejor_coincidencia = nombre
    
    print(f"Color detectado: {mejor_coincidencia}, distancia: {menor_distancia:.2f}, color promedio: {mean_color}")
    return mejor_coincidencia

def clasificar_pieza(frame):
    """Clasificación con imagen mejorada para mejor detección"""
    # Mejorar la imagen antes del análisis
    frame_mejorado = mejorar_imagen_para_deteccion(frame)
    
    # Conversión a escala de grises de la imagen mejorada
    gray = cv2.cvtColor(frame_mejorado, cv2.COLOR_BGR2GRAY)
    
    # Umbralización con valor más bajo para capturar objetos oscuros
    _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    
    # Operaciones morfológicas más agresivas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.dilate(binary, kernel, iterations=2)
    binary = cv2.erode(binary, kernel, iterations=1)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Contornos encontrados: {len(contours)}")
    
    if not contours:
        return 'mal fabricada', 'desconocido', None, None
    
    # Filtrar por área mínima más estricta
    contours_filtrados = [c for c in contours if cv2.contourArea(c) > 5000]  # Área mínima más alta
    print(f"Contornos filtrados: {len(contours_filtrados)}")
    
    if not contours_filtrados:
        return None, None, None, None  # Retornar None si no hay contornos válidos
    
    main_contour = max(contours_filtrados, key=cv2.contourArea)
    area = cv2.contourArea(main_contour)
    print(f"Área del contorno principal: {area}")
    
    # Crear máscara
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    
    # Obtener centro
    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
    
    # Detectar color en el frame ORIGINAL (no mejorado para mantener colores reales)
    color = detectar_color_pieza(frame, mask)
    
    # Verificar centro negro (perforada) - también en frame original
    centro_mask = np.zeros_like(gray)
    cv2.circle(centro_mask, (cx, cy), 135, 255, -1)  # Aumentado 200%: de 45 a 135 píxeles
    centro_color = cv2.mean(frame, mask=centro_mask)[:3]
    
    # Detección inteligente de piezas perforadas
    es_centro_negro = np.all(np.array(centro_color) < 55)  # Umbral permisivo
    
    # Verificación adicional: si es oscuro, verificar si realmente tiene color
    if es_centro_negro:
        # Verificar si se puede detectar un color conocido en el centro
        color_centro = detectar_color_pieza(frame, centro_mask)
        if color_centro != 'desconocido':
            # Si detectamos un color conocido Y el promedio no es extremadamente bajo, no es perforada
            promedio_centro = np.mean(centro_color)
            if promedio_centro > 25:  # Si no es completamente negro
                print(f"Centro oscuro pero con color detectado: {color_centro}, promedio: {promedio_centro:.1f}, no es perforada")
                es_centro_negro = False
    
    if es_centro_negro:
        print(f"Centro negro detectado: {centro_color}")
        return 'perforada', color, main_contour, (cx, cy)
    
    # Verificar uniformidad centro vs cuerpo en frame original
    cuerpo_mask = cv2.bitwise_and(mask, cv2.bitwise_not(centro_mask))
    if cv2.countNonZero(cuerpo_mask) > 0:
        cuerpo_color = cv2.mean(frame, mask=cuerpo_mask)[:3]
        diferencia = np.linalg.norm(np.array(centro_color) - np.array(cuerpo_color))
        print(f"Centro: {centro_color}, Cuerpo: {cuerpo_color}, Diferencia: {diferencia}")
        
        if diferencia < 45:  # Umbral más tolerante
            return 'bien fabricada', color, main_contour, (cx, cy)
        else:
            return 'mal fabricada', color, main_contour, (cx, cy)
    
    return 'mal fabricada', color, main_contour, (cx, cy)

class InspeccionGUI(tk.Frame):
    def __init__(self, master=None, video_path=None):
        super().__init__(master)
        self.master = master
        self.master.title("Inspección de Bloques Reciclables - Con Mejora de Imagen")
        self.master.geometry("1200x800")
        self.pack(fill=tk.BOTH, expand=True)
        
        if video_path is None:
            video_path = "/home/anime/Desktop/visionArtificial/parcial/video_2.mp4"
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            print(f"ERROR: No se pudo abrir el video {self.video_path}")
            return
        
        # Contadores
        self.bien_fabricadas = 0
        self.perforadas = 0
        self.mal_fabricadas = 0
        self.total = 0
        
        # Variables para detección
        self.figure_present = False
        self.start_frame = 0
        self.frames_buffer = []
        
        self.create_widgets()
        self.update_video()

    def create_widgets(self):
        # Frame principal para el video
        self.frame_video = tk.Frame(self, bg="black")
        self.frame_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Título del video
        tk.Label(self.frame_video, text="VIDEO EN VIVO", font=("Arial", 12, "bold"), 
                bg="black", fg="white").pack(pady=5)
        
        self.label_frame = tk.Label(self.frame_video, bg="black")
        self.label_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame para la detección
        self.frame_deteccion = tk.Frame(self, bg="darkblue")
        self.frame_deteccion.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Título de la detección
        tk.Label(self.frame_deteccion, text="ÚLTIMAS DETECCIONES POR TIPO", font=("Arial", 12, "bold"), 
                bg="darkblue", fg="white").pack(pady=5)
        
        # Frame para las 3 pantallas de tipos (en columna)
        self.frame_tipos = tk.Frame(self.frame_deteccion, bg="darkblue")
        self.frame_tipos.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Pantalla 1: Bien fabricada (arriba)
        self.frame_bien = tk.Frame(self.frame_tipos, bg="green", relief=tk.RAISED, bd=2)
        self.frame_bien.pack(fill=tk.BOTH, expand=True, pady=2)
        
        tk.Label(self.frame_bien, text="BIEN FABRICADA", font=("Arial", 10, "bold"), 
                bg="green", fg="white").pack(pady=2)
        
        self.label_bien_frame = tk.Label(self.frame_bien, bg="green", text="Sin detección", 
                                        fg="white", font=("Arial", 8))
        self.label_bien_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Pantalla 2: Perforada (medio)
        self.frame_perforada = tk.Frame(self.frame_tipos, bg="blue", relief=tk.RAISED, bd=2)
        self.frame_perforada.pack(fill=tk.BOTH, expand=True, pady=2)
        
        tk.Label(self.frame_perforada, text="PERFORADA", font=("Arial", 10, "bold"), 
                bg="blue", fg="white").pack(pady=2)
        
        self.label_perforada_frame = tk.Label(self.frame_perforada, bg="blue", text="Sin detección", 
                                             fg="white", font=("Arial", 8))
        self.label_perforada_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Pantalla 3: Mal fabricada (abajo)
        self.frame_mal = tk.Frame(self.frame_tipos, bg="red", relief=tk.RAISED, bd=2)
        self.frame_mal.pack(fill=tk.BOTH, expand=True, pady=2)
        
        tk.Label(self.frame_mal, text="MAL FABRICADA", font=("Arial", 10, "bold"), 
                bg="red", fg="white").pack(pady=2)
        
        self.label_mal_frame = tk.Label(self.frame_mal, bg="red", text="Sin detección", 
                                       fg="white", font=("Arial", 8))
        self.label_mal_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Pantalla 1: Contadores (antes era la segunda)
        self.frame_contadores = tk.Frame(self, bg="lightgreen", width=200)
        self.frame_contadores.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=10)
        self.frame_contadores.pack_propagate(False)
        
        tk.Label(self.frame_contadores, text="CONTADORES", font=("Arial", 14, "bold"), bg="lightgreen").pack(pady=10)
        
        self.label_bien = tk.Label(self.frame_contadores, text="Bien fabricadas:\n0", 
                                  font=("Arial", 12), bg="lightgreen", fg="darkgreen")
        self.label_bien.pack(pady=5)
        
        self.label_perforadas = tk.Label(self.frame_contadores, text="Perforadas:\n0", 
                                        font=("Arial", 12), bg="lightgreen", fg="blue")
        self.label_perforadas.pack(pady=5)
        
        self.label_mal = tk.Label(self.frame_contadores, text="Mal fabricadas:\n0", 
                                 font=("Arial", 12), bg="lightgreen", fg="red")
        self.label_mal.pack(pady=5)
        
        # Pantalla 2: Totalizador (antes era la tercera)
        self.frame_total = tk.Frame(self, bg="lightyellow", width=200)
        self.frame_total.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=10)
        self.frame_total.pack_propagate(False)
        
        tk.Label(self.frame_total, text="TOTALIZADOR", font=("Arial", 14, "bold"), bg="lightyellow").pack(pady=10)
        
        self.label_total = tk.Label(self.frame_total, text="Total procesadas:\n0", 
                                   font=("Arial", 16, "bold"), bg="lightyellow")
        self.label_total.pack(pady=20)
        
        self.label_porcentajes = tk.Label(self.frame_total, text="", font=("Arial", 10), bg="lightyellow")
        self.label_porcentajes.pack(pady=10)

    def update_video(self):
        if not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.after(1, self.update_video)  # Ya está en 1
            return
        
        # Usar imagen mejorada para detección
        frame_mejorado = mejorar_imagen_para_deteccion(frame)
        gray = cv2.cvtColor(frame_mejorado, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filtrar contornos por área mínima para evitar ruido
        contours_validos = [c for c in contours if cv2.contourArea(c) > 5000]
        has_figure = len(contours_validos) > 0
        
        self.frames_buffer.append(frame.copy())  # Guardar frame original
        
        if has_figure and not self.figure_present:
            self.figure_present = True
            self.start_frame = len(self.frames_buffer)
            print(f"Figura detectada, empezando en frame {self.start_frame}")
            
        elif not has_figure and self.figure_present:
            self.figure_present = False
            end_frame = len(self.frames_buffer)
            mid_frame_idx = (self.start_frame + end_frame) // 2
            
            print(f"Figura terminada, analizando frame medio {mid_frame_idx}")
            
            if mid_frame_idx < len(self.frames_buffer):
                test_frame = self.frames_buffer[mid_frame_idx]
                resultado = clasificar_pieza(test_frame)
                
                # Verificar que la clasificación fue exitosa
                if resultado[0] is not None:
                    tipo, color, main_contour, centro = resultado
                    
                    # Actualizar contadores
                    self.total += 1
                    if tipo == 'bien fabricada':
                        self.bien_fabricadas += 1
                    elif tipo == 'perforada':
                        self.perforadas += 1
                    else:
                        self.mal_fabricadas += 1
                    
                    # Crear imagen de resultado con las detecciones
                    frame_resultado = test_frame.copy()
                    if main_contour is not None:
                        cv2.drawContours(frame_resultado, [main_contour], -1, (0,255,0), 3)
                    if centro is not None:
                        cv2.circle(frame_resultado, centro, 137, (0,0,255), 2)  # Círculo de visualización actualizado a 137
                        cv2.circle(frame_resultado, centro, 3, (255,255,255), -1)
                    
                    # Mostrar información en el frame
                    cv2.putText(frame_resultado, f"Tipo: {tipo}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    cv2.putText(frame_resultado, f"Color: {color}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    
                    print(f"Detectada pieza {self.total}: {tipo} - Color: {color}")
                    
                    # Actualizar GUI con el frame de detección
                    self.actualizar_pantallas(tipo, color, frame_resultado)
                else:
                    print("No se pudo clasificar la pieza, frame sin objeto válido")
            
            # Limpiar buffer
            self.frames_buffer = []
        else:
            # Mostrar frame actual solo si no estamos mostrando una detección
            self.mostrar_frame(frame)
        
        # Continuar actualizando más rápido
        self.after(1, self.update_video)  # Ya está en 1

    def mostrar_frame(self, frame):
        """Muestra el frame actual en la GUI"""
        resultado_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(resultado_rgb)
        # Redimensionar para que quepa en la GUI
        img = img.resize((400, 300), Image.Resampling.LANCZOS)  # Más pequeño para dar espacio
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_frame.imgtk = imgtk
        self.label_frame.configure(image=imgtk)

    def mostrar_frame_en_pantalla_tipo(self, frame_resultado, tipo):
        """Muestra el frame en la pantalla correspondiente al tipo detectado"""
        resultado_rgb = cv2.cvtColor(frame_resultado, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(resultado_rgb)
        # Redimensionar para las pantallas de tipo en layout vertical (más grandes)
        img = img.resize((300, 200), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        
        if tipo == 'bien fabricada':
            self.label_bien_frame.imgtk = imgtk
            self.label_bien_frame.configure(image=imgtk)
        elif tipo == 'perforada':
            self.label_perforada_frame.imgtk = imgtk
            self.label_perforada_frame.configure(image=imgtk)
        else:  # mal fabricada
            self.label_mal_frame.imgtk = imgtk
            self.label_mal_frame.configure(image=imgtk)

    def actualizar_pantallas(self, tipo, color, frame_resultado):
        """Actualiza las pantallas con la información detectada"""
        
        # Mostrar el frame en la pantalla correspondiente al tipo
        self.mostrar_frame_en_pantalla_tipo(frame_resultado, tipo)
        
        # Pantalla 1: Contadores actualizados
        self.label_bien.configure(text=f"Bien fabricadas:\n{self.bien_fabricadas}")
        self.label_perforadas.configure(text=f"Perforadas:\n{self.perforadas}")
        self.label_mal.configure(text=f"Mal fabricadas:\n{self.mal_fabricadas}")
        
        # Pantalla 2: Totalizador
        self.label_total.configure(text=f"Total procesadas:\n{self.total}")
        
        # Calcular porcentajes
        if self.total > 0:
            porc_bien = (self.bien_fabricadas / self.total) * 100
            porc_perf = (self.perforadas / self.total) * 100
            porc_mal = (self.mal_fabricadas / self.total) * 100
            
            porcentajes_text = f"Bien: {porc_bien:.1f}%\nPerforadas: {porc_perf:.1f}%\nMal fab.: {porc_mal:.1f}%"
            self.label_porcentajes.configure(text=porcentajes_text)

if __name__ == "__main__":
    root = tk.Tk()
    # Usar ruta absoluta para asegurar que encuentra el video
    video_path = "/home/anime/Desktop/visionArtificial/parcial/video_2.mp4"
    app = InspeccionGUI(master=root, video_path=video_path)
    app.mainloop()
