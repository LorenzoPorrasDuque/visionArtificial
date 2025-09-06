import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

# Colores de referencia para identificación de bloques
CO        tk.Label(self.frame_contadores, text="CONTADORES", font=("Arial", 14, "bold"), bg="lightgreen").pack(pady=10)
        
        self.label_bien = tk.Label(self.frame_contadores, text="Bien fabricadas:\n0", 
                                  font=("Arial", 12), bg="lightgreen", fg="darkgreen")
        self.label_bien.pack(pady=5)
        
        self.label_perforadas = tk.Label(self.frame_contadores, text="Perforadas:\n0", 
                                        font=("Arial", 12), bg="lightgreen", fg="blue")
        self.label_perforadas.pack(pady=5)
        
        self.label_mal = tk.Label(self.frame_contadores, text="Mal fabricadas:\n0", 
                                 font=("Arial", 12), bg="lightgreen", fg="red")
        self.label_mal.pack(pady=5)
    'roja': (40, 40, 200),
    'naranja': (0, 140, 255),
    'verde': (0, 200, 0),
    'amarilla': (0, 220, 220),
    'morada': (180, 40, 180)
}
TOL_COLOR = 60

# --- Lógica de inspección basada en video.py ---
def detectar_color_pieza(img, mask):
    """Detecta el color de una pieza basado en colores de referencia"""
    mean_color = cv2.mean(img, mask=mask)[:3]
    for nombre, ref in COLORES_REF.items():
        if np.linalg.norm(np.array(mean_color) - np.array(ref)) < TOL_COLOR:
            return nombre
    return 'desconocido'

def detectar_perforacion(img, main_contour):
    """Detecta si hay perforación en el centro de la pieza (centro negro)"""
    M = cv2.moments(main_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
    
    centro_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(centro_mask, (cx, cy), 20, 255, -1)
    centro_color = cv2.mean(img, mask=centro_mask)[:3]
    # Si el centro es negro (valores bajos), está perforada
    return np.all(np.array(centro_color) < 40), (cx, cy)

def detectar_uniformidad_color(img, mask):
    """Detecta si el color de la pieza es uniforme o tiene diferencias grandes"""
    # Obtener todos los píxeles de la pieza
    pixeles = img[mask > 0]
    if len(pixeles) == 0:
        return False
    
    # Calcular la desviación estándar del color
    std_b = np.std(pixeles[:, 0])
    std_g = np.std(pixeles[:, 1]) 
    std_r = np.std(pixeles[:, 2])
    
    # Si la desviación estándar es alta, el color no es uniforme
    umbral_uniformidad = 30  # Ajustable según necesidad
    es_uniforme = (std_b < umbral_uniformidad and 
                   std_g < umbral_uniformidad and 
                   std_r < umbral_uniformidad)
    
    return es_uniforme

def clasificar_pieza(frame):
    """Clasifica una pieza en tiempo real con la nueva lógica corregida"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 'mal fabricada', 'desconocido', None, None
    
    main_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [main_contour], -1, 255, -1)
    
    color = detectar_color_pieza(frame, mask)
    perforada, centro = detectar_perforacion(frame, main_contour)
    es_uniforme = detectar_uniformidad_color(frame, mask)
    
    # Nueva lógica de clasificación:
    # 1. Si tiene centro negro (perforada) -> "perforada"
    # 2. Si el color es uniforme -> "bien fabricada" 
    # 3. Si el color no es uniforme -> "mal fabricada"
    
    if perforada:
        return 'perforada', color, main_contour, centro
    elif es_uniforme and color != 'desconocido':
        return 'bien fabricada', color, main_contour, centro
    else:
        return 'mal fabricada', color, main_contour, centro

class InspeccionGUI(tk.Frame):
    def __init__(self, master=None, video_path=None):
        super().__init__(master)
        self.master = master
        self.master.title("Inspección de Bloques Reciclables - Tiempo Real")
        self.master.geometry("1200x800")
        self.pack(fill=tk.BOTH, expand=True)
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Contadores siguiendo la nueva lógica
        self.bien_fabricadas = 0
        self.perforadas = 0
        self.mal_fabricadas = 0
        self.total = 0
        
        # Variables para detección de figuras
        self.figure_present = False
        self.start_frame = 0
        self.frames_buffer = []
        
        self.create_widgets()
        self.update_video()

    def create_widgets(self):
        """Crea la interfaz gráfica con 3 pantallas como se solicita"""
        # Frame principal para el video
        self.frame_video = tk.Frame(self, bg="black")
        self.frame_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.label_frame = tk.Label(self.frame_video, bg="black")
        self.label_frame.pack(fill=tk.BOTH, expand=True)
        
        # Pantalla 1: Tipo de pieza analizada
        self.frame_tipo = tk.Frame(self, bg="lightblue", width=200)
        self.frame_tipo.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=10)
        self.frame_tipo.pack_propagate(False)
        
        tk.Label(self.frame_tipo, text="TIPO DE PIEZA", font=("Arial", 14, "bold"), bg="lightblue").pack(pady=10)
        self.label_tipo = tk.Label(self.frame_tipo, text="Sin detectar", font=("Arial", 12), bg="lightblue", wraplength=180)
        self.label_tipo.pack(pady=5)
        
        self.label_color = tk.Label(self.frame_tipo, text="Color: -", font=("Arial", 12), bg="lightblue")
        self.label_color.pack(pady=5)
        
        # Pantalla 2: Contadores
        self.frame_contadores = tk.Frame(self, bg="lightgreen", width=200)
        self.frame_contadores.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=10)
        self.frame_contadores.pack_propagate(False)
        
        tk.Label(self.frame_contadores, text="CONTADORES", font=("Arial", 14, "bold"), bg="lightgreen").pack(pady=10)
        
        self.label_bien = tk.Label(self.frame_contadores, text="Bien fabricadas: 0", 
                                  font=("Arial", 12), bg="lightgreen", fg="darkgreen")
        self.label_bien.pack(pady=5)
        
        self.label_sin = tk.Label(self.frame_contadores, text="Sin perforar: 0", 
                                 font=("Arial", 12), bg="lightgreen", fg="orange")
        self.label_sin.pack(pady=5)
        
        self.label_mal = tk.Label(self.frame_contadores, text="Mal fabricadas: 0", 
                                 font=("Arial", 12), bg="lightgreen", fg="red")
        self.label_mal.pack(pady=5)
        
        # Pantalla 3: Totalizador
        self.frame_total = tk.Frame(self, bg="lightyellow", width=200)
        self.frame_total.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=10)
        self.frame_total.pack_propagate(False)
        
        tk.Label(self.frame_total, text="TOTALIZADOR", font=("Arial", 14, "bold"), bg="lightyellow").pack(pady=10)
        
        self.label_total = tk.Label(self.frame_total, text="Total procesadas: 0", 
                                   font=("Arial", 16, "bold"), bg="lightyellow")
        self.label_total.pack(pady=20)
        
        # Porcentajes
        self.label_porcentajes = tk.Label(self.frame_total, text="", font=("Arial", 10), bg="lightyellow")
        self.label_porcentajes.pack(pady=10)

    def update_video(self):
        """Actualiza el video y procesa la detección siguiendo la lógica de video.py"""
        ret, frame = self.cap.read()
        if not ret:
            # Reiniciar video cuando termine
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        
        # Detección de figuras siguiendo la lógica de video.py
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        has_figure = len(contours) > 0
        
        # Buffer de frames
        self.frames_buffer.append(frame.copy())
        
        # Lógica de detección cuando aparece y desaparece una figura
        if has_figure and not self.figure_present:
            self.figure_present = True
            self.start_frame = len(self.frames_buffer)
            
        elif not has_figure and self.figure_present:
            self.figure_present = False
            end_frame = len(self.frames_buffer)
            mid_frame_idx = (self.start_frame + end_frame) // 2
            
            if mid_frame_idx < len(self.frames_buffer):
                test_frame = self.frames_buffer[mid_frame_idx]
                tipo, color, main_contour, centro = clasificar_pieza(test_frame)
                
                # Actualizar contadores
                self.total += 1
                if tipo == 'bien fabricada':
                    self.bien_fabricadas += 1
                elif tipo == 'sin perforar':
                    self.sin_perforar += 1
                else:
                    self.mal_fabricadas += 1
                
                # Crear imagen de resultado
                resultado = test_frame.copy()
                if main_contour is not None:
                    cv2.drawContours(resultado, [main_contour], -1, (0,255,0), 3)
                if centro is not None:
                    cv2.circle(resultado, centro, 20, (0,0,255), 2)
                    cv2.circle(resultado, centro, 3, (255,255,255), -1)
                
                # Mostrar información en el frame
                cv2.putText(resultado, f"Tipo: {tipo}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(resultado, f"Color: {color}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                # Actualizar GUI
                self.actualizar_pantallas(tipo, color, resultado)
            
            # Limpiar buffer
            self.frames_buffer = []
        
        # Si hay figura presente, mostrar frame actual
        if has_figure:
            self.mostrar_frame(frame)
        
        # Continuar actualizando
        self.after(30, self.update_video)

    def mostrar_frame(self, frame):
        """Muestra el frame actual en la GUI"""
        resultado_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(resultado_rgb)
        # Redimensionar para que quepa en la GUI
        img = img.resize((600, 400), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_frame.imgtk = imgtk
        self.label_frame.configure(image=imgtk)

    def actualizar_pantallas(self, tipo, color, frame_resultado):
        """Actualiza las 3 pantallas con la información detectada"""
        # Mostrar frame de detección
        resultado_rgb = cv2.cvtColor(frame_resultado, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(resultado_rgb)
        img = img.resize((600, 400), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label_frame.imgtk = imgtk
        self.label_frame.configure(image=imgtk)
        
        # Pantalla 1: Tipo de pieza
        self.label_tipo.configure(text=f"Última detectada: {tipo}")
        self.label_color.configure(text=f"Color: {color}")
        
        # Pantalla 2: Contadores
        self.label_bien.configure(text=f"Bien fabricadas: {self.bien_fabricadas}")
        self.label_sin.configure(text=f"Sin perforar: {self.sin_perforar}")
        self.label_mal.configure(text=f"Mal fabricadas: {self.mal_fabricadas}")
        
        # Pantalla 3: Totalizador
        self.label_total.configure(text=f"Total procesadas: {self.total}")
        
        # Calcular porcentajes
        if self.total > 0:
            porc_bien = (self.bien_fabricadas / self.total) * 100
            porc_sin = (self.sin_perforar / self.total) * 100
            porc_mal = (self.mal_fabricadas / self.total) * 100
            
            porcentajes_text = f"Bien: {porc_bien:.1f}%\nSin perf.: {porc_sin:.1f}%\nMal fab.: {porc_mal:.1f}%"
            self.label_porcentajes.configure(text=porcentajes_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = InspeccionGUI(master=root, video_path="video_2.mp4")
    app.mainloop()
