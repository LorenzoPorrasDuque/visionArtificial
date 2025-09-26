import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
import threading
import queue
import time
from pathlib import Path
import numpy as np
from real_time_lpr import RealTimeLPR

class LPRApplication:
    """
    GUI Application for License Plate Recognition
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition System")
        self.root.geometry("1000x700")
        
        # Initialize variables
        self.lpr_system = None
        self.camera_active = False
        self.video_active = False
        self.current_frame = None
        self.frame_queue = queue.Queue()
        
        # Create GUI
        self.create_widgets()
        
        # Initialize LPR system
        self.initialize_lpr()
    
    def create_widgets(self):
        """
        Create GUI widgets
        """
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), padx=(0, 10))
        
        # Video display
        display_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding="10")
        display_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Results panel
        results_frame = ttk.LabelFrame(main_frame, text="Recognition Results", padding="10")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(10, 0))
        
        # Create control widgets
        self.create_control_widgets(control_frame)
        
        # Create display widgets
        self.create_display_widgets(display_frame)
        
        # Create results widgets
        self.create_results_widgets(results_frame)
    
    def create_control_widgets(self, parent):
        """
        Create control panel widgets
        """
        # Camera controls
        camera_frame = ttk.LabelFrame(parent, text="Camera", padding="5")
        camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_camera_btn = ttk.Button(camera_frame, text="Start Camera", 
                                         command=self.start_camera)
        self.start_camera_btn.grid(row=0, column=0, padx=5)
        
        self.stop_camera_btn = ttk.Button(camera_frame, text="Stop Camera", 
                                        command=self.stop_camera, state=tk.DISABLED)
        self.stop_camera_btn.grid(row=0, column=1, padx=5)
        
        # Video file controls
        video_frame = ttk.LabelFrame(parent, text="Video File", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.select_video_btn = ttk.Button(video_frame, text="Select Video", 
                                         command=self.select_video_file)
        self.select_video_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.process_video_btn = ttk.Button(video_frame, text="Process Video", 
                                          command=self.process_video, state=tk.DISABLED)
        self.process_video_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.selected_video_label = ttk.Label(video_frame, text="No video selected", 
                                            wraplength=200)
        self.selected_video_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Settings
        settings_frame = ttk.LabelFrame(parent, text="Settings", padding="5")
        settings_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(settings_frame, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=0.9, 
                                   variable=self.confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.confidence_label = ttk.Label(settings_frame, text="0.5")
        self.confidence_label.grid(row=0, column=2)
        
        # Bind scale to update label
        confidence_scale.configure(command=self.update_confidence_label)
        
        # Status
        status_frame = ttk.LabelFrame(parent, text="Status", padding="5")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="System ready", 
                                    foreground="green")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Performance info
        self.fps_label = ttk.Label(status_frame, text="FPS: 0")
        self.fps_label.grid(row=1, column=0, sticky=tk.W)
    
    def create_display_widgets(self, parent):
        """
        Create video display widgets
        """
        # Video canvas
        self.video_canvas = tk.Canvas(parent, width=640, height=480, bg='black')
        self.video_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weight
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
    
    def create_results_widgets(self, parent):
        """
        Create results panel widgets
        """
        # Results text area
        self.results_text = scrolledtext.ScrolledText(parent, width=40, height=20)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons for results
        results_controls = ttk.Frame(parent)
        results_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        clear_btn = ttk.Button(results_controls, text="Clear Results", 
                             command=self.clear_results)
        clear_btn.grid(row=0, column=0, padx=5)
        
        save_btn = ttk.Button(results_controls, text="Save Results", 
                            command=self.save_results)
        save_btn.grid(row=0, column=1, padx=5)
        
        # Configure grid weight
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
    
    def initialize_lpr(self):
        """
        Initialize the LPR system
        """
        try:
            script_dir = Path(__file__).parent
            yolo_model_path = script_dir / "bestPlateCar.pt"
            classifier_model_path = script_dir / "character_classifier.joblib"
            scaler_path = script_dir / "feature_scaler.joblib"
            
            # Check if all models exist
            if not all([yolo_model_path.exists(), classifier_model_path.exists(), scaler_path.exists()]):
                self.status_label.config(text="Models not found - please train models first", 
                                       foreground="red")
                return
            
            self.lpr_system = RealTimeLPR(
                yolo_model_path=str(yolo_model_path),
                classifier_model_path=str(classifier_model_path),
                scaler_path=str(scaler_path)
            )
            
            self.status_label.config(text="LPR system initialized", foreground="green")
            
        except Exception as e:
            self.status_label.config(text=f"Error initializing LPR: {str(e)}", 
                                   foreground="red")
            messagebox.showerror("Error", f"Failed to initialize LPR system: {str(e)}")
    
    def update_confidence_label(self, value):
        """
        Update confidence label
        """
        self.confidence_label.config(text=f"{float(value):.2f}")
        if self.lpr_system:
            self.lpr_system.conf_threshold = float(value)
    
    def start_camera(self):
        """
        Start camera feed
        """
        if not self.lpr_system:
            messagebox.showerror("Error", "LPR system not initialized")
            return
        
        self.camera_active = True
        self.start_camera_btn.config(state=tk.DISABLED)
        self.stop_camera_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Camera active", foreground="blue")
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        # Start display update
        self.update_display()
    
    def stop_camera(self):
        """
        Stop camera feed
        """
        self.camera_active = False
        self.start_camera_btn.config(state=tk.NORMAL)
        self.stop_camera_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Camera stopped", foreground="orange")
        
        # Clear canvas
        self.video_canvas.delete("all")
    
    def camera_loop(self):
        """
        Camera processing loop (runs in separate thread)
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            self.stop_camera()
            return
        
        frame_count = 0
        start_time = time.time()
        
        while self.camera_active:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with LPR
            recognized_plates = self.lpr_system.process_frame(frame)
            
            # Draw results
            self.lpr_system.draw_results(frame, recognized_plates)
            
            # Add recognition results to text area
            if recognized_plates:
                timestamp = time.strftime("%H:%M:%S")
                for plate in recognized_plates:
                    result_text = f"[{timestamp}] Detected: {plate['text']} (Conf: {plate['confidence']:.2f})\\n"
                    self.root.after(0, self.add_result_text, result_text)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                self.root.after(0, self.update_fps, fps)
                start_time = time.time()
            
            # Put frame in queue for display
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass
        
        cap.release()
    
    def update_display(self):
        """
        Update video display (runs in main thread)
        """
        try:
            # Get latest frame from queue
            frame = self.frame_queue.get_nowait()
            
            # Convert frame for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                h, w = frame_rgb.shape[:2]
                aspect = w / h
                
                if canvas_width / canvas_height > aspect:
                    new_height = canvas_height
                    new_width = int(new_height * aspect)
                else:
                    new_width = canvas_width
                    new_height = int(new_width / aspect)
                
                frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                # Update canvas
                self.video_canvas.delete("all")
                self.video_canvas.create_image(canvas_width//2, canvas_height//2, 
                                             image=photo, anchor=tk.CENTER)
                self.video_canvas.image = photo  # Keep a reference
                
        except queue.Empty:
            pass
        
        # Schedule next update
        if self.camera_active:
            self.root.after(30, self.update_display)
    
    def select_video_file(self):
        """
        Select video file for processing
        """
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if filename:
            self.selected_video = filename
            self.selected_video_label.config(text=f"Selected: {Path(filename).name}")
            self.process_video_btn.config(state=tk.NORMAL)
    
    def process_video(self):
        """
        Process selected video file
        """
        if not hasattr(self, 'selected_video'):
            messagebox.showerror("Error", "No video selected")
            return
        
        if not self.lpr_system:
            messagebox.showerror("Error", "LPR system not initialized")
            return
        
        # Process video in separate thread
        self.video_active = True
        self.status_label.config(text="Processing video...", foreground="blue")
        
        def process_thread():
            try:
                self.lpr_system.process_video_file(self.selected_video, save_output=True)
                self.root.after(0, self.video_processing_complete)
            except Exception as e:
                self.root.after(0, lambda: self.video_processing_error(str(e)))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def video_processing_complete(self):
        """
        Called when video processing is complete
        """
        self.video_active = False
        self.status_label.config(text="Video processing complete", foreground="green")
        messagebox.showinfo("Complete", "Video processing finished!")
    
    def video_processing_error(self, error_msg):
        """
        Called when video processing encounters an error
        """
        self.video_active = False
        self.status_label.config(text="Video processing failed", foreground="red")
        messagebox.showerror("Error", f"Video processing failed: {error_msg}")
    
    def add_result_text(self, text):
        """
        Add text to results area
        """
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)
    
    def update_fps(self, fps):
        """
        Update FPS display
        """
        self.fps_label.config(text=f"FPS: {fps:.1f}")
    
    def clear_results(self):
        """
        Clear results text area
        """
        self.results_text.delete(1.0, tk.END)
    
    def save_results(self):
        """
        Save results to file
        """
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results: {str(e)}")

def main():
    """
    Main function to run the GUI application
    """
    # Install required package if not already installed
    try:
        import PIL
    except ImportError:
        print("Installing required package: Pillow")
        import subprocess
        subprocess.check_call(["pip", "install", "Pillow"])
        import PIL
    
    # Create and run application
    root = tk.Tk()
    app = LPRApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()