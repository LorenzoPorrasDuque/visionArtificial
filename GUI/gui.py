from logger import Logger
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import camera
import tkinter.font as font
import numpy as np
 
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.logReport = Logger('LoggerApp')
        self.logReport.logger.info(f"[INFO] Initializing constructor Application ...")
        self.master = master
        self.width = 1200
        self.height = 720
        self.frame = None
        self.master.geometry("%dx%d" % (self.width, self.height))
        
        # Video analysis variables
        self.video_cap = None
        self.total_figures = 0
        self.circles = 0
        self.rings = 0
        self.figure_present = False
        self.start_frame = 0
        self.frames_buffer = []
        self.video_running = False
 
        #Create Widgets
        self.createWidgets()
        self.createFrame()
 
       
        self.master.mainloop()
 
    def createFrame(self):
        # First video frame (existing camera)
        self.labelVideo1 = tk.Label(self.master,
                                    borderwidth=2,
                                    relief="solid")
       
        self.labelVideo1.place(x=10, y=40)
        
        # Second video frame (original video)
        self.labelVideo2 = tk.Label(self.master,
                                    borderwidth=2,
                                    relief="solid")
       
        self.labelVideo2.place(x=350, y=40)
        
        # Third video frame (detection results)
        self.labelVideo3 = tk.Label(self.master,
                                    borderwidth=2,
                                    relief="solid")
       
        self.labelVideo3.place(x=690, y=40)
        
        self.createImageZeros()
        self.labelVideo1.configure(image=self.imgTk)
        self.labelVideo1.image = self.imgTk
        
        # Configure second and third video with same image
        self.labelVideo2.configure(image=self.imgTk)
        self.labelVideo2.image = self.imgTk
        
        self.labelVideo3.configure(image=self.imgTk)
        self.labelVideo3.image = self.imgTk
 
    def createImageZeros(self):
        self.frame = np.zeros([480, 320, 3], dtype=np.uint8)
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        imgArray = Image.fromarray(self.frame)
        self.imgTk = ImageTk.PhotoImage(image=imgArray)
       
 
 
    def createWidgets(self):
        self.fontLabelText = font.Font(
            family='Helvetica', size=8, weight='normal'
        )
        
        # Camera 1 label (existing)
        self.labelNameCamera = tk.Label(
            self.master, text="Camera 1", fg='#000000'
        )
        self.labelNameCamera['font'] = self.fontLabelText
        self.labelNameCamera.place(x=10, y=10)

        # Camera 2 label (new)
        self.labelNameCamera2 = tk.Label(
            self.master, text="Video Analysis", fg='#000000'
        )
        self.labelNameCamera2['font'] = self.fontLabelText
        self.labelNameCamera2.place(x=350, y=10)

        # Camera 3 label (detection results)
        self.labelNameCamera3 = tk.Label(
            self.master, text="Detection Results", fg='#000000'
        )
        self.labelNameCamera3['font'] = self.fontLabelText
        self.labelNameCamera3.place(x=690, y=10)

        # Counter labels and values for the second video
        self.labelCount1 = tk.Label(
            self.master, text="Simple:", fg='#000000'
        )
        self.labelCount1['font'] = self.fontLabelText
        self.labelCount1.place(x=690, y=530)

        self.labelCount1Value = tk.Label(
            self.master, text="0", fg='#FF0000', font=('Arial', 16, 'bold')
        )
        self.labelCount1Value.place(x=740, y=525)

        self.labelCount2 = tk.Label(
            self.master, text="Double:", fg='#000000'
        )
        self.labelCount2['font'] = self.fontLabelText
        self.labelCount2.place(x=790, y=530)

        self.labelCount2Value = tk.Label(
            self.master, text="0", fg='#0000FF', font=('Arial', 16, 'bold')
        )
        self.labelCount2Value.place(x=840, y=525)

        self.btnInitCamera = tk.Button(
            self.master,
            text="Init Camera",
            bg="#007A39",
            fg='#ffffff',
            width=12,
            command=self.initCamera
        )

        self.btnInitCamera.place(x=100, y=600)

        #create another button to stop camera
        self.btnStopCamera = tk.Button(
            self.master,
            text="Stop Camera",
            bg="#D62828",
            fg='#ffffff',
            width=12,
            command=self.stopCamera()
        )
        self.btnStopCamera.place(x=250, y=600)
        
        # Button to start video analysis
        self.btnStartVideo = tk.Button(
            self.master,
            text="Start Video",
            bg="#0066CC",
            fg='#ffffff',
            width=12,
            command=self.startVideoAnalysis
        )
        self.btnStartVideo.place(x=350, y=600)
        
        # Button to stop video analysis
        self.btnStopVideo = tk.Button(
            self.master,
            text="Stop Video",
            bg="#CC6600",
            fg='#ffffff',
            width=12,
            command=self.stopVideoAnalysis
        )
        self.btnStopVideo.place(x=500, y=600) 
    def initCamera(self):
        self.camera1 = camera.RunCamera(src=0, name="Camera_1")
        self.camera1.start()
        self.showVideo()
 
    def convertToFrameTk(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgArray = Image.fromarray(frame)
        return ImageTk.PhotoImage(image=imgArray)
       
 
    def showVideo(self):
        try:
            if self.camera1.ret is not None:
                imgtk = self.convertToFrameTk(self.camera1.frame)
                self.labelVideo1.configure(image=imgtk)
                self.labelVideo1.image = imgtk
            self.labelVideo1.after(10, self.showVideo)
                 
               
        except Exception as e:
            self.logReport.logger.error(f"[ERROR] Error in showVideo: {e}")
       
 
    def stopCamera(self):
        pass
    
    def updateCounters(self, group1_count, group2_count):
        """Update the counter labels with new values"""
        self.labelCount1Value.configure(text=str(group1_count))
        self.labelCount2Value.configure(text=str(group2_count))
    
    def updateVideo2(self, frame):
        """Update the second video display with a new frame"""
        try:
            imgtk = self.convertToFrameTk(frame)
            self.labelVideo2.configure(image=imgtk)
            self.labelVideo2.image = imgtk
        except Exception as e:
            self.logReport.logger.error(f"[ERROR] Error in updateVideo2: {e}")
    
    def updateVideo3(self, frame):
        """Update the third video display with detection results"""
        try:
            imgtk = self.convertToFrameTk(frame)
            self.labelVideo3.configure(image=imgtk)
            self.labelVideo3.image = imgtk
        except Exception as e:
            self.logReport.logger.error(f"[ERROR] Error in updateVideo3: {e}")
    
    def startVideoAnalysis(self):
        """Start the video analysis with real-time counting"""
        try:
            self.video_cap = cv2.VideoCapture("contornos/camara/video_1_12.avi")
            if not self.video_cap.isOpened():
                self.logReport.logger.error("[ERROR] Could not open video file")
                return
            
            self.video_running = True
            self.total_figures = 0
            self.circles = 0
            self.rings = 0
            self.figure_present = False
            self.start_frame = 0
            self.frames_buffer = []
            
            self.updateCounters(self.circles, self.rings)
            self.processVideoFrame()
            
        except Exception as e:
            self.logReport.logger.error(f"[ERROR] Error starting video: {e}")
    
    def stopVideoAnalysis(self):
        """Stop the video analysis"""
        self.video_running = False
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        self.logReport.logger.info(f"[INFO] Video stopped. Total: {self.total_figures}, Simple: {self.circles}, Double: {self.rings}")
    
    def processVideoFrame(self):
        """Process each frame of the video with the same logic as video.py"""
        if not self.video_running or not self.video_cap:
            return
        
        try:
            ret, frame = self.video_cap.read()
            if not ret:
                self.stopVideoAnalysis()
                self.logReport.logger.info("[INFO] Video finished")
                return
            
            # Display the current frame
            self.updateVideo2(frame)
            
            # Apply the same logic as video.py
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            has_figure = len(contours) > 0
            
            if has_figure and not self.figure_present:
                self.figure_present = True
                self.start_frame = len(self.frames_buffer)
            
            elif not has_figure and self.figure_present:
                self.figure_present = False
                end_frame = len(self.frames_buffer)
                mid_frame_idx = (self.start_frame + end_frame) // 2
                
                if mid_frame_idx < len(self.frames_buffer):
                    self.classifyFigure(mid_frame_idx, end_frame)
                
                self.frames_buffer = []
            
            self.frames_buffer.append(frame.copy())
            
            # Schedule next frame processing
            self.labelVideo2.after(30, self.processVideoFrame)
            
        except Exception as e:
            self.logReport.logger.error(f"[ERROR] Error processing video frame: {e}")
    
    def classifyFigure(self, mid_frame_idx, end_frame):
        """Classify the detected figure - EXACT SAME LOGIC AS video.py + show detection frame"""
        try:
            classified = False
            frame_offset = 0
            
            while not classified and frame_offset < len(self.frames_buffer):
                test_idx = (
                    mid_frame_idx - frame_offset
                    if frame_offset % 2 == 0
                    else mid_frame_idx + (frame_offset + 1) // 2
                )
                
                if 0 <= test_idx < len(self.frames_buffer):
                    test_frame = self.frames_buffer[test_idx]
                    test_gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
                    _, test_binary = cv2.threshold(test_gray, 50, 255, cv2.THRESH_BINARY)
                    
                    contours, hierarchy = cv2.findContours(
                        test_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                    )
                    
                    if hierarchy is not None:
                        internal_contours = 0
                        for i in range(len(hierarchy[0])):
                            if hierarchy[0][i][3] != -1:
                                internal_contours += 1
                        
                        if internal_contours == 1 or internal_contours == 2:
                            self.total_figures += 1
                            
                            # Create detection image (same as video.py)
                            detection_img = test_binary.copy()
                            
                            if internal_contours == 1:
                                self.circles += 1
                                cv2.putText(
                                    detection_img,
                                    f"Simple #{self.circles}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                )
                                self.logReport.logger.info(f"[INFO] Simple circle detected #{self.circles}")
                                
                            elif internal_contours == 2:
                                self.rings += 1
                                cv2.putText(
                                    detection_img,
                                    f"Doble #{self.rings}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (255, 255, 255),
                                    2,
                                )
                                self.logReport.logger.info(f"[INFO] Double circle detected #{self.rings}")
                            
                            # Convert binary to 3-channel for display
                            detection_img_color = cv2.cvtColor(detection_img, cv2.COLOR_GRAY2BGR)
                            
                            # Show detection in third video box
                            self.updateVideo3(detection_img_color)
                            
                            # Update GUI counters in real-time
                            self.updateCounters(self.circles, self.rings)
                            classified = True
                            
                        elif internal_contours > 2:
                            # Check previous frames for double circles - EXACT SAME AS video.py
                            for back_idx in range(end_frame - 10, end_frame):
                                if back_idx >= 0 and back_idx < len(self.frames_buffer):
                                    back_frame = self.frames_buffer[back_idx]
                                    back_gray = cv2.cvtColor(back_frame, cv2.COLOR_BGR2GRAY)
                                    _, back_binary = cv2.threshold(back_gray, 50, 255, cv2.THRESH_BINARY)
                                    
                                    back_contours, back_hierarchy = cv2.findContours(
                                        back_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
                                    )
                                    
                                    if back_hierarchy is not None:
                                        back_internal = 0
                                        for i in range(len(back_hierarchy[0])):
                                            if back_hierarchy[0][i][3] != -1:
                                                back_internal += 1
                                        
                                        if back_internal == 2:
                                            self.total_figures += 1
                                            self.rings += 1
                                            
                                            # Create detection image
                                            detection_img = back_binary.copy()
                                            cv2.putText(
                                                detection_img,
                                                f"Doble #{self.rings}",
                                                (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                1,
                                                (255, 255, 255),
                                                2,
                                            )
                                            
                                            # Convert and show detection
                                            detection_img_color = cv2.cvtColor(detection_img, cv2.COLOR_GRAY2BGR)
                                            self.updateVideo3(detection_img_color)
                                            
                                            self.logReport.logger.info(f"[INFO] Double circle detected (fallback) #{self.rings}")
                                            self.updateCounters(self.circles, self.rings)
                                            classified = True
                                            break
                
                frame_offset += 1
                
        except Exception as e:
            self.logReport.logger.error(f"[ERROR] Error in classifyFigure: {e}")
       
 
 
def main():
    root = tk.Tk()
    root.title("GUI CAMERA")
    appRunCamera = Application(master=root)