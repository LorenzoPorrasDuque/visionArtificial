import threading
import cv2
import time
from logger import Logger
 
 
 
class RunCamera():
    def __init__(self, src=0, name="Camera_1"):
        try:
            self.name = name
            self.src = src
            self.ret = None
            self.frame = None
            self.stopped = False
            self.loggerReport = Logger('LoggerCamera')
            self.loggerReport.logger.info(f"[INFO] Initializing constructor RunCamera ...")
        except Exception as e:
            self.loggerReport.logger.error(f"[ERROR] Error in constructor RunCamera: {e}")
           
 
    def start(self):
        try:
            self.stream = cv2.VideoCapture(self.src)
            time.sleep(1)
            self.ret, self.frame = self.stream.read()
            if self.stream.isOpened():
                self.my_thread = threading.Thread(target=self.get, name=self.name, daemon=True)
                self.my_thread.start()
                self.loggerReport.logger.info(f"[INFO] Camera {self.name} started ...")
            else:
                self.loggerReport.logger.error(f"[ERROR] Camera {self.name} not opened ...")
 
        except Exception as e:
            self.loggerReport.error(f"[ERROR] Error in start camera: {e}")
 
    def get(self):
        while not self.stopped:
            if not self.ret:
                pass
            else:
                try:
                    self.ret, self.frame = self.stream.read()
                except Exception as e:
                    self.loggerReport.logger.error(f"[ERROR] Error in get frame: {e}")