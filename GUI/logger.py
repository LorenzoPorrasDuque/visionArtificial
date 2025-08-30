import logging
from colorlog import ColoredFormatter
 
class Logger:
    def __init__(self, name, log_file="GUI/logs/app.log", level=logging.DEBUG):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.log_file = log_file
        if not self.logger.handlers:
            ## Handler write in console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_formatter = ColoredFormatter(
                "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'green',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_formatter)
 
            ## Handler write in log
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(level)
            file_formatter = ColoredFormatter(
                "%(asctime)s - %(name)s %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
 
            ## add handlers to logger
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
   
    def debug(self, msg):
        self.logger.debug(msg)
 
    def info(self, msg):
        self.logger.info(msg)
 
    def warning(self, msg):
        self.logger.warning(msg)
 
    def error(self, msg):
        self.logger.error(msg)
   
    def critical(self, msg):
        self.logger.critical(msg)