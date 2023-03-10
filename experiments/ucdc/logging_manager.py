import logging
 
from logging.handlers import RotatingFileHandler

class LoggingManager:
    def __init__(self):

        self.logger = logging.getLogger()

        self.logger.setLevel(logging.DEBUG)
        

        formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')

        # file_handler = RotatingFileHandler('.data/logs/activity.log', 'a', 1000000, 1)

        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)
        

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)
        