import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    """Configure le système de logging."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'faq_api.log'),
        maxBytes=10_000_000,  
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    faq_api_logger = logging.getLogger("faq_api")
    faq_api_logger.setLevel(logging.DEBUG) 
    faq_api_logger.propagate = False 
    faq_api_logger.addHandler(console_handler)
    faq_api_logger.addHandler(file_handler)
