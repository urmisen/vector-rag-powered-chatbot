import logging
import sys
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log levels."""
    
    # Color codes
    COLORS = {
        'INFO': '\033[94m',    # Blue
        'WARNING': '\033[93m', # Yellow
        'SUCCESS': '\033[92m', # Green
        'ERROR': '\033[91m',   # Red
        'DEBUG': '\033[95m',   # Magenta
        'CRITICAL': '\033[91m', # Red
        'RESET': '\033[0m'     # Reset
    }
    
    def format(self, record):
        # Map log levels to our custom format with colors
        level_mapping = {
            'DEBUG': '[DEBUG]',
            'INFO': '[INFO]',
            'WARNING': '[WARNING]',
            'ERROR': '[ERROR]',
            'CRITICAL': '[ERROR]'
        }
        
        # If we have a custom SUCCESS level, map it
        if hasattr(record, 'success') and record.success:
            levelname = '[SUCCESS]'
            color = self.COLORS['SUCCESS']
        else:
            levelname = level_mapping.get(record.levelname, f'[{record.levelname}]')
            color = self.COLORS.get(record.levelname, '')
        
        # Format the message with color
        log_message = f"{self.formatTime(record)} - {record.name} - {color}{levelname}{self.COLORS['RESET']} - {record.getMessage()}"

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            log_message = f"{log_message}\n{exc_text}"
        elif record.exc_text:
            log_message = f"{log_message}\n{record.exc_text}"

        return log_message

def get_logger(name: str = "MCPClient") -> logging.Logger:
    """Configure and return a logger with custom formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    
    # Clear existing handlers to avoid duplication
    logger.handlers.clear()
    
    # File handler (no colors in file)
    if log_file := os.getenv("LOG_FILE"):
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler (with colors) - Use stderr instead of stdout to avoid interfering with MCP JSON-RPC
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    return logger

# Add SUCCESS level
logging.SUCCESS = 25
logging.addLevelName(logging.SUCCESS, "SUCCESS")

def success(self, message, *args, **kwargs):
    """Log 'message % args' with severity 'SUCCESS'."""
    if self.isEnabledFor(logging.SUCCESS):
        self._log(logging.SUCCESS, message, args, **kwargs)

# Add success method to Logger class
logging.Logger.success = success

logger = get_logger()