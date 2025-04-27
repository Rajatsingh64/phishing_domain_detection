import logging
import os
from datetime import datetime

# Create log file name with a safe format
log_file_name = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

# Get log directory path
log_dir = os.path.join(os.getcwd(), "logs")

# Ensure the log directory exists
os.makedirs(log_dir, exist_ok=True)

# Define full path for the log file
log_file_path = os.path.join(log_dir, log_file_name)

# Configure logging
logging.basicConfig(
        filename=log_file_path,
        format="[%(asctime)s] Line: %(lineno)d | %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
)
    
#Suppress debug logs from external libraries
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING) 
logging.getLogger("numexpr.utils").setLevel(logging.WARNING) 