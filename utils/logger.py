import os
from datetime import datetime
import logging
# Set up logging

def get_logger(name, dataset_name=None):
    """Get a logger with the specified name."""
    log_dir = 'logs'
    if dataset_name is not None:
        log_dir = os.path.join(log_dir, dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    # print(f"Log directory: {log_dir}")
    log_file = os.path.join(log_dir, f'ppo_moe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Configure the root logger only once
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()  # Also output to console
            ]
        )
    
    # Return a logger with the specified name
    return logging.getLogger(name)
