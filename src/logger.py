import logging
import os
from pathlib import Path
from datetime import datetime

root_dir=Path(__file__).resolve().parents[1]
logs_dir=os.path.join(root_dir, 'logs')
logs_filename = f'{datetime.now().strftime("%d_%m_%y_%H_%M_%S")}.log'
os.makedirs(logs_dir, exist_ok=True)
logs_filepath = os.path.join(logs_dir, logs_filename)

logging.basicConfig(
    level=logging.INFO,
    filename=logs_filepath,
    format="[ %(asctime)s ] %(lineno)d - %(levelname)s - %(module)s- %(message)s"
)

if __name__=='__main__':
    logging.info('Program started')