import logging
import logging.config
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s')

list_of_file = [
    'src/__init__.py',
    'src/components/__init__.py',
    'src/pipeline/__init__.py',
    'src/utils/__init__.py',
    'src/utils/common.py',
    'src/entity/__init__.py',
    'src/entity/config_entity.py',
    'src/config/__init__.py',
    'src/config/configuration.py',
    'src/constants/__init__.py',
    'src/logger.py',
    'src/exception.py',
    'requirements.txt',
    'config.yaml',
    'main.py',
    'app.py',
    'setup.py'
]

for file in list_of_file:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)

    if filedir:
        if not os.path.exists(filedir):
            logging.info(f'Creating directory: {filedir}')
            os.makedirs(filedir, exist_ok=True)
        else:
            logging.info(f'Directory already exists:{filedir}')

    if (not os.path.exists(filename)) or (os.path.getsize(filepath)==0):
        logging.info(f'Creating filepath: {filepath}')
        with open(filepath, 'w') as f:
            pass
    else:
        logging.info(f'{file} already exists')