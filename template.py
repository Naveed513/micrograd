from pathlib import Path
import logging

format = '%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(name)s:%(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=format,
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

list_of_files = [
    'micrograd/__init__.py',
    'micrograd/engine.py',
    'micrograd/nn.py',
    'micrograd/utils.py',
    'logdir/logs.log',
    'test/test_engine.py',
    'templates/index.html',
    'templates/result.html',
    'setup.py',
    'requirements.txt'
]

for file in list_of_files:
    file_path = Path(file)
    if not file_path.exists():
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch(exist_ok=True)
        logger.info('Created %s successfully.',file_path)
    else:
        logger.info('%s Already exists.', file_path)
