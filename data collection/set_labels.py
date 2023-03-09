import subprocess
import os

LABELIMG_PATH = os.path.join('data', 'labeled')

subprocess.run(['cmd', '/c', f'cd {LABELIMG_PATH} && python labelImg.py'], shell=True)