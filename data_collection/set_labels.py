import subprocess
import os

LABELIMG_PATH = os.path.join('data', 'labelImg')

# Check if path exists
if not os.path.exists(LABELIMG_PATH):
    print('Cannot find labelImg.py. Execute python setup_labelImg.py first.')
    exit()

else:
    subprocess.run(['cmd', '/c', f'cd {LABELIMG_PATH} && python labelImg.py'], shell=True)