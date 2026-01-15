import os 
import subprocess

# Set path to labeled images
LABELIMG_PATH = os.path.join('data', 'labelImg')

# Clone labelImg repository
if not os.path.exists(LABELIMG_PATH):
    subprocess.run(['mkdir', LABELIMG_PATH], shell=True)
    subprocess.run(['git', 'clone', 'https://github.com/tzutalin/labelImg', LABELIMG_PATH], shell=True)

# Install labelImg
if os.name == 'posix':
    subprocess.run(['make', 'qt5py3'], shell=True)
    subprocess.run(['make', 'qt5py3'], shell=True)
    
if os.name =='nt':
    subprocess.run(['cmd', '/c', f'cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc'], shell=True)
                    
