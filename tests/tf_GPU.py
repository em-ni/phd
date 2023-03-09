# Test GPU with tensorflow
# Follow: https://www.tensorflow.org/install/ to install tensorflow

import tensorflow as tf
from tensorflow.python.client import device_lib
import sys

# Print python version
print('python --version: ', sys.version)

# Print tf version
print(tf.__version__)

# Print GPU info
print(device_lib.list_local_devices())

# Check if GPU is enabled
print('GPU is enabled: ', tf.test.is_built_with_cuda())

# Check if GPU is visible
print('GPU is visible: ', tf.config.list_physical_devices('GPU'))







