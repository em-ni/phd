# Test GPU with pytorch
import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
