# Test GPU with pytorch
import torch
print("CUDA version: ", torch.version.cuda)

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

# Print GPU live usage
print(torch.cuda.memory_summary(device=torch.cuda.current_device()))

# Print GPU memory usage
print("Allocated Memmory: ",torch.cuda.memory_allocated(device=torch.cuda.current_device()))
print("Reserved Memmory: ", torch.cuda.memory_reserved(device=torch.cuda.current_device()))


