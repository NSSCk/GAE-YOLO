import torch
import torchvision
# Test environment configuration
print(f"PyTorch: {torch.__version__}")        
print(f"torchvision: {torchvision.__version__}")
print(f"PyTorch-CUDA: {torch.version.cuda}")        
print(f"GPU-numbers: {torch.cuda.device_count()}")
print(f"GPU-run: {torch.cuda.get_device_name()}")
