import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved : {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print(f"Max Reserved : {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

print("done")