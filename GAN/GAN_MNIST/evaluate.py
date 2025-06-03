import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import os

# Load your Generator model definition
class GeneratorNet(torch.nn.Module):
  def __init__(self):
    super(GeneratorNet,self).__init__()
    n_features = 100
    n_out = 784
    self.hidden0 = nn.Sequential(nn.Linear(n_features, 256),nn.LeakyReLU(0.2))
    self.hidden1 = nn.Sequential(nn.Linear(256,512),nn.LeakyReLU(0.2))
    self.hidden2 = nn.Sequential(nn.Linear(512,1024),nn.LeakyReLU(0.2))
    self.out = nn.Sequential(nn.Linear(1024,n_out),nn.Tanh())
  def forward(self,x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    return x
Generator = GeneratorNet()

def evaluate(generator_path, device='cuda', num_images=64, nz=100):
    # Load and prepare generator
    netG = GeneratorNet().to(device)
    netG.load_state_dict(torch.load(generator_path, map_location=device))
    netG.eval()

    # Generate fresh random noise
    noise = torch.randn(num_images, nz, device=device)

    with torch.no_grad():
        fake_images = netG(noise).cpu()

    # Reshape from (N, 784) → (N, 1, 28, 28)
    fake_images = fake_images.view(-1, 1, 28, 28)

    # Make grid
    grid = vutils.make_grid(fake_images, nrow=int(np.sqrt(num_images)), normalize=True)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(grid, (1, 2, 0)), cmap="gray")
    plt.show()

    # Save to file
    os.makedirs("evaluation_results", exist_ok=True)
    vutils.save_image(fake_images, "evaluation_results/generated_samples.png", nrow=int(np.sqrt(num_images)), normalize=True)

for i in range(5):
  evaluate(generator_path="./generator.pth", device="cuda", num_images=64)
