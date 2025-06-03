import torch
from torchvision.utils import save_image
import torch.nn as nn
import os
import time
import math
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 1, 0),
            self._block(features_g * 16, features_g * 8, 4, 2, 1),
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

# Generator parameters
latent_dim = 100
channels_img = 3
features_g = 64

# Initialize and load model
gen = Generator(z_dim=latent_dim, channels_img=channels_img, features_g=features_g).to(device)
gen.load_state_dict(torch.load("gen.pth", map_location=device))
gen.eval()

for i in range(5):
  num_images = 100
  z = torch.randn(num_images, latent_dim, 1, 1).to(device)

  # Generate fake images
  with torch.no_grad():
      fake_images = gen(z)

  # Create directory
  os.makedirs("generated_images", exist_ok=True)

  # Format local time
  timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
  filename = f"generated_images/sample_{timestamp}.png"

  # Save image
  save_image(fake_images, filename, nrow=num_images // int(math.sqrt(num_images)), normalize=True)

  print(f"✅ Images saved to '{filename}'")
  time.sleep(1)

