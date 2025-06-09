import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

# Create directory to save generated images
os.makedirs("samples", exist_ok=True)

# Hyperparameters
latent_dim = 100
batch_size = 16
lr = 0.0002
epochs = 50  # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataloader = DataLoader(
    datasets.MNIST(root='./data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)


# Generator model
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(3 * 3 * 256, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Initialize models
G = Generator().to(device)
D = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


# Save generated images
def sample_image(epoch):
    z = torch.randn(16, latent_dim).to(device)
    fake = G(z).detach().cpu()
    grid = make_grid(fake, nrow=4, normalize=True)
    plt.figure(figsize=(4, 4))
    plt.axis('off')
    plt.title(f"Epoch {epoch}")
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    img_path = f"samples/epoch_{epoch}.png"
    plt.savefig(img_path)
    plt.close()


# Training loop
for epoch in range(1, epochs + 1):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.size(0)

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Train Discriminator
        D_loss = criterion(D(real_imgs), real_labels) + criterion(D(fake_imgs.detach()), fake_labels)
        D.zero_grad()
        D_loss.backward()
        opt_D.step()

        # Train Generator
        G_loss = criterion(D(fake_imgs), real_labels)
        G.zero_grad()
        G_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch}/{epochs} - D Loss: {D_loss.item():.4f} | G Loss: {G_loss.item():.4f}")

    # Save sample image every epoch
    sample_image(epoch)

# Save final model checkpoints (optional)
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")

