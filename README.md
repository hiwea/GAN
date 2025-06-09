# MNIST GAN Generator with Image Sampling and Checkpointing
**Author:** Hiwa Aziz Abbas

This project implements a Generative Adversarial Network (GAN) using PyTorch for generating handwritten digits from the MNIST dataset. It includes real-time sample generation every epoch, training progress logging, and model checkpointing.

## 🧠 Model Summary

The GAN consists of:
- **Generator**: Fully connected + ConvTranspose2d layers to upsample latent vectors to 28x28 images.
- **Discriminator**: CNN-based binary classifier to distinguish between real and fake images.

## 📁 Project Structure
- gan_mnist.py # Main GAN training script
- samples/ # Folder to save generated images
- generator.pth # Final trained generator checkpoint
- discriminator.pth # Final trained discriminator checkpoint
- requirements.txt # Python dependencies
- README.md # Project documentation
- data/ # MNIST dataset (auto-downloaded)

## 🚀 Features

- **Dataset**: MNIST digits (downloaded automatically)
- **Training**:
  - Optimizer: Adam
  - Batch size: 16 (adjustable)
  - Epochs: 50 (adjustable)
- **Sample Generation**:
  - Saves 16 generated digit samples as PNG every epoch
  - Output saved in the `samples/` directory
- **Checkpointing**:
  - Saves final `generator.pth` and `discriminator.pth` model weights

## 📦 Requirements

- Python 3.8+
- PyTorch (with CUDA for GPU support)
- torchvision
- matplotlib

Install dependencies:
```bash
pip install -r requirements.txt

