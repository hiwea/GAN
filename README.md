# MNIST GAN Generator with Image Sampling and Checkpointing
**Author:** Hiwa Aziz Abbas

This project implements a Generative Adversarial Network (GAN) using PyTorch for generating handwritten digits from the MNIST dataset. It includes real-time sample generation every epoch, training progress logging, and model checkpointing.

## 🧠 Model Summary

The GAN consists of:
- **Generator**: Fully connected + ConvTranspose2d layers to upsample latent vectors to 28x28 images.
- **Discriminator**: CNN-based binary classifier to distinguish between real and fake images.

## 📁 Project Structure


