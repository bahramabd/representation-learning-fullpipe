

This repository contains Python implementations for predicting disease stages using deep learning-based representation learning and classification methods.

The project focuses on transforming high-dimensional brain-related feature matrices into compact latent representations and using these representations for multi-class disease stage classification.



This repository includes different approaches:


1. A multi-layer perceptron is trained directly on high-dimensional input features.  
This serves as a baseline model for evaluating the effect of representation learning.

2. A Variational Autoencoder is used to learn compact latent representations from the input features.  
After training the VAE, the latent vectors are extracted and passed to a separate MLP classifier.

3. An end-to-end architecture where the encoder, decoder, and classifier are trained together. 
The model jointly optimizes reconstruction quality, latent regularization, and classification performance.

## Main Concepts

- Multi-class classification
- Variational Autoencoders
- Representation learning
- Latent feature extraction
- MLP classification
- Subject-wise evaluation
- Deep learning model comparison

## Technologies

- Python
- PyTorch
- NumPy
- scikit-learn
- Matplotlib
- SciPy

