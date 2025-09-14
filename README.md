# Image Compression Project

This repository demonstrates two powerful approaches to image compression using machine learning and signal processing techniques. The project showcases both **Autoencoders** (deep learning approach) and **Principal Component Analysis (PCA)** (classical approach) for compressing and reconstructing images.

## 🎓 Project Information

**Author:** Sanyam Jain  
**Roll No:** 22124053  
**Course:** Undergraduate Project  

## 📝 Overview

Image compression is crucial in modern digital applications where storage and transmission efficiency matter. This project explores two fundamental approaches:

1. **Neural Network Autoencoders** - Deep learning approach using PyTorch
2. **Principal Component Analysis (PCA)** - Classical dimensionality reduction technique

## 🚀 Features

- **Linear Autoencoders** for MNIST digit compression
- **Convolutional Autoencoders** for improved spatial feature extraction
- **Custom PCA implementation** from scratch
- **Real image compression** using the provided sample image
- **Visual comparisons** between original and reconstructed images
- **Compression ratio analysis**

## 📁 Repository Structure

```
image-compression/
├── README.md                              # This file
├── Autoencoders.ipynb                     # Neural network autoencoders implementation
├── image_compression_using_pca_.ipynb     # PCA-based compression implementation
└── The First Touch.png                    # Sample image for testing (1920x1080)
```

## 🛠️ Dependencies

### For Autoencoders:
```python
torch
torchvision
matplotlib
numpy
```

### For PCA:
```python
numpy
PIL (Pillow)
matplotlib
sklearn (for dataset comparison)
```

## 📖 Methods Implemented

### 1. Autoencoder Approaches

#### Linear Autoencoders
- **Architecture**: Simple feedforward neural network
- **Input**: 28×28 MNIST images (flattened to 784 dimensions)
- **Compression**: Reduces to lower-dimensional bottleneck layer
- **Activation**: ReLU for encoder, Sigmoid for decoder
- **Loss Function**: Mean Squared Error (MSE)

#### Convolutional Autoencoders
- **Architecture**: CNN-based encoder-decoder
- **Advantages**: 
  - Preserves spatial locality
  - More efficient feature extraction
  - Better reconstruction quality
  - Fewer parameters required
- **Layers**: Conv2d → ReLU → MaxPool2d → ConvTranspose2d

### 2. PCA Implementation

#### Custom PCA Class
- **From-scratch implementation** without sklearn
- **Key Components**:
  - Data centering (mean subtraction)
  - Covariance matrix computation
  - Eigenvalue/eigenvector calculation
  - Component selection based on variance
- **Methods**:
  - `fit(data)`: Train PCA on dataset
  - `compress(data)`: Reduce dimensionality
  - `decompress(data)`: Reconstruct original dimensions

## 🎯 Usage

### Running Autoencoders

1. Open `Autoencoders.ipynb` in Jupyter Notebook or Google Colab
2. Install required dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```
3. Run all cells to see:
   - MNIST data loading and preprocessing
   - Linear autoencoder training and results
   - Convolutional autoencoder implementation
   - Visual comparisons of original vs reconstructed images

### Running PCA Compression

1. Open `image_compression_using_pca_.ipynb`
2. Ensure `The First Touch.png` is in the same directory
3. Install dependencies:
   ```bash
   pip install numpy pillow matplotlib scikit-learn
   ```
4. Run cells to see:
   - Image loading and grayscale conversion
   - PCA fitting and compression
   - Reconstruction and quality analysis
   - Visual comparison of compressed vs original

## 📊 Results and Analysis

### Autoencoder Results
- **Linear Autoencoders**: Effective for digit compression with reasonable reconstruction
- **Convolutional Autoencoders**: Superior performance with better spatial feature preservation
- **Compression Ratios**: Adjustable based on bottleneck layer size

### PCA Results
- **Dimensionality Reduction**: Significant size reduction while preserving key features
- **Quality vs Compression Trade-off**: Adjustable number of components allows control over quality/size balance
- **Real Image Testing**: Successfully compresses the provided 1920×1080 sample image

## 🔍 Key Insights

1. **Autoencoders vs PCA**:
   - Autoencoders: Better for complex, non-linear patterns
   - PCA: Faster, more interpretable, good for linear relationships

2. **Convolutional vs Linear**:
   - Convolutional layers preserve spatial relationships
   - Linear layers work but lose spatial context

3. **Compression Trade-offs**:
   - Higher compression → Lower quality
   - Optimal balance depends on application requirements

## 🎓 Educational Value

This project demonstrates:
- **Deep Learning Fundamentals**: Autoencoder architectures and training
- **Classical ML Techniques**: PCA implementation and theory
- **Computer Vision**: Image preprocessing and visualization
- **PyTorch Framework**: Model building, training, and evaluation
- **Mathematical Concepts**: Eigenvalues, covariance matrices, dimensionality reduction

## 🚀 Future Enhancements

- [ ] Variational Autoencoders (VAE) implementation
- [ ] Comparison with modern compression algorithms (JPEG, WebP)
- [ ] Color image compression support
- [ ] Quantitative metrics (PSNR, SSIM)
- [ ] Interactive compression ratio selector
- [ ] Real-time compression demo

## 📚 References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- PyTorch Documentation: https://pytorch.org/docs/
- Original research papers on autoencoders and PCA

## 🤝 Contributing

This is an educational project. Suggestions and improvements are welcome!

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning.

---

*Developed as part of an undergraduate computer science project exploring the intersection of classical and modern approaches to image compression.*