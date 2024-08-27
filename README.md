# CUDA-Accelerated Language Model

This project implements a CUDA-accelerated Language Model using C++ and NVIDIA's cuDNN and cuBLAS libraries. The model is designed to be efficient and scalable, suitable for training on large datasets.

## Features

- LSTM-based Recurrent Neural Network (RNN) architecture
- CUDA acceleration for high-performance computing
- Adaptive learning rate schedule with warmup and cooldown phases
- Adam optimizer for efficient weight updates
- Gradient clipping to prevent exploding gradients
- Dropout regularization for better generalization

## Requirements

- CUDA Toolkit (version 10.0 or later recommended)
- cuDNN library
- cuBLAS library
- C++11 compatible compiler
- CMake (for building)

## Class Overview: LanguageModel

The `LanguageModel` class is the core of this implementation. It provides the following key functionalities:

### Constructor

```cpp
LanguageModel(int vocabSize, int hiddenSize, int numLayers, float lr = 0.001, 
              float minLr = 0.00001, int warmup = 1000, int cooldown = 5000, 
              float clip = 1.0, float dropout = 0.1)
