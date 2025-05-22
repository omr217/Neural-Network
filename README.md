# 🧠 Fashion-MNIST Classification with PyTorch

This project demonstrates how to classify images from the **Fashion-MNIST** dataset using a **simple feedforward neural network** implemented in PyTorch. It includes dataset loading, model definition, training, and evaluation.

## 📦 Features

- Loads and processes the Fashion-MNIST dataset using `torchvision`
- Defines a custom feedforward neural network (`SimpleNN`)
- Implements a `Trainer` class to handle training and evaluation
- Uses `CrossEntropyLoss` and SGD optimizer
- Supports GPU (if available)

### 🧪 Dataset

**Fashion-MNIST** is a dataset of Zalando's article images—consisting of 60,000 training examples and 10,000 test examples.

Each example is a 28x28 grayscale image, associated with a label from 10 fashion categories.

The dataset is automatically downloaded using `torchvision.datasets.FashionMNIST`.

#### 📁 Project Structure

```python
- Dataset loading using torchvision
- Train/test split with random_split
- Neural network definition (SimpleNN)
- Training and evaluation logic (Trainer)
```

##### 🧠 Model Architecture

Input Layer (784 units - 28x28 flattened)
   ↓
Fully Connected Layer (128 units)
   ↓
ReLU Activation
   ↓
Fully Connected Layer (10 units for classification)

###### 📌 Notes

The model is basic and can be improved using CNNs for better accuracy.

Learning rate and epoch count can be adjusted for tuning performance.

Works with CPU and GPU (automatically detected).

###### 📜 License

This project is open-source and available under the MIT License.
