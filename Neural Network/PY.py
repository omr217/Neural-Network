import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Define the FashionMNISTDataset class
class FashionMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

# Download Fashion-MNIST dataset using torchvision
transform = transforms.Compose([transforms.ToTensor()])
fashion_mnist_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

# Split into training and test sets
train_size = int(0.8 * len(fashion_mnist_dataset))
test_size = len(fashion_mnist_dataset) - train_size
train_dataset, test_dataset = random_split(fashion_mnist_dataset, [train_size, test_size])

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 2: Neural Network Model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Step 3: Training the Model

class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

# Example Usage

model = SimpleNN(input_size=28*28, hidden_size=128, output_size=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu')

trainer.train(epochs=5)

test_accuracy = trainer.evaluate()

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
