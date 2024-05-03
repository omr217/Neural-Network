*IMPORTANT:I use PyCharm for IDE and when I tried to use pytorch with the 3.12 version I got some incompatiblity errors.For that reason I tested the code with version of 3.10 *

EXPLANATION

transforms.Compose: This is used to compose a series of transformations to be applied to the input data.

datasets.FashionMNIST: Downloads the Fashion-MNIST dataset using torchvision. It includes parameters such as the root directory, specifying that it's for training, and applying the defined transform

random_split: Splits the dataset into training and test sets. It calculates the sizes based on the provided ratios.

SimpleNN: Defines a simple neural network with one hidden layer. It uses linear layers and ReLU activation functions.

Trainer: Defines a class for training and evaluating the model. It takes the model, data loaders, loss function , optimizer, and the device for computation.

train: The training method iterates over the specified number of epochs. It sets the model to training mode, iterates through the batches in the training data loader, performs forward and backward passes, and updates the model parameters.

evaluate: Evaluates the model on the test set. It sets the model to evaluation mode, iterates through the batches in the test data loader, computes predictions, and calculates accuracy.

Example Usage: Creates an instance of the model, sets up the loss function and optimizer, creates a trainer, trains the model for 5 epochs, evaluates on the test set, and prints the test accuracy.