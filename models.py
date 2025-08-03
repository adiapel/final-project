import torch
from torchvision import datasets
from torchvision import transforms
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

PRETRAINED_MODEL_PATH = "lenet_mnist_model.pth"
EPSILONS = [0.01, 0.1, 0.2, 0.5]
# Required normalization for pretrained ResNet on CIFAR-10
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


class CNN(nn.Module):
    """
    this class creates a convolutional neural network
    """
    def __init__(self):
        """
        a c-tor for the class
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        # MNIST are gray scale so 1 channel, other params are standard for CNN
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        this function defines the data flows in the network
        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def create_MNIST_model():
    """
    this method creates a model for MNIST dataset
    :return:
    """
    transform = transforms.ToTensor()  # convert images to tensor vectors
    # creates model and loads it from data folder
    model = CNN()
    model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    # load MNIST test data
    test_data = datasets.MNIST(root="../data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)  # one example
    return model, test_loader


def create_CIFAR_10_model():
    """
    this method creates a model for CIFAR-10 dataset
    :return:
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    # the next row needs to be run only once to train the model before using it
    # train_model(model, train_loader)
    model.eval()
    model.load_state_dict(torch.load("cifar10_resnet18_finetuned.pth"))
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])  # convert images to tensor vectors and normalize
    test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=transform, download=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    return model, test_loader


def train_model(model, train_loader):
    """
     this method trains resnet18 model on CIFAR-10 dataset and need to be run only once
    :param model:
    :param train_loader:
    :return:
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    for epoch in range(5):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1} loss: {running_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), "cifar10_resnet18_finetuned.pth")
