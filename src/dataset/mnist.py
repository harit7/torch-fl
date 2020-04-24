from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets.mnist import MNIST

class MNISTData:
    def __init__(self,dataPath=None):
        self.trainData = MNIST('./data/mnist', download=True,
                                transform=transforms.Compose([
                                transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        
        self.testData  = MNIST('./data/mnist', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((32, 32)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    