from torch.utils.data import Dataset
from torch.autograd import Variable
import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import List

class Net(nn.Module):
    # constructor
    def __init__(self):
        super(Net, self).__init__()

        # convolution layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)

        # max pool and ReLU layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # fully connected layers
        self.fully_connected_1 = nn.Linear(16 * 5 * 5, 120)
        self.fully_connected_2 = nn.Linear(120, 48)
        self.fully_connected_3 = nn.Linear(48, 24)
        self.dropout = nn.Dropout(p=0.5)


    # forward propagation method
    def forward(self, x):
        # first convolution layer, and ReLU
        x = self.conv1(x)
        x = self.relu(x)

        # second convolution, ReLU, max pool
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # third convolution, ReLU, max pool
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(-1, 16 * 5 * 5)
        
        # linear fully connected layers
        x = self.fully_connected_1(x)
        x = self.relu(x)

        x = self.fully_connected_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fully_connected_3(x)

        return x

class SignLanguageDataset(Dataset):

    # reads samples and annotations from csv
    def read_annotations_samples(path: str):
        mapping = list(range(25))
        mapping.pop(9)

        #initialize arrays for both labels and samples
        labels, samples = [], []
        with open(path) as f:
            _ = next(f)
            for line in csv.reader(f):
                # the first number in the line is the label
                label = int(line[0])
                labels.append(mapping.index(label))
                # the rest of the numbers are the 784 pixel values
                samples.append(list(map(int, line[1:])))
        return labels, samples

    def __init__(self,
                 path: str="data/sign_mnist_train.csv",
                 mean: List[float]=[0.485],
                 std: List[float]=[0.229]):

        # get samples and labels from above method
        labels, samples = SignLanguageDataset.read_annotations_samples(path)
        # convert to numpy arrays
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1, 28, 28, 1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))

        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])
        return {
            'image': transform(self._samples[index]).float(),
            'label': torch.from_numpy(self._labels[index]).float()
        }

def get_train_loader(batch_size=32):
    trainset = SignLanguageDataset('data/sign_mnist_train.csv')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size,
                                                shuffle=True)
    testset = SignLanguageDataset('data/sign_mnist_test.csv')
    testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size,
                                             shuffle=False)
    return trainloader, testloader
    

def train(net, criterion, optimizer, trainloader, epoch):

    net.train()
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = Variable(data['image'].float())
        labels = Variable(data['label'].long())
        optimizer.zero_grad()

        pred = net(inputs)
        loss = criterion(pred, labels[:, 0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            print("[" + str(epoch + 1) + ", " + str(i) + "] loss: " + str(running_loss / (i + 1)))

    # average loss
    running_loss /= (i + 1)
    return running_loss


# -------------------VALIDATION---------------------- #
def evaluate(outputs: Variable, labels: Variable) -> float:
    y = labels.numpy()
    y_pred = np.argmax(outputs, axis=1)
    return float(np.sum(y_pred == y))

def batch_evaluate(
        net: Net,
        dataloader: torch.utils.data.DataLoader) -> float:
    score = 0.0
    n = 0.0
    for batch in dataloader:
        n += len(batch['image'])
        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        score += evaluate(outputs, batch['label'][:, 0])
    return score / n

# setting up parameters
net = Net().float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
trainloader, testloader = get_train_loader()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print("Data loaded")

train_acc = list()
train_loss = list()
val_acc = list()
best_val_loss = 1
epoch = 15

for i in range(epoch):
    # train and evaluate
    loss = train(net, criterion, optimizer, trainloader, i)
    accuracy = batch_evaluate(net, trainloader)
    val_accuracy = batch_evaluate(net, testloader)
    # add loss and accuracy values to respective lists
    train_loss.append(loss)
    train_acc.append(accuracy)
    val_acc.append(val_accuracy)
    # change learning rate of net
    scheduler.step()

    # print 
    print('-----TRAIN ACCURACY-----')
    print(str(i+1) + ": " + str(train_acc))
    print('-----TEST ACCURACY-----')
    print(str(i+1) + ": " + str(val_acc))
    print('------------------------')


# save model parameters
torch.save(net.state_dict(), "attempt_3_parameters.pth")

# graphing
fig=plt.figure(figsize=(20, 10))
#plt.plot(np.arange(1, epoch+1), train_loss, label="Train loss")
plt.plot(np.arange(1, epoch+1), train_acc, label="Train accuracy")
plt.plot(np.arange(1, epoch+1), val_acc, label="Test accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Plots")
plt.legend(loc='upper right')
plt.savefig('attempt_3_parameters.png')
plt.show()

train_loss = 0.0



        
