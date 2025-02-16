import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classs = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  #list of all 10 classes
class_names_5 = ['airplane', 'automobile', 'bird', 'cat', 'truck']         #list of 5 classes
class_indices_5 = [0, 1, 2, 3, 9]

train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_10 = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
datatotrain = Subset(train_10, [i for i in range(len(train_10)) if train_10.targets[i] in class_indices_5])
sizeofdatattotrain = int(0.8 * len(train_10))
sizeofval = len(train_10) - sizeofdatattotrain
trainset, valset = torch.utils.data.random_split(train_10, [sizeofdatattotrain, sizeofval])

class_images = []
for i in range(10):
    class_images.append(next(img for img, name in trainset if name == i))
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.transpose(class_images[i], (1, 2, 0)))
    plt.title(classs[i])
plt.show()

test_10 = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
datatotest = Subset(test_10, [i for i in range(len(test_10)) if test_10.targets[i] in class_indices_5])
valid_size = 0.1
number_of_data_to_train = len(datatotrain)
indices = list(range(number_of_data_to_train))
split = int(np.floor(valid_size * number_of_data_to_train))
np.random.seed(42)
np.random.shuffle(indices)
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
train_loader = torch.utils.data.DataLoader(datatotrain, batch_size=64, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(datatotrain, batch_size=64, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(datatotest, batch_size=64, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def make_layer(self, block, out_channel, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channel, out_channel, stride))
        self.in_channel = out_channel
        for _ in range(1, blocks):
            layers.append(block(out_channel, out_channel))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

num_of_epoch = 20
best_valid_acc = 0.0
loss_in_traindata = []
acc_in_traindata = []
loss_in_validdata = []
acc_in_vailddata = []

for epoch in range(num_of_epoch):
    model.train()
    loss_traindata = 0.0
    correct1 = 0
    train_total = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_traindata += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        correct1 += (predicted == labels).sum().item()
    train_accuracy = 100 * correct1 / train_total
    loss_traindata /= len(train_loader)

    loss_in_traindata.append(loss_traindata)
    acc_in_traindata.append(train_accuracy)

    model.eval()

    loss_validdata = 0.0
    correct2 = 0
    valid_total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_validdata += loss.item()
            _, predicted = torch.max(outputs, 1)
            valid_total += labels.size(0)
            correct2 += (predicted == labels).sum().item()
    valid_accuracy = 100 * correct2 / valid_total
    loss_validdata /= len(valid_loader)

    loss_in_validdata.append(loss_validdata)
    acc_in_vailddata.append(valid_accuracy)

    scheduler.step(loss_validdata)

    print(f'Epoch [{epoch + 1}/{num_of_epoch}], '
          f'Train Loss: {loss_traindata:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Valid Loss: {loss_validdata:.4f}, Valid Accuracy: {valid_accuracy:.2f}%')

    if valid_accuracy > best_valid_acc:
        best_valid_acc = valid_accuracy
        torch.save(model.state_dict(), 'resnet_cifar10.pth')

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(loss_in_traindata, label='Train Loss')
plt.plot(loss_in_validdata, label='Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.legend()
plt.title('Losses vs Epochs')
plt.subplot(1, 2, 2)
plt.plot(acc_in_traindata, label='Train Accuracy')
plt.plot(acc_in_vailddata, label='Valid Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.legend()
plt.title('Accuracy vs Epoch')
plt.show()