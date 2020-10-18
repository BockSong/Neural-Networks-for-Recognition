import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms, datasets, models

import matplotlib.pyplot as plt

max_iters = 50
batch_size = 8
learning_rate = 1e-3
input_size = 28 # 28 * 28 = 784
cnn_layers = [2, 'P']
linear_layers = [392, 10] # 784 * 2 / (2*2) = 392

class ConvNet(nn.Module):
    def __init__(self, num_feats, cnn_layers, linear_layers, feat_dim=1):
        super(ConvNet, self).__init__()
        self.features, self.classifier = [], []

        in_channels = feat_dim
        for idx, elem in enumerate(cnn_layers):
            if elem == 'P':
                self.features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                self.features.append(nn.Conv2d(in_channels=in_channels, out_channels=elem, padding=1,
                                             kernel_size=3, stride=1, bias=False))
                self.features.append(nn.BatchNorm2d(elem))
                self.features.append(nn.ReLU(inplace=True))
                in_channels = elem
        self.features = nn.Sequential(*self.features)

        for idx in range(len(linear_layers) - 1):
            self.classifier.append(nn.Linear(linear_layers[idx], linear_layers[idx + 1]))
        self.classifier = nn.Sequential(*self.classifier)
        
    def forward(self, x, evalMode=False):
        features = self.features(x)
        # flatten before the linear layers
        features = features.view(features.size(0), -1)
        
        label_output = self.classifier(features)
        return label_output


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    print("IF_CUDA: ", cuda)

    # load dataset
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_mnist = datasets.MNIST(root="../data",train=True,download=True,transform=trans)

    train_data_num = len(train_mnist)
    print(train_data_num)
    train_loader = DataLoader(train_mnist, shuffle=True, batch_size=batch_size)

    # define model
    model = ConvNet(input_size, cnn_layers, linear_layers, feat_dim=1)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training model
    device = torch.device("cuda" if cuda else "cpu")
    model.to(device)

    train_accs, train_loss = [], []

    for epoch in range(max_iters):
        model.train()
        running_loss, running_acc = 0, 0

        for batch_num, (feats, labels) in enumerate(train_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(feats)

            #print(outputs.size())
            #print(labels.size())
            loss = criterion(outputs, labels.long())
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            _, pred_labels = torch.max(outputs, 1)
            running_acc += torch.sum(torch.eq(pred_labels, labels.long())).item()

            torch.cuda.empty_cache()
            del feats, labels, loss
        
        this_loss, this_acc = running_loss / len(train_loader), running_acc / train_data_num
        #val_loss, val_acc = test(model, val_batches, device)

        print('Epoch ', epoch + 1, ': ', end='')
        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}'.format(this_loss, this_acc))

        train_accs.append(this_acc)
        train_loss.append(this_loss)
        
    # draw plots
    epochs = list(range(max_iters))

    plt.figure("Accuracy")
    plt.title("Acc over epochs")
    plt.xlabel("Epochs") 
    plt.ylabel("Accuracy") 
    plt.plot(epochs, train_accs) 
    plt.savefig("712Accuracy")
    #plt.show()

    plt.figure("Loss")
    plt.title("Averaged loss over epochs")
    plt.xlabel("Epochs") 
    plt.ylabel("Loss") 
    plt.plot(epochs, train_loss) 
    plt.savefig("712Loss")
    #plt.show()
