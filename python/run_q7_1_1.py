import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms, datasets, models

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

max_iters = 50
batch_size = 8
learning_rate = 1e-2
input_size = 1024
hidden_size = 64
output_size = 36

class MyNet(nn.Module):
    def __init__(self, size_list):
        super(MyNet, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.Sigmoid())

        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

'''
def test(model, data_loader, device):
    model.eval()
    test_loss = []
    accuracy, total = 0, 0

    for batch_num, (feats, labels) in enumerate(data_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)
        
        _, pred_labels = torch.max(nn.functional.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
        
        loss = criterion(outputs, labels.long())
        
        accuracy += torch.sum(torch.eq(pred_labels, labels.long())).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    return np.mean(test_loss), accuracy / total
'''

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    print("IF_CUDA: ", cuda)

    # load dataset
    train_data = scipy.io.loadmat('../data/nist36_train.mat')
    #test_data = scipy.io.loadmat('../data/nist36_test.mat')
    #valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

    train_x = torch.from_numpy(train_data['train_data']).type(torch.float32)
    # labels must be in Long type and in direct value (rather than the one-hot format)
    train_y = torch.from_numpy(np.argmax(train_data['train_labels'], axis=1)).type(torch.LongTensor)
    train_data_num = train_y.size()[0]
    print(train_data_num)
    train_loader = DataLoader(TensorDataset(train_x, train_y), shuffle=True, batch_size=batch_size)

    # define model
    model = MyNet([input_size, hidden_size, output_size])
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training model
    device = torch.device("cuda" if cuda else "cpu")
    model.to(device)

    for epoch in range(max_iters):
        model.train()
        running_loss, acc = 0, 0

        for batch_num, (feats, labels) in enumerate(train_loader):
            feats, labels = feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(feats)

            _, pred_labels = torch.max(nn.functional.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)
            acc += torch.sum(torch.eq(pred_labels, labels.long())).item()

            #print(outputs.size())
            #print(labels.size())
            loss = criterion(outputs, labels.long())
            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            
            torch.cuda.empty_cache()
            del feats, labels, loss
        
        train_loss, train_acc = running_loss / len(train_loader), acc / train_data_num
        #val_loss, val_acc = test(model, val_batches, device)

        print('Epoch ', epoch + 1, ': ', end='')
        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}'.format(train_loss, train_acc))
        