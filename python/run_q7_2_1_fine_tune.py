import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms, datasets, models

import matplotlib.pyplot as plt

max_iters = 30
batch_size = 8
learning_rate = 1e-3


if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    print("IF_CUDA: ", cuda)

    # load dataset
    data_transform = transforms.Compose([
        transforms.Resize(256), # downsampling first
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    train_dataset = datasets.ImageFolder(root='../data/oxford-flowers17/train',
                                            transform=data_transform)
    train_data_num = len(train_dataset)
    print(train_data_num)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # load pre-trained model
    model = models.squeezenet1_1(pretrained=True)

    # replace the classifier layer
    num_classes = len(train_dataset.classes)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    print(model)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

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
    plt.savefig("721ftAccuracy")
    #plt.show()

    plt.figure("Loss")
    plt.title("Averaged loss over epochs")
    plt.xlabel("Epochs") 
    plt.ylabel("Loss") 
    plt.plot(epochs, train_loss) 
    plt.savefig("721ftLoss")
    #plt.show()
