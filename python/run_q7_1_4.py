import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms, datasets, models

import matplotlib.pyplot as plt

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from q4 import *

max_iters = 2
batch_size = 8
learning_rate = 1e-3
input_size = 28 # 28 * 28 = 784
cnn_layers = [2, 'P']
linear_layers = [392, 47] # 784 * 2 / (2*2) = 392

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
    trans = transforms.ToTensor()
    datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'
    train_emnist = datasets.EMNIST(root='../data', split='balanced', train=True, download=True, transform=trans)
    print(train_emnist)

    train_data_num = len(train_emnist)
    print(train_data_num)
    train_loader = DataLoader(train_emnist, shuffle=True, batch_size=batch_size)

    # define model
    model = ConvNet(input_size, cnn_layers, linear_layers, feat_dim=1)
    print(model)
    '''
    # Training model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    
    torch.save(model.state_dict(),'./q7_1_4_weights.pkl')
    '''
    model.load_state_dict(torch.load('./q7_1_4_weights.pkl'))

    # evaluate on findLetters
    for idx, img in enumerate(os.listdir('../images')):
        print(img)
        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
        bboxes, bw = findLetters(im1)
        '''
        plt.figure(str(idx))
        plt.imshow(bw, cmap='gray')
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
        plt.savefig("vis42_" + str(idx))
        #plt.show()
        '''

        # find the rows using..RANSAC, counting, clustering, etc.
        ##########################
        ##### your code here #####
        ##########################
        heights = [b[2] - b[0] for b in bboxes]
        mean_height = sum(heights) / len(heights)
        
        # sort
        bboxes = sorted(bboxes, key=lambda b:b[0])

        # row clustering
        rows = []
        row = []
        line_bottom = bboxes[0][2]
        for bbox in bboxes:
            if bbox[0] < line_bottom + mean_height:
                row.append(bbox)
            else:
                rows.append(row)
                # start new row
                row = [bbox]
                line_bottom = bbox[2]
        rows.append(row)

        # crop the bounding boxes
        # note.. before you flatten, transpose the image (that's how the dataset is!)
        # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
        ##########################
        ##### your code here #####
        ##########################
        lines = []
        for row in rows:
            line = []
            for bbox in row:
                minr, minc, maxr, maxc = bbox[0], bbox[1], bbox[2], bbox[3]
                cropped = bw[minr:maxr, minc:maxc]

                # padding
                w, h = maxc - minc, maxr - minr
                if w > h:
                    w_pad = 0
                    h_pad = (w - h) // 2
                else:
                    h_pad = 0
                    w_pad = (h - w) // 2
                p = np.pad(cropped, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant', constant_values=(1, 1))
                res = skimage.transform.resize(p, (input_size, input_size))
                trans = np.transpose(res)
                flat = np.reshape(trans, (1, -1))
                line.append(flat)

            lines.append(line)
        
        # load the weights
        #import pickle
        #params = pickle.load(open('q3_weights.pickle','rb'))

        import string
        idx2str = string.digits + string.ascii_uppercase + "abdefghnqrt"
        #print(idx2str)
        text = ""

        # run the crops through your neural network and print them out
        model.eval()
        for line in lines:
            for input in line:
                #print(input.shape)
                input = np.reshape(input, (1, 1, input_size, input_size))
                input = torch.from_numpy(input).type(torch.float32)
                outputs = model(input)
                _, pred_labels = torch.max(nn.functional.softmax(outputs, dim=1), 1)
                pred_labels = pred_labels.view(-1)
                #print(pred_labels[0])
        
                text += idx2str[pred_labels[0]]
            text += "\n"

        print(text)
