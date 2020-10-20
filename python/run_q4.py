import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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
    
    # sort by rows
    bboxes = sorted(bboxes, key=lambda b:b[0])

    # row clustering
    rows = []
    row = []
    line_bottom = bboxes[0][2]
    for bbox in bboxes:
        if bbox[2] < line_bottom + mean_height:
            # if in the same row
            row.append(bbox)
        else:
            # sort by columns and save the current row
            row = sorted(row, key=lambda b:b[1])
            rows.append(row)
            # start a new row
            row = [bbox]
            line_bottom = bbox[2]
    # sort and add last row
    row = sorted(row, key=lambda b:b[1])
    rows.append(row)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    lines = []
    for i, row in enumerate(rows):
        line = []
        for j, bbox in enumerate(row):
            minr, minc, maxr, maxc = bbox[0], bbox[1], bbox[2], bbox[3]
            cropped = bw[minr:maxr, minc:maxc]
            
            # padding
            w, h = maxc - minc, maxr - minr
            if w > h:
                w_pad = w // 8
                h_pad = (w - h) // 2 + w_pad
            else:
                h_pad = h // 8
                w_pad = (h - w) // 2 + h_pad
            p = np.pad(cropped, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant', constant_values=(1, 1))
            res = skimage.transform.resize(p, (32, 32))
            
            # greyscale morphological erosion
            mor = skimage.morphology.erosion(res, skimage.morphology.square(3))
            
            # transpose to match training dataset style
            trans = np.transpose(mor)
            '''
            plt.figure(str(i) + ", " + str(j))
            plt.imshow(trans, cmap='gray')
            plt.show()
            '''
            flat = np.reshape(trans, (1, -1))
            line.append(flat)

        lines.append(line)
    
    # load the weights
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    # run the crops through your neural network and print them out
    ##########################
    ##### your code here #####
    ##########################
    #idx2str = string.ascii_uppercase + string.digits
    #print(idx2str)
    text = ""

    for line in lines:
        for input in line:
            #print(input.shape)
            h1 = forward(input, params, 'layer1')
            probs = forward(h1,params,'output',softmax)
            pred = np.argmax(probs, axis=1)
            text += letters[pred[0]]
        text += "\n"

    print(text)
