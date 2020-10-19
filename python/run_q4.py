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
    
    #xywh = [((b[1]+b[3])//2, (b[0]+b[2])//2, b[3]-b[1], b[2]-b[0]) for b in bboxes]

    # sort
    #xywh = sorted(xywh, key=lambda b:b[1])
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
            #print(bbox)
            #minr, minc, maxr, maxc = bbox[1]-bbox[3]//2, bbox[0]-bbox[2]//2, bbox[1]+bbox[3]//2, bbox[0]+bbox[2]//2
            minr, minc, maxr, maxc = bbox[0], bbox[1], bbox[2], bbox[3]
            #print(minr, minc, maxr, maxc)
            cropped = bw[minr:maxr, minr:maxr]
            # padding
            if bbox[2] > bbox[3]:
                w_pad = 0
                h_pad = (bbox[2] - bbox[3]) // 2
            else:
                h_pad = 0
                w_pad = (bbox[3] - bbox[2]) // 2
            p = np.pad(cropped, ((h_pad, h_pad), (w_pad, w_pad)), mode='constant', constant_values=(1, 1))
            res = skimage.transform.resize(p, (32, 32))
            trans = np.transpose(res)
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
    idx2str = string.ascii_uppercase + string.digits
    text = ""

    for line in lines:
        for input in line:
            #print(input.shape)
            h1 = forward(input, params, 'layer1')
            probs = forward(h1,params,'output',softmax)
            pred = np.argmax(probs, axis=1)
            text += idx2str[pred[0]]
        text += "\n"

    print(text)
