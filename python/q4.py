import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    # blur or denoise
    #blur = skimage.filters.gaussian(image)
    denoise = skimage.restoration.denoise_bilateral(image, multichannel=True)

    # greyscale
    grey = skimage.color.rgb2gray(denoise)

    # threshold
    thresh = skimage.filters.threshold_otsu(grey)
    binary = grey <= thresh

    # morphology: using opening may fail; lower para in sqaure() may lead to failure
    cl = skimage.morphology.closing(binary, skimage.morphology.square(7))
    bw = (1 - cl).astype(np.float)
    #np.set_printoptions(threshold=np.inf)
    #print(cl[:500,:500])

    # labeling using grouping method
    label_img = skimage.measure.label(cl, connectivity=2)
    props = skimage.measure.regionprops(label_img)

    # add bboxes and skip small boxes
    mean_area = sum([x.area for x in props]) / len(props)
    thresh_area = mean_area / 3.

    for x in props:
        if x.area > thresh_area:
            bboxes.append(x.bbox)

    return bboxes, bw