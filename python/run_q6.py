import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32
# do PCA
##########################
##### your code here #####
##########################
#'''
mean_ = np.mean(train_x, axis=0)
X = train_x - mean_
U, S, V = np.linalg.svd(X)

projection_m = V[:dim, :]
print(projection_m.shape) # (32, 1024)

# rebuild a low-rank version
lrank = np.dot(X, np.transpose(projection_m))

# rebuild it
recon = np.dot(lrank, projection_m)
recon += mean_
#'''
# build valid dataset
recon_valid = None
##########################
##### your code here #####
##########################
mean_ = np.mean(valid_x, axis=0)
X = valid_x - mean_
U, S, V = np.linalg.svd(X)
projection_m = V[:dim, :]

lrank = np.dot(X, np.transpose(projection_m))
recon = np.dot(lrank, projection_m)
recon += mean_

# visualize the comparison and compute PSNR
##########################
##### your code here #####
##########################
for c in range(5):
    for n in range(2):
        img = valid_x[c*100+n].reshape(32,32).T
        plt.figure(c*100+n)
        plt.imshow(img)
        plt.savefig("62_" + str(c) + str(n))
        #plt.show()

        rec = recon[c*100+n].reshape((1, -1))

        img_rec = rec.reshape(32,32).T
        plt.figure(c*100+n+50)
        plt.imshow(img_rec)
        plt.savefig("62_" + str(c) + str(n) + "_rec")
        #plt.show()

avg_psnr = psnr(valid_x, recon)
print(avg_psnr)
