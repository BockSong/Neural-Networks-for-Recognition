import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 50
# pick a batch size, learning rate
batch_size = 8
learning_rate = 1e-3
hidden_size = 64
##########################
##### your code here #####
##########################
input_size = 1024
output_size = 36

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(input_size, hidden_size, params, 'layer1')
initialize_weights(hidden_size, output_size, params, 'output')
assert(params['Wlayer1'].shape == (input_size, hidden_size))
assert(params['blayer1'].shape == (hidden_size, ))

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def weight_visualize(name):
    fig = plt.figure(1, (8., 8.))
    if hidden_size < 128:
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        img_w = params['Wlayer1'].reshape((32,32,hidden_size))
        for i in range(hidden_size):
            grid[i].imshow(img_w[:,:,i])  # The AxesGrid object work as a list of axes.
        plt.savefig(name)
        #plt.show()

weight_visualize("weight_init.jpg")

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        # forward
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        #print("input shape: " + str(yb.shape) + str(probs.shape))
        loss, acc = compute_loss_and_acc(yb, probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        #print(acc)
        total_acc += acc

        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        delta3 = backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient
        params['W' + "output"] -= learning_rate * params['grad_W' + "output"]
        params['b' + "output"] -= learning_rate * params['grad_b' + "output"]

        params['W' + "layer1"] -= learning_rate * params['grad_W' + "layer1"]
        params['b' + "layer1"] -= learning_rate * params['grad_b' + "layer1"]

    total_acc /= batch_num

    h1 = forward(valid_x, params, 'layer1')
    probs = forward(h1,params,'output',softmax)
    _, valid_acc = compute_loss_and_acc(valid_y, probs)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t train acc : {:.2f} \t val acc : {:.2f}".format(itr,total_loss,total_acc,valid_acc))

# run on validation set and report accuracy! should be above 75%
valid_acc = 0
##########################
##### your code here #####
##########################

val_batches = get_random_batches(valid_x,valid_y,batch_size)
val_batch_num = len(val_batches)

for xb,yb in val_batches:
    h1 = forward(xb, params, 'layer1')
    probs = forward(h1,params,'output',softmax)

    _, acc = compute_loss_and_acc(yb, probs)
    valid_acc += acc

valid_acc /= val_batch_num


print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
weight_visualize("weight_learned.jpg")

# Q3.1.4

fig = plt.figure(1, (6., 8.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(12, 6),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

indices = params['cache_output'][2].argmax(axis=0)
images = valid_x[indices]
images = images.reshape(36, 32, 32)

vis = np.zeros((36, 1024))
inps = np.eye(36)
for i,inp in enumerate(inps):
    vis[i] = inp @ params['Woutput'].T @ params['Wlayer1'].T 
vis = vis.reshape(36, 32, 32)

displayed = np.zeros((72, 32, 32))
displayed[::2] = images
displayed[1::2] = vis
for ax, im in zip(grid, displayed):
    ax.imshow(im.T)
plt.savefig("out.jpg")
plt.show()

# Q3.1.5
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()