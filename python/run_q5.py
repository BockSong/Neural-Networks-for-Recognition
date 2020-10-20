import numpy as np
import scipy.io
from nn import *
from util import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
input_size = 1024
hidden1_size = 32
hidden2_size = 32
hidden3_size = 32
output_size = 1024
momentum = 0.9

initialize_weights(input_size, hidden1_size, params, 'layer1')
initialize_weights(hidden1_size, hidden2_size, params, 'layer2')
initialize_weights(hidden2_size, hidden3_size, params, 'layer3')
initialize_weights(hidden3_size, output_size, params, 'output')
assert(params['Wlayer1'].shape == (input_size, hidden1_size))
assert(params['blayer1'].shape == (hidden1_size, ))

for key in params.keys():
    params[key + "_m"] = np.zeros_like(params[key])

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        # forward
        h1 = forward(xb, params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h2, params, 'layer3', relu)
        pred = forward(h3, params, 'output', sigmoid)

        # loss
        #print("input shape: " + str(yb.shape) + str(probs.shape))
        loss = np.sum(np.square(xb - pred))
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss

        # backward
        delta1 = 2 * (xb - pred)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        delta5 = backwards(delta4,params,'layer1',relu_deriv)

        # apply gradient
        params['W' + "output" + "_m"] = params['W' + "output" + "_m"] * momentum - learning_rate * params['grad_W' + "output"]
        params['b' + "output" + "_m"] = params['b' + "output" + "_m"] * momentum - learning_rate * params['grad_b' + "output"]
        params['W' + "output"] += params['W' + "output" + "_m"]
        params['b' + "output"] += params['b' + "output" + "_m"]

        params['W' + "layer3" + "_m"] = params['W' + "layer3" + "_M"] * momentum - learning_rate * params['grad_W' + "layer3"]
        params['b' + "layer3" + "_m"] = params['b' + "layer3" + "_M"] * momentum - learning_rate * params['grad_b' + "layer3"]
        params['W' + "layer3"] += params['W' + "layer3" + "_M"]
        params['b' + "layer3"] += params['b' + "layer3" + "_M"]

        params['W' + "layer2" + "_m"] = params['W' + "layer2" + "_m"] * momentum - learning_rate * params['grad_W' + "layer2"]
        params['b' + "layer2" + "_m"] = params['b' + "layer2" + "_m"] * momentum - learning_rate * params['grad_b' + "layer2"]
        params['W' + "layer2"] += params['W' + "layer2" + "_M"]
        params['b' + "layer2"] += params['b' + "layer2" + "_M"]

        params['W' + "layer1" + "_m"] = params['W' + "layer1" + "_m"] * momentum - learning_rate * params['grad_W' + "layer1"]
        params['b' + "layer1" + "_m"] = params['b' + "layer1" + "_m"] * momentum - learning_rate * params['grad_b' + "layer1"]
        params['W' + "layer1"] += params['W' + "layer1" + "_M"]
        params['b' + "layer1"] += params['b' + "layer1" + "_M"]

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################


# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
