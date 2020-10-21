import numpy as np
import scipy.io
from nn import *
from util import *
from collections import Counter

import matplotlib.pyplot as plt

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

train_data_num = len(train_x)

initialize_weights(input_size, hidden1_size, params, 'layer1')
initialize_weights(hidden1_size, hidden2_size, params, 'layer2')
initialize_weights(hidden2_size, hidden3_size, params, 'layer3')
initialize_weights(hidden3_size, output_size, params, 'output')
assert(params['Wlayer1'].shape == (input_size, hidden1_size))
assert(params['blayer1'].shape == (hidden1_size, ))

keys = [k for k in params.keys()]
for key in keys:
    params[key + "_m"] = np.zeros_like(params[key])
#'''
train_loss = []

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
        delta1 = 2 * (pred - xb)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta4 = backwards(delta3,params,'layer2',relu_deriv)
        delta5 = backwards(delta4,params,'layer1',relu_deriv)

        # apply gradient
        layer_name = ["output", "layer3", "layer2", "layer1"]
        for name in layer_name:
            params['W' + name + "_m"] = params['W' + name + "_m"] * momentum - learning_rate * params['grad_W' + name]
            params['b' + name + "_m"] = params['b' + name + "_m"] * momentum - learning_rate * params['grad_b' + name]
            params['W' + name] += params['W' + name + "_m"]
            params['b' + name] += params['b' + name + "_m"]

    total_loss /= train_data_num
    train_loss.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# draw plots
epochs = list(range(max_iters))

plt.figure("Loss")
plt.title("Averaged loss over epochs")
plt.xlabel("Epochs") 
plt.ylabel("Loss") 
plt.plot(epochs, train_loss) 
plt.savefig("52Loss")
#plt.show()
'''

import pickle
'''
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
#'''
#params = pickle.load(open('q5_weights.pickle','rb'))

# Q5.3.1
#import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################
#'''
for c in range(5):
    for n in range(2):
        input = valid_x[c*100+n].reshape((1, -1))

        img = input.reshape(32,32).T
        plt.figure(c*100+n)
        plt.imshow(img)
        plt.savefig("531_" + str(c) + str(n))
        #plt.show()

        h1 = forward(input, params, 'layer1', relu)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h2, params, 'layer3', relu)
        rec = forward(h3, params, 'output', sigmoid)

        img_rec = rec.reshape(32,32).T
        plt.figure(c*100+n+50)
        plt.imshow(img_rec)
        plt.savefig("531_" + str(c) + str(n) + "_rec")
        #plt.show()
#'''
# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
h1 = forward(valid_x, params, 'layer1', relu)
h2 = forward(h1, params, 'layer2', relu)
h3 = forward(h2, params, 'layer3', relu)
pred = forward(h3, params, 'output', sigmoid)

avg_psnr = psnr(valid_x, pred)
print(avg_psnr)
