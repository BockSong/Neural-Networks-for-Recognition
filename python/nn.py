import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    a = np.sqrt(6. / (in_size + out_size))
    W = np.random.uniform(low=-a, high=a, size=(in_size, out_size))
    b = np.zeros((out_size))

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    ##########################
    res = 1 / (1. + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    ##########################
    ##### your code here #####
    ##########################
    pre_act = np.dot(X, W) + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    examples, classes = x.shape[0], x.shape[1]
    res = np.empty((examples, classes))

    ##########################
    ##### your code here #####
    ##########################
    c = - np.max(x, axis=1)
    c = np.reshape(c, (c.shape[0], 1))
    xc = x + c

    sumexp = np.sum(np.exp(xc), axis=1)
    sumexp = sumexp.reshape((sumexp.shape[0], 1))
    res = np.exp(xc) / sumexp

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes], label
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    label = np.argmax(y, axis=1)
    pred = np.argmax(probs, axis=1)
    corr = np.count_nonzero(np.where(label == pred, 1, 0)) * 1.
    #print(label)
    #print(pred)
    acc = corr / label.shape[0]
    loss = - np.sum(y * np.log(probs))

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################
    dz = delta * activation_deriv(post_act)

    #print(X.shape)
    grad_X = np.dot(dz, W.T)
    #print(W.shape)
    grad_W = np.dot(X.T, dz)
    grad_b = np.sum(dz, axis=0)
    #print("b: " + str(b.shape))
    #print("db" + str(grad_b.shape))

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    #print(x.shape)
    #print(y.shape)
    # remove edge data to match batch size
    num = x.shape[0] // batch_size
    new_shape = num * batch_size
    x = x[:new_shape, :]
    y = y[:new_shape, :]
    # create batches
    bx = np.split(x, num, axis=0)
    by = np.split(y, num, axis=0)
    #print(len(bx))
    #print(len(by))
    for i in range(num):
        batches.append((bx[i], by[i]))
    
    #print(batches)
    return batches
