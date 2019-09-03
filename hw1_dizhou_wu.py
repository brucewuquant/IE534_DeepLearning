import numpy as np
import h5py
import time
import copy
from random import randint

#load MNIST data
MNIST_data = h5py.File('./MNISTdata.hdf5', 'r')

x_train = np.float32(MNIST_data['x_train'][:])
y_train = np.int32(np.array(MNIST_data['y_train'][:, 0]))
x_test = np.float32(MNIST_data['x_test'][:])
y_test = np.int32(np.array(MNIST_data['y_test'][:, 0]))
MNIST_data.close()

num_inputs = 28 * 28
num_outputs = 10  # classes, denoted a K-dim in class
d_h = 100  # no of hidden neurons
model = {}
model['W1'] = np.random.randn(d_h, num_inputs) / np.sqrt(num_inputs)
model['b1'] = np.random.randn(d_h)
model['b2'] = np.random.randn(num_outputs)
model['C'] = np.random.randn(num_outputs, d_h) / np.sqrt(d_h)
model_grads = copy.deepcopy(model)

def softmax_function(z):
    ZZ = np.exp(z) / np.sum(np.exp(z))
    return ZZ


def relu_function(z):
    return np.maximum(z, 0)


def pd_relu(z, alpha=0.01):
    assert isinstance(z, np.ndarray) == True, 'z must be a numpy array'
    Z = np.ones_like(z)
    Z[z < 0] = alpha
    return Z


def forward(x, y, model):
    Z = np.matmul(model['W1'], x) + model['b1']
    H = relu_function(Z)
    U = np.matmul(model['C'], H) + model['b2']
    p = softmax_function(U)
    return Z, H, U, p


def backward(x, y, Z, H, U, p, model, model_grads):

    e_y = np.zeros(num_outputs)
    e_y[y] = 1.0

    pd_u = -(e_y - p)
    pd_b2 = pd_u
    # Note np.newaxis is alias for None
    # Alternatively:
    #pd_c = np.matmul(pd_u.reshape(-1,1), H.reshape(1,-1))
    pd_c = np.matmul(pd_u[:, np.newaxis], H[np.newaxis, :])

    delta = np.matmul(model['C'].T, pd_u)
    pd_b1 = np.multiply(delta, pd_relu(Z))

    pd_w = np.matmul(pd_b1[:, np.newaxis], x[np.newaxis, :])

    model_grads['C'] = pd_c
    model_grads['b2'] = pd_b2
    model_grads['b1'] = pd_b1
    model_grads['W1'] = pd_w
    return model_grads

time1 = time.time()
LR = 0.01
num_epochs = 11
for epochs in range(num_epochs):
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0

    for n in range(len(x_train)):
        n_random = randint(0, len(x_train) - 1)
        y = y_train[n_random]
        x = x_train[n_random][:]
        Z, H, U, p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x, y, Z, H, U, p, model, model_grads)
        model['C'] = model['C'] - LR * model_grads['C']
        model['b2'] = model['b2'] - LR * model_grads['b2']
        model['b1'] = model['b1'] - LR * model_grads['b1']
        model['W1'] = model['W1'] - LR * model_grads['W1']
    print(total_correct / np.float(len(x_train)))
time2 = time.time()
print(time2 - time1)


#test data
total_correct = 0
for n in range(len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    *_, p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct / np.float(len(x_test)))
