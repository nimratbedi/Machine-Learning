import numpy as np


# convolutional layer
class Convolution:

    def __init__(self, in_channel, num_of_filter, size_of_kernel, pad, stride, lr, name_of_layer):
        self.filter = num_of_filter
        self.kernel = size_of_kernel
        self.chnls = in_channel
        self.weights = np.zeros((self.filter, self.chnls, self.kernel, self.kernel))

        print("Inside Convolution")

        print("self.weights shape : ", self.weights.shape)
        #print("self.weights before initializaion : ", self.weights)
        self.bias = np.zeros((self.filter, 1))
        # initializing weights
        for i in range(0, self.filter):
            self.weights[i, :, :, :] = np.random.normal(loc=0, scale=np.sqrt(
                1. / (self.chnls * self.kernel * self.kernel)), size=(self.chnls, self.kernel, self.kernel))
        #print("self.weights after initializaion : ", self.weights)


        #print("self.weights after adding new weights shape : ", self.weights.shape)
        #print("self.weights after adding new weights : ", self.weights)
        self.pading = pad
        self.Stride = stride
        self.learning_rate = lr
        self.name_of_layer = name_of_layer

    # forward propagation
    def fwd_pass(self, input_data):
        print("Convolution Fwd Pass input data shape : ",input_data.shape)
        chnnls = input_data.shape[2]
        Wd = input_data.shape[0] + 2 * self.pading
        Ht = input_data.shape[1] + 2 * self.pading
        self.input_data = np.zeros((Wd, Ht, chnnls))
        for c in range(input_data.shape[2]):
            self.input_data[:, :, c] = self.zero_pad(input_data[:, :, c], self.pading)
        Ww = int((Wd - self.kernel) / self.Stride) + 1
        Hh = int((Ht - self.kernel) / self.Stride) + 1
        features = np.zeros((Ww, Hh, self.filter))
        # running convolution
        try:
            for fill in range(self.filter):
                for chn in range(chnnls):
                    for q in range(Ww):
                        for n in range(Hh):
                            features[q, n, fill] = np.sum(
                                self.input_data[q:q + self.kernel, n:n + self.kernel, chn] * self.weights[fill, chn, :,:]) + self.bias[fill]
        except IndexError:
            print("index error!!!")
        print("Convolution Fwd Pass output data shape : ",features.shape)
        return features

    # Zero pad
    def zero_pad(self, input_data, size):
        width, height = input_data.shape[0], input_data.shape[1]
        new_wd = 2 * size + width
        new_ht = 2 * size + height
        output = np.zeros((new_wd, new_ht))
        output[size:width + size, size:height + size] = input_data
        return output

    # backward propagation
    def bck_pass(self, Dy):
        Wd, Ht, chnnls = self.input_data.shape
        Dx = np.zeros(self.input_data.shape)
        Dw = np.zeros(self.weights.shape)
        Db = np.zeros(self.bias.shape)
        Wd, Ht, F = Dy.shape
        print("Dx shape",Dx.shape)
        print("Dw shape",Dw.shape)
        print("Db shape",Db.shape)
        print("Convolution Back propagation input_data.shape (Wd, Ht, chnnls): ",self.input_data.shape)
        print("Convolution Back propagation Dy.shape (Wd, Ht, F): ",Dy.shape)

        for d in range(F):
            for chn in range(3):
                for q in range(Wd):
                    for n in range(Ht):
                        #print("d in F :",d," chn in chnnls : ",chn," q in Wd : ",q," n in Ht : ",n)
                        Dw[d, chn, :, :] += Dy[q, n, d] * self.input_data[q:q + self.kernel, n:n + self.kernel, chn]
                        Dx[q:q + self.kernel, n:n + self.kernel, chn] += Dy[q, n, d] * self.weights[d, chn, :, :]
        for d in range(F):
            Db[d] = np.sum(Dy[:, :, d])
        self.weights -= self.learning_rate * Dw
        self.bias -= self.learning_rate * Db
        print("Convolution Bck Prop Return Dx shape :",Dx.shape)
        return Dx

    # weight and bias filling
    def feeding_wts(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # weight and bias extraction
    def extract_wts(self):
        return {self.name_of_layer + '.weights': self.weights, self.name_of_layer + '.bias': self.bias}


# fully Connected layer
class FullyConnected:

    def __init__(self, num_inputs, num_outputs, lr, name_of_layer):
        self.weights = 0.01 * np.random.rand(num_inputs, num_outputs)
        self.bias = np.zeros((num_outputs, 1))
        self.learning_rate = lr
        self.name_of_layer = name_of_layer

    # back propagation
    def bck_pass(self, Dy):
        if Dy.shape[0] == self.input_data.shape[0]:
            Dy = Dy.T
        Dw = Dy.dot(self.input_data)
        Db = np.sum(Dy, axis=1, keepdims=True)
        Dx = np.dot(Dy.T, self.weights.T)

        self.weights -= self.learning_rate * Dw.T
        self.bias -= self.learning_rate * Db

        return Dx

    # forward propagation
    def fwd_pass(self, input_data):
        self.input_data = input_data
        return np.dot(self.input_data, self.weights) + self.bias.T

    # weight and bias filling
    def feeding_wts(self, weights, bias):
        self.weights = weights
        self.bias = bias

    # weight and bias extraction
    def extract_wts(self):
        return {self.name_of_layer + '.weights': self.weights, self.name_of_layer + '.bias': self.bias}


# maxpooling layer
class Maxpool:

    def __init__(self, size_of_pool, stride, name_of_layer):
        self.pool = size_of_pool
        self.Stride = stride
        self.name_of_layer = name_of_layer

    # forward propagation
    def fwd_pass(self, input_data):
        try:
            print("In the forward pass of maxpooling")
            self.input_data = input_data
            Wd, Ht, chnnls = input_data.shape
            new_width = int((Wd - self.pool) / self.Stride) + 1
            new_height = int((Ht - self.pool) / self.Stride) + 1

            l1 = int(Wd / self.Stride)
            l2 = int(Ht / self.Stride)

            output = np.zeros((new_width, new_height, chnnls))
            for c in range(chnnls):
                for q in range(l1):
                    for n in range(l2):
                        output[q, n, c] = np.max(
                            self.input_data[q * self.Stride:q * self.Stride + self.pool,
                            n * self.Stride:n * self.Stride + self.pool, c])
        except:
            print("maxpool layer error!!!")
        print("Maxpool output shape : ",output.shape)
        #print("Maxpool output : ",output)
        return output
    
    # back propagation
    def bck_pass(self, Dy):
        print()
        print("In the backward pass of maxpooling")
        print("self.input_data shape",self.input_data.shape)
        mask = np.ones_like(self.input_data) * 0.25
        print("Dy.shape=",Dy.shape)
        print("mask shape=",mask.shape)
        #print("Dy =",Dy)
        #print("mask =",mask)
        #print("np repeat inner shape : ",np.repeat(Dy[0:-4, 0:-4:], 2, axis=0).shape)
        print("np repeat inner shape : ",np.repeat(np.repeat(Dy[0:Dy.shape[0], 0:Dy.shape[0]:], 2, axis=0), 2, axis=1).shape)
        #print("np repeat inner matrix : ",np.repeat(np.repeat(Dy[0:Dy.shape[0], 0:Dy.shape[0]:], 2, axis=0), 2, axis=1))
        return mask * (np.repeat(np.repeat(Dy[0:Dy.shape[0], 0:Dy.shape[0]:], 2, axis=0), 2, axis=1))
        

    # weight and bias extraction
    def extract_wts(self):
        return


# Creating the Flattening layer Class
class Flatten:
    def __init__(self):
        pass

    # back propagation
    def bck_pass(self, Dy):
        return Dy.reshape(self.Wd, self.Ht, self.chnls)

    # forward propagation
    def fwd_pass(self, input_data):
        self.Wd, self.Ht, self.chnls, = input_data.shape
        return input_data.reshape(1, self.chnls * self.Wd * self.Ht)

    # weight and bias extraction
    def extract_wts(self):
        return


# softmax layer
class Softmax:
    def __init__(self):
        pass

    # back propagation
    def bck_pass(self, Dy):
        return self.output.T - Dy.reshape(Dy.shape[1], 1)

    # forward propagation
    def fwd_pass(self, input_data):
        exp_probability = np.exp(input_data, dtype=np.float)
        self.output = exp_probability / np.sum(exp_probability)
        return self.output

    # weight and bias extraction
    def extract_wts(self):
        return


# ReLu activation
class ReLu:
    def __init__(self):
        pass

    # back propagation
    def bck_pass(self, Dy):
        Dx = Dy.copy()
        Dx[self.input_data < 0] = 0
        return Dx

    # forward propagation
    def fwd_pass(self, input_data):
        self.input_data = input_data
        relu = input_data.copy()
        relu[relu < 0] = 0
        return relu

    # weight and bias extraction
    def extract_wts(self):
        return
