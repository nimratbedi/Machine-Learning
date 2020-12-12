import matplotlib.image as mpimg
from skimage.transform import rescale, resize
import glob
import numpy as np
import pickle
import sys
import json
import requests
from skimage import io

# class for creating the layer architecture
class CNN:
    def __init__(self):
        learning_rate = 0.01
        self.layers = []
        print("In CNN class")
        # L: 0
        # Call convolution from layer.py
        print("Network.py Convolution 1-----------------")
        self.layers.append(
            Convolution(in_channel=3, num_of_filter=16, size_of_kernel=3, pad=0, stride=1, lr=learning_rate,
                        name_of_layer='conv_1'))

        # Ly: 1
        print("Network.py Relu 1-----------------")
        self.layers.append(ReLu())
        # Ly: 2
        print("Network.py Maxpool 1-----------------")
        self.layers.append(Maxpool(size_of_pool=2, stride=2, name_of_layer='maxpool_1'))
        # Ly: 3
        print("Network.py Convolution 2-----------------")
        self.layers.append(
            Convolution(in_channel=3, num_of_filter=32, size_of_kernel=3, pad=0, stride=1, lr=learning_rate,
                        name_of_layer='conv_2'))
        # Ly: 4
        print("Network.py Relu 2-----------------")
        self.layers.append(ReLu())
        # Ly: 5
        print("Network.py Maxpool 2-----------------")
        self.layers.append(Maxpool(size_of_pool=2, stride=2, name_of_layer='maxpool_2'))
        # Ly: 3
        print("Network.py Convolution 3-----------------")
        self.layers.append(
            Convolution(in_channel=3, num_of_filter=64, size_of_kernel=3, pad=0, stride=1, lr=learning_rate,
                        name_of_layer='conv_2'))
        # Ly: 4
        print("Network.py Relu 3-----------------")
        self.layers.append(ReLu())
        # Ly: 5
        print("Network.py Maxpool 3-----------------")
        self.layers.append(Maxpool(size_of_pool=2, stride=2, name_of_layer='maxpool_2'))
        # Ly: 6
        print("Network.py Convolution 4-----------------")
        self.layers.append(
            Convolution(in_channel=3, num_of_filter=128, size_of_kernel=3, pad=0, stride=1, lr=learning_rate,
                        name_of_layer='conv_2'))
        # Ly: 4
        print("Network.py Relu 4-----------------")
        self.layers.append(ReLu())
        # Ly: 5
        print("Network.py Maxpool 4-----------------")
        self.layers.append(Maxpool(size_of_pool=2, stride=2, name_of_layer='maxpool_2'))
        # self.layers.append(Convolution(in_channel=16, num_of_filter=72, size_of_kernel=5, pad=2, stride=1, lr=learning_rate, name_of_layer='conv_5'))
        # Ly: 7
        # self.layers.append(ReLu())
        # Ly: 8
        print("Network.py Flatten-----------------")
        self.layers.append(Flatten())
        # Ly: 9
        print("Network.py Fully connected 1-----------------")
        self.layers.append(FullyConnected(num_inputs=2048, num_outputs=64, lr=learning_rate, name_of_layer='fc_6'))
        # Ly: 10
        print("Network.py Relu 3-----------------")
        self.layers.append(ReLu())
        # Ly: 11
        print("Network.py Fully Connected 2-----------------")
        self.layers.append(FullyConnected(num_inputs=64, num_outputs=2, lr=learning_rate, name_of_layer='fc_7'))
        # Ly: 12
        print("Network.py Softmax-----------------")
        self.layers.append(Softmax())
        self.layers_num = len(self.layers)

    # model testing
    def test(self, test_input, test_input_label, test_input_size):
        accuracy_total = 0
        for j in range(test_input_size):
            X0 = test_input[j]
            Y0 = test_input_label[j]
            for lays in range(self.layers_num):
                output = self.layers[lays].fwd_pass(X0)
                X0 = output
            # Calculating accuracy of test
            if np.argmax(output) == np.argmax(Y0):
                accuracy_total += 1
        sys.stdout.write("\n")
        print('Number of test data :{0:d},  test accuracy :{1:.2f}'.format(test_input_size,
                                                                           float(accuracy_total) / float(
                                                                               test_input_size)))

    # model training
    def train(self, training_input, training_input_label, size_of_batch, epocs, wts_file):
        print("Network.py Inside training")
        accuracy_total = 0
        for ep in range(epocs):
            for index_of_batch in range(0, training_input.shape[0], size_of_batch):
                # batch-input for giving the training data
                if index_of_batch + size_of_batch < training_input.shape[0]:
                    training_data = training_input[index_of_batch:index_of_batch + size_of_batch]
                    training_labels = training_input_label[index_of_batch:index_of_batch + size_of_batch]
                else:
                    training_data = training_input[index_of_batch:training_input.shape[0]]
                    training_labels = training_input_label[index_of_batch:training_input_label.shape[0]]
                loss = 0
                accuracy = 0

                for bs in range(size_of_batch):
                    training_img = training_data[bs]
                    training_label = training_labels[bs]
                    # forward pass
                    for lays in range(self.layers_num):
                        output = self.layers[lays].fwd_pass(training_img)
                        training_img = output
                    loss += self.cross_entropy_loss(output, training_label)
                    if np.argmax(output) == np.argmax(training_label):
                        accuracy += 1
                        accuracy_total += 1
                    Dy = output
                    print("Network.py Dy = output & Dy.shape=", Dy.shape)
                    for lays in range(self.layers_num - 1, -1, -1):
                        Dout = self.layers[lays].bck_pass(Dy)
                        Dy = Dout

                # Calculating loss and accuracy of model.
                loss /= size_of_batch
                batch_accuracy = float(accuracy) / float(size_of_batch)
                training_accuracy = float(accuracy_total) / float((index_of_batch + size_of_batch) * (ep + 1))
                print('Epoch: {0:d}/{1:d}, Iter:{2:d}, loss: {3:.2f}, Batch Accuracy: {4:.2f}, Training Accuracy: {5:.2f}'.format(
                    ep + 1, epocs, index_of_batch + size_of_batch, loss, batch_accuracy, training_accuracy))
            # dumping weights and biases into the pickle file after each epocs.
            weights = []
            for j in range(self.layers_num):
                wt = self.layers[j].extract_wts()
                weights.append(wt)
            with open(wts_file, 'ab') as pick_file:
                pickle.dump(weights, pick_file, protocol=pickle.HIGHEST_PROTOCOL)

    # predicting data with trained weights
    def pred_img_train_wts(self, input_data, wts_file):
        with open(wts_file, 'rb') as pick_file:
            wt = pickle.load(pick_file)
        # loading the trained weights and biases
        self.layers[0].feeding_wts(wt[0]['conv_1.weights'], wt[0]['conv_1.bias'])
        self.layers[3].feeding_wts(wt[3]['conv_3.weights'], wt[3]['conv_3.bias'])
        # self.layers[6].feeding_wts(wt[6]['conv_5.weights'], wt[6]['conv_5.bias'])
        self.layers[6].feeding_wts(wt[6]['fc_6.weights'], wt[6]['fc_6.bias'])
        self.layers[8].feeding_wts(wt[8]['fc_7.weights'], wt[8]['fc_7.bias'])
        for l in range(self.layers_num):
            output = self.layers[l].fwd_pass(input_data)
            input_data = output
        scene_class = np.argmax(output)
        # Calculating probabilities of each class labels
        prob = output[0, scene_class]
        return scene_class, prob

    # calculate cross entropy loss
    def cross_entropy_loss(self, inputs, class_labels):
        output_numb = class_labels.shape[0]
        prob = np.sum(class_labels.reshape(1, output_numb) * inputs)
        entropy_loss = -np.log(prob)
        return entropy_loss


    # Testing model from the trained weights
    def test_train_wts(self, test_input, test_input_label, test_input_size, wts_file):
        with open(wts_file, 'rb') as pick_file:
            wt = pickle.load(pick_file)

        # Loading the trained weights and biases
        self.layers[0].feeding_wts(wt[0]['conv_1.weights'], wt[0]['conv_1.bias'])
        self.layers[3].feeding_wts(wt[3]['conv_3.weights'], wt[3]['conv_3.bias'])
        # self.layers[6].feeding_wts(wt[6]['conv_5.weights'], wt[6]['conv_5.bias'])
        self.layers[6].feeding_wts(wt[6]['fc_6.weights'], wt[6]['fc_6.bias'])
        self.layers[8].feeding_wts(wt[8]['fc_7.weights'], wt[8]['fc_7.bias'])
        # Calculating accuracy
        accuracy_total = 0
        for j in range(test_input_size):
            x = test_input[j]
            y = test_input_label[j]
            for l in range(self.layers_num):
                output = self.layers[l].fwd_pass(x)
                x = output
            if np.argmax(output) == np.argmax(y):
                accuracy_total += 1
        sys.stdout.write("\n")
        print('Number of test data :{0:d}, test accuracy :{1:.2f}'.format(test_input_size,
                                                                          float(accuracy_total) / float(
                                                                              test_input_size)))


# convolutional layer
class Convolution:

    def __init__(self, in_channel, num_of_filter, size_of_kernel, pad, stride, lr, name_of_layer):
        self.filter = num_of_filter
        self.kernel = size_of_kernel
        self.chnls = in_channel
        self.weights = np.zeros((self.filter, self.chnls, self.kernel, self.kernel))

        print("Inside Convolution")

        print("self.weights shape : ", self.weights.shape)
        # print("self.weights before initializaion : ", self.weights)
        self.bias = np.zeros((self.filter, 1))
        # initializing weights
        for i in range(0, self.filter):
            self.weights[i, :, :, :] = np.random.normal(loc=0, scale=np.sqrt(
                1. / (self.chnls * self.kernel * self.kernel)), size=(self.chnls, self.kernel, self.kernel))
        # print("self.weights after initializaion : ", self.weights)

        # print("self.weights after adding new weights shape : ", self.weights.shape)
        # print("self.weights after adding new weights : ", self.weights)
        self.pading = pad
        self.Stride = stride
        self.learning_rate = lr
        self.name_of_layer = name_of_layer

    # forward propagation
    def fwd_pass(self, input_data):
        print("Convolution Fwd Pass input data shape : ", input_data.shape)
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
                                self.input_data[q:q + self.kernel, n:n + self.kernel, chn] * self.weights[fill, chn, :,
                                                                                             :]) + self.bias[fill]
        except IndexError:
            print("index error!!!")
        print("Convolution Fwd Pass output data shape : ", features.shape)
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
        print("Dx shape", Dx.shape)
        print("Dw shape", Dw.shape)
        print("Db shape", Db.shape)
        print("Convolution Back propagation input_data.shape (Wd, Ht, chnnls): ", self.input_data.shape)
        print("Convolution Back propagation Dy.shape (Wd, Ht, F): ", Dy.shape)

        for d in range(F):
            for chn in range(3):
                for q in range(Wd):
                    for n in range(Ht):
                        # print("d in F :",d," chn in chnnls : ",chn," q in Wd : ",q," n in Ht : ",n)
                        Dw[d, chn, :, :] += Dy[q, n, d] * self.input_data[q:q + self.kernel, n:n + self.kernel, chn]
                        Dx[q:q + self.kernel, n:n + self.kernel, chn] += Dy[q, n, d] * self.weights[d, chn, :, :]
        for d in range(F):
            Db[d] = np.sum(Dy[:, :, d])
        self.weights -= self.learning_rate * Dw
        self.bias -= self.learning_rate * Db
        print("Convolution Bck Prop Return Dx shape :", Dx.shape)
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
        print("Maxpool output shape : ", output.shape)
        # print("Maxpool output : ",output)
        return output

    # back propagation
    def bck_pass(self, Dy):
        print()
        print("In the backward pass of maxpooling")
        print("self.input_data shape", self.input_data.shape)
        mask = np.ones_like(self.input_data) * 0.25
        print("Dy.shape=", Dy.shape)
        print("mask shape=", mask.shape)
        # print("Dy =",Dy)
        # print("mask =",mask)
        # print("np repeat inner shape : ",np.repeat(Dy[0:-4, 0:-4:], 2, axis=0).shape)
        print("np repeat inner shape : ",
              np.repeat(np.repeat(Dy[0:Dy.shape[0], 0:Dy.shape[0]:], 2, axis=0), 2, axis=1).shape)
        # print("np repeat inner matrix : ",np.repeat(np.repeat(Dy[0:Dy.shape[0], 0:Dy.shape[0]:], 2, axis=0), 2, axis=1))
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

def largest_multiple(num):
    half = int(num/2)
    cur_large_mul = 1;
    for r in range(1, half):
        if num%r == 0:
            if r > cur_large_mul:
                cur_large_mul = r
    return cur_large_mul

print('loading and preparing training data...')
# train_covid = https://api.jsonbin.io/b/5fbd79a14f12502c21d82224
# train_normal = https://api.jsonbin.io/b/5fbd794f4f12502c21d82210
urls_train = ["https://api.jsonbin.io/b/5fbd79a14f12502c21d82224", "https://api.jsonbin.io/b/5fbd794f4f12502c21d82210"]
# Loading training data
training_img = []
for x in urls_train:
    r = requests.get(x)
    data_url = r.json()

    for url in data_url:
        final_url = url+".jpg"
        training_img.append(resize(io.imread(final_url) / 255, (94, 94, 3)))
        train_imgs = np.array(training_img)
    train_imgs -= int(np.mean(train_imgs))
print("train_imgs.shape : ", train_imgs.shape)
print("train_images quantity : ", train_imgs.shape[0])

print('preparing training labels.........')
# training class
no_of_class = 2
i = 0
class_labels = []
for x in urls_train:
    r = requests.get(x)
    data_url = r.json()
    class_labels.extend([i] * len(data_url))
    i += 1
class_labels = np.array(class_labels)
training_labels = np.eye(no_of_class)[class_labels]
print("class_labels : ", class_labels)
print("training_labels : ", training_labels)


print('loading and preparing test data.........')
# test_covid = https://api.jsonbin.io/b/5fbd78e34f12502c21d821e8
# test_normal = https://api.jsonbin.io/b/5fbd787c4f12502c21d821c0
urls_test = ["https://api.jsonbin.io/b/5fbd78e34f12502c21d821e8", "https://api.jsonbin.io/b/5fbd787c4f12502c21d821c0"]
# preparing test images
test_img = []
for x in urls_test:
    r = requests.get(x)
    data_url = r.json()

    for url in data_url:
        final_url = url+".jpg"
        test_img.append(resize(io.imread(final_url) / 255, (94, 94, 3)))
        test_imgs = np.array(test_img)
    test_imgs -= int(np.mean(test_imgs))
print("test_imgs.shape : ", test_imgs.shape)
print("test_images quantity : ", test_imgs.shape[0])


print('preparing testing labels........')
# training class
no_of_class = 2
i = 0
class_labels = []
for x in urls_test:
    r = requests.get(x)
    data_url = r.json()
    class_labels.extend([i] * len(data_url))
    i += 1
class_labels = np.array(class_labels)
testing_labels = np.eye(no_of_class)[class_labels]
print("class_labels : ", class_labels)
print("testing_labels : ", testing_labels)


test_imgs_count = 6
size_of_batch = 2
epocs = 2

# Call CNN from network.py
model = CNN()

print('training model********')
#  train model
# Call train from network.py
model.train(train_imgs, training_labels, size_of_batch, epocs, 'cnn_model_weights.pkl')

print('testing model***********')
#  test model
# Call test from network.py
model.test(test_imgs, testing_labels, test_imgs_count)


