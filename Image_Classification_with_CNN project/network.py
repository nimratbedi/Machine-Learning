import pickle
from model.layer import *
from model.loss import *
import sys


# class for creating the layer architecture
class CNN:
    def __init__(self):
        learning_rate = 0.01
        self.layers = []
        print("In CNN class")
        # L: 0 
        # Call convolution from layer.py
        print("Network.py Convolution 1-----------------")
        self.layers.append(Convolution(in_channel=3, num_of_filter=16, size_of_kernel=3, pad=0, stride=1,lr=learning_rate, name_of_layer='conv_1'))
        
        # Ly: 1
        print("Network.py Relu 1-----------------")
        self.layers.append(ReLu())
        # Ly: 2
        print("Network.py Maxpool 1-----------------")
        self.layers.append(Maxpool(size_of_pool=2, stride=2, name_of_layer='maxpool_1'))
        # Ly: 3
        print("Network.py Convolution 2-----------------")
        self.layers.append(Convolution(in_channel=3, num_of_filter=32, size_of_kernel=3, pad=0, stride=1, lr=learning_rate, name_of_layer='conv_2'))
        # Ly: 4
        print("Network.py Relu 2-----------------")
        self.layers.append(ReLu())
        # Ly: 5
        print("Network.py Maxpool 2-----------------")
        self.layers.append(Maxpool(size_of_pool=2, stride=2, name_of_layer='maxpool_2'))
        # Ly: 3
        print("Network.py Convolution 3-----------------")
        self.layers.append(Convolution(in_channel=3, num_of_filter=64, size_of_kernel=3, pad=0, stride=1, lr=learning_rate, name_of_layer='conv_2'))
        # Ly: 4
        print("Network.py Relu 3-----------------")
        self.layers.append(ReLu())
        # Ly: 5
        print("Network.py Maxpool 3-----------------")
        self.layers.append(Maxpool(size_of_pool=2, stride=2, name_of_layer='maxpool_2'))
        # Ly: 6
        print("Network.py Convolution 4-----------------")
        self.layers.append(Convolution(in_channel=3, num_of_filter=128, size_of_kernel=3, pad=0, stride=1, lr=learning_rate, name_of_layer='conv_2'))
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
        print('Number of test data :{0:d},  test accuracy :{1:.2f}'.format(test_input_size, float(accuracy_total) / float(test_input_size)))

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
                    loss += cross_entropy_loss(output, training_label)
                    if np.argmax(output) == np.argmax(training_label):
                        accuracy += 1
                        accuracy_total += 1
                    Dy = output
                    print("Network.py Dy = output & Dy.shape=",Dy.shape)
                    for lays in range(self.layers_num - 1, -1, -1):
                        Dout = self.layers[lays].bck_pass(Dy)
                        Dy = Dout

                # Calculating loss and accuracy of model.
                loss /= size_of_batch
                batch_accuracy = float(accuracy) / float(size_of_batch)
                training_accuracy = float(accuracy_total) / float((index_of_batch + size_of_batch) * (ep + 1))
                print('Epoch: {0:d}/{1:d}, Iter:{2:d}, loss: {3:.2f}, BAcc: {4:.2f}, TAcc: {5:.2f}'.format(
                    ep+1, epocs, index_of_batch + size_of_batch, loss, batch_accuracy, training_accuracy))
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
        print('Number of test data :{0:d}, test accuracy :{1:.2f}'.format(test_input_size, float(accuracy_total) / float(test_input_size)))
