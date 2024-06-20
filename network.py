# from functions import sigmoid, random
from functools import reduce

import numpy as np
from scipy.special import softmax

# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D


class PairedLayer:
    def __init__(self, amount):
        self.amount = amount


class PairedGridLayer:
    def __init__(self, width, height):
        self.width = width
        self.height = height


def random(*shape):
    """Random numbers between -1 and +1"""
    return np.random.uniform(-1, 1, shape)


class Network:
    def __init__(self):
        """
        ```
        o          o  o  o  o  o      o
         o          o  o  o  o  o      o

        o          o  o  o  o  o      o
         o          o  o  o  o  o      o

        o          o  o  o  o  o      o
         o          o  o  o  o  o      o

        """
        self.layers = [
            PairedLayer(4 * 2 + 1),
            PairedGridLayer(5, 4),
            PairedLayer(4 * 2 + 1),
        ]

        self.randomize_weights_biases()

    def randomize_weights_biases(self):
        self.biases = []
        for layer in self.layers[1:]:  # No biases in the input layer, hence the '[1:]'
            if isinstance(layer, PairedLayer):
                self.biases.append(random(layer.amount, 2))
            elif isinstance(layer, PairedGridLayer):
                self.biases.append(random(layer.height, layer.width, 2))

        self.weights = []
        for layer1, layer2 in zip(self.layers[:-1], self.layers[1:]):
            if isinstance(layer1, PairedLayer) and isinstance(layer2, PairedGridLayer):
                self.weights.append(
                    random(layer1.amount, layer2.height, layer2.width, 2)
                )
                # pair1 = []
                # # Color
                # print(layer1.amount)
                # for _ in range(layer1.amount):
                #     pair1.append(random(layer2.height, layer2.width))
                # pair2 = []
                # for _ in range(layer1.amount):
                #     pair2.append(random(layer2.height, layer2.width))

                # self.weights.append(list(zip(pair1, pair2)))
            elif isinstance(layer1, PairedGridLayer) and isinstance(
                layer2, PairedLayer
            ):
                self.weights.append(
                    random(layer2.amount, layer1.height, layer1.width, 2)
                )
                # pair1 = []
                # # Color
                # for _ in range(layer2.amount):
                #     pair1.append(random(layer1.height, layer1.width))
                # pair2 = []
                # for _ in range(layer2.amount):
                #     pair2.append(random(layer1.height, layer1.width))

                # self.weights.append(list(zip(pair1, pair2)))

    def calculate_output(self, input):
        def compute_layer(input, layer):
            bias, weight = layer

            for i, pairs in enumerate(n.weights):
                for layer1, layer2 in pairs:
                    for j, weights in enumerate(layer1):
                        for k, weight in enumerate(weights):
                            print(input[i][0])
                            print(self.biases[i][j][0])
                            print(weight)
                    for weights in layer2:
                        for weight in weights:
                            print(weight)

            # weighted_inputs = np.dot(weight, input) + bias

            # return weighted_inputs

        result = reduce(compute_layer, zip(self.biases, self.weights), input)

        return result


from pprint import pprint

n = Network()
# pprint(n.biases)
# pprint("WEIGHTS")
# pprint(n.weights)

input = [[3, 5] for _ in range(9)]


inputs1 = []
inputs2 = []
for i, layer in enumerate(n.weights):
    # print(i) # layer index
    # print(n.layers[1:][i], n.biases[i])

    if isinstance(n.layers[1:][i], PairedGridLayer):
        for input_index, input_connections in enumerate(layer):
            for height, row in enumerate(input_connections):
                for width, node_pair in enumerate(row):
                    activation = node_pair[0] * input[i][0] + n.biases[i][height][width][0]
                    print(activation)

                # print(x)

    # for l, (layer1, layer2) in enumerate(pairs):
    #     #print(l) # node index
    #     total1 = 0
    #     for j, weights in enumerate(layer1):
    #         #print(j) # height 4
    #         for k, weight in enumerate(weights):
    #             #print(k) # width 5
    #             total1 += weight * input[i][0] + n.biases[i][j][0]
    #             #print(result)
    #     print(total1)

    # for weights in layer2:
    #     for weight in weights:
    #         #print(weight)
    #         pass


# for pairs in n.weights:
#     for layer1, layer2 in pairs:
#         for height in layer1:
#             for width in height:
#                 print(width)
#         for height in layer2:
#             for width in height:
#                 print(width)


# self.biases = [
#     random(y) for y in layers[1:]
# ]  # No biases in the input layer, hence the '[1:]'

# self.weights: list[np.ndarray] = [
#     random(y, x) for x, y in zip(layers[:-1], layers[1:])
# ]

# def activation_function(self, output):
#     return output

# def activation_dir(self, output):
#     return output

# def loss_dir(self, prediction, label):
#     return 2 * (prediction - label)

# def calculate_output(self, input):
#     self.activations = []
#     self.weighted_inputs = []

#     def compute_layer(input, layer):
#         bias, weight = layer

#         weighted_inputs = np.dot(weight, input) + bias
#         activation = self.activation_function(weighted_inputs)

#         self.weighted_inputs.append(weighted_inputs)
#         self.activations.append(activation)

#         return activation

#     result = reduce(compute_layer, zip(self.biases, self.weights), input)

#     return result

# def calculate_outputs2(self, inputs):
#     self.activations = []
#     self.weighted_inputs = []

#     def cal(input):
#         activations = []
#         weighted_inputs = []

#         def compute_layer(input, layer):
#             bias, weight = layer

#             weighted_inputs1 = np.dot(weight, input) + bias
#             activation = self.activation_function(weighted_inputs)

#             weighted_inputs.append(weighted_inputs1)
#             activations.append(activation)

#             return activation

#         result = reduce(compute_layer, zip(self.biases, self.weights), input)

#         self.weighted_inputs.append(weighted_inputs)
#         self.activations.append(activations)

#         return result

#     return np.apply_along_axis(cal, -1, inputs)

# def calculate_outputs_graph(self, inputs):
#     return np.apply_along_axis(
#         lambda x: x[0] < x[1], -1, self.calculate_outputs(inputs)
#     )

# def calculate_outputs(self, inputs):
#     return np.apply_along_axis(self.calculate_output, -1, inputs)

# def loss(self, prediction, label):
#     return (prediction - label) ** 2  # positive (emphasize differences)

# def get_loss(self, inputs, labels):
#     outputs = self.calculate_outputs(inputs)

#     return np.average(self.loss(outputs, labels))

# def cal_learn(self, input, labels, learn_rate=0.1):
#     outputs = self.calculate_outputs2(input)
#     node_values = self.loss_dir(self.activations, labels) * self.activation_dir(
#         self.weighted_inputs
#     )
#     gradientB = node_values
#     gradientW = np.dot(outputs, node_values)

#     for i in range(self.num_layers - 1):
#         self.biases[i] -= gradientB[i] * learn_rate
#         self.weights[i] -= gradientW[i] * learn_rate

#     loss = self.get_loss(input, labels)
#     return loss

# def learn(self, input, labels, learn_rate=0.1):
#     increment = 0.000001
#     original_loss = self.get_loss(input, labels)

#     for layer_i in range(self.num_layers - 1):
#         layer_weights = self.weights[layer_i]
#         for i in range(len(layer_weights)):
#             weights = layer_weights[i]
#             for j in range(len(weights)):
#                 weights[j] += increment
#                 loss = self.get_loss(input, labels)
#                 change_in_loss = loss - original_loss
#                 weights[j] -= increment

#                 self.loss_gradient_weights[layer_i][i][j] = (
#                     change_in_loss / increment
#                 )

#         biases = self.biases[layer_i]
#         for i in range(len(biases)):
#             biases[i] += increment
#             change_in_loss = self.get_loss(input, labels) - original_loss
#             biases[i] -= increment
#             self.loss_gradient_biases[layer_i][i] = change_in_loss / increment

#     for i in range(self.num_layers - 1):
#         self.biases[i] -= self.loss_gradient_biases[i] * learn_rate
#         self.weights[i] -= self.loss_gradient_weights[i] * learn_rate

#     loss = self.get_loss(input, labels)
#     return loss

# def calculate_losss_of_weights(
#     self,
#     input,
#     output,
#     points,
#     range_,
# ):
#     variable1s = np.linspace(-range_, range_, points)
#     variable2s = np.linspace(-range_, range_, points)

#     # x, y, z = (
#     #     np.empty(points**2),
#     #     np.empty(points**2),
#     #     np.empty((points, points)),
#     # )
#     x, y, z = (
#         np.empty((points, points)),
#         np.empty((points, points)),
#         np.empty((points, points)),
#     )
#     for i, variable1 in enumerate(variable1s):
#         for j, variable2 in enumerate(variable2s):
#             parameter, layer_index, *index = self.variable1
#             parameter[layer_index][index] = variable1
#             parameter, layer_index, *index = self.variable2
#             parameter[layer_index][index] = variable2
#             loss = self.get_loss(input, output)
#             # index = i * points + j
#             x[i, j] = variable1
#             y[i, j] = variable2
#             z[i, j] = loss
#     return x, y, z

# def null_biases(self):
#     """nullifies the biases to 0"""
#     self.biases = np.zeros_like(self.biases)


# n = Network([2, 3, 2])
# n.activation_function = sigmoid
# inputs = np.array([[0.5, 0.5], [1, 1]])
# output = np.array([[1, 0], [0, 1]])
# n.plot_2d_gradient(inputs, output, 20)
# n.plot_3d_gradient(inputs, output, 50)

# # Gradient Descent Example
# network = Network([1, 1, 1])  #  o-o-o network
# input = np.array([0.5])
# expected_result = np.array([1])
# network.null_biases()

# network.weights[0][0][0] = -2
# network.weights[1][0][0] = 1.9

# pathX = []
# pathY = []
# pathZ = []
# for i in range(100):
#     network.learn(input, expected_result, 0.1)
#     network.null_biases()
#     pathX.append(network.weights[0][0][0])
#     pathY.append(network.weights[1][0][0])
#     pathZ.append(network.get_loss(input, expected_result))

# print(pathX, pathY)

# print(network.get_loss(input, expected_result))
# network.plot_3d_gradient(input, expected_result, pathX, pathY, pathZ, value=2)


# input = np.array([0.5])
# output = np.array([0.5])
# # x = n.calculate_output(input)
# # print(x)
# # print(n.get_loss(x, output))
# for i in range(1):
#     print(n.learn(input, output))
# print(n.get_loss(x, output))
# n.null_biases()
# n.plot_2d_gradient(input, output)
