# Author: Tyler Sorg
# Course: CS445 Machine Learning
# Project: Multi-layer neural network with one hidden layer.

import math
import random
import sys

import numpy as np

# TODO: Determine these hyper-parameters from the command line
# Default definitions of hyper-parameters
_eta = 0.3  # learning rate 0 < _eta < 1
_alpha = 0.3  # momentum 0 < _alpha < 1
_lambda = 0  # weight decay if I get to it 0 < _lambda < 1

_number_hidden_nodes = 4  # not including hidden bias. The hidden bias is at index _number_hidden_nodes
_number_of_epochs = 5
_truncate_bool = False # If truncating the standardized inputs is desired.

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # Convenient global variable for classification.
np.seterr(all='raise')  # To catch possible ways float arithmetic could end badly for me.


def build_from_example_encodings(training_examples_preformatted):
    resulting_list = []
    for preformatted_example in training_examples_preformatted:
        # preformatted_example = 'T,2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8'
        converted_example = preformatted_example.strip().split(',')
        # converted_example = ['T', '2', '8', '3', '5', '1', '8', '13', '0', '6', '6', '10', '8', '0', '8', '0', '8']
        for i in range(1, len(converted_example)):
            converted_example[i] = float(converted_example[i])
            # ['T', 2.0, 8.0, 3.0, 5.0, 1.0, 8.0, 13.0, 0.0, 6.0, 6.0, 10.0, 8.0, 0.0, 8.0, 0.0, 8.0]
        resulting_list.append(converted_example)
    return resulting_list


def random_weights(size_of_next_layer):
    # Returns list of 17 random floats between -0.25 and 0.25.
    return [random.uniform(-1, 1) / 4 for x in range(size_of_next_layer)]


def initialize_weights1(_number_hidden_nodes):
    weights1 = dict()
    for i in range(17):  # maybe make it 16 and have a separate w1['bias1'][j]
        weights1[i] = random_weights(_number_hidden_nodes)  # weights1[i][0] is an edge to hidden_node_1
    # weights1['bias1'] = random_weights(_number_hidden_nodes)
    return weights1


def initialize_old_changes_in_weights1(_number_hidden_nodes):
    weights1 = dict()
    for i in range(17):
        weights1[i] = [0.0 for x in range(_number_hidden_nodes)]
    return weights1


def initialize_weights2(_number_hidden_nodes):
    weights2 = dict()
    for j in range(_number_hidden_nodes):
        weights2[j] = random_weights(26)  # j = hidden nodes indexed 0 through _number_hidden_nodes
    weights2[_number_hidden_nodes] = random_weights(26)
    return weights2


def initialize_old_changes_in_weights2(_number_hidden_nodes):
    weights2 = dict()
    for j in range(_number_hidden_nodes):
        weights2[j] = [0.0 for x in range(26)]
    weights2[_number_hidden_nodes] = [0.0 for x in range(26)]
    return weights2


# def sigmoid(z):
#     return 1.0 / (1.0 + np.exp(-z))

def sigmoid(x):  # From Neil at StackOverflow: http://stackoverflow.com/a/29863846
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))


# Get the "correct" answer to compare the network's vote to.
def compute_target_array(letter):
    # letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if letter in alphabet:
        # index = letters.index(letter)
        index = alphabet.index(letter)
        target_array = []
        for i in range(26):
            if index is i:
                target_array.append(0.9)
            else:
                target_array.append(0.1)
        return target_array
    else:
        print 'Target array letter not valid.'


# Not used. Don't want to get rid of it.
def sum_squared_error(target_array, output_activations):
    sumSquaredError = 0
    for k in range(len(output_activations)):
        squared_difference_k = (target_array[k] - output_activations[k]) ** 2
        sumSquaredError += squared_difference_k
    return float(sumSquaredError) / 2


# Calculate the array of activations for all the hidden nodes.
def compute_hidden_layer_activations(x_inputs_array, weights1):
    # given array of 17 inputs, compute h_j for all j in hidden layer.
    activations_h_j = list()
    for j in range(_number_hidden_nodes):  # for j = 0..n-1 where n = number of hidden units. weights
        dot_product_j = 0
        for i in range(len(x_inputs_array)):  # for i = 0..16
            dot_product_j += x_inputs_array[i] * weights1[i][j]
        activations_h_j.append(sigmoid(dot_product_j))  # h_j[j] is the activation of hidden node j.
    activations_h_j.append(1.0)
    return activations_h_j


# Calculate the array of activations for all the output nodes.
def compute_output_layer_activations(hidden_unit_activations, weights2):
    activations_o_k = [0.0 for k in range(len(weights2[0]))]
    for k in range(len(weights2[0])):  # for k = 0..25
        dot_product_k = 0
        for j in range(len(hidden_unit_activations)):  # for j = 0..n-1 AND j=n == hidden bias
            dot_product_k += hidden_unit_activations[j] * weights2[j][k]
        activations_o_k[k] = sigmoid(dot_product_k)  # h_j[j] is the activation of hidden node j.
    return activations_o_k


def compute_delta_k_array(target_array, output_activations):
    delta_k_array = list()
    for k in range(26):
        t_k = target_array[k]
        o_k = output_activations[k]
        delta_k_array.append(sigmoid_prime(o_k) * (t_k - o_k))
    return delta_k_array


def compute_delta_j_array(weights2, hidden_activations, delta_k_array):
    delta_j_array = list()
    for j in range(_number_hidden_nodes + 1):
        sum_wkj_times_dk = 0
        for k in range(26):
            sum_wkj_times_dk += weights2[j][k] * delta_k_array[k]
        h_j = hidden_activations[j]
        delta_j_array.append(sigmoid_prime(h_j) * sum_wkj_times_dk)
    return delta_j_array


def update_wkj(weights2, eta, delta_k_array, hidden_layer_activations, alpha, old_change_in_weights2):
    for j in range(_number_hidden_nodes + 1):
        h_j = hidden_layer_activations[j]
        for k in range(26):
            old_change_in_weight2_kj = old_change_in_weights2[j][k]  # Old weight
            d_k = delta_k_array[k]
            current_change_in_w_kj = eta * d_k * h_j + alpha * old_change_in_weight2_kj
            old_change_in_weights2[j][k] = current_change_in_w_kj  # store the current change in weight_kj
            weights2[j][k] += current_change_in_w_kj


def update_wji(weights1, eta, delta_j_array, inputs, alpha, old_change_in_weights1):
    for i in range(len(inputs)):
        x_i = inputs[i]
        for j in range(_number_hidden_nodes):
            old_change_in_weight_ji = old_change_in_weights1[i][j]  # Old weight change
            d_j = delta_j_array[j]
            current_change_in_weight_ji = eta * d_j * x_i + alpha * old_change_in_weight_ji
            old_change_in_weights1[i][j] = current_change_in_weight_ji
            weights1[i][j] += current_change_in_weight_ji


def weight_decay():
    pass


def standardize_examples(examples, training_mu, training_stddevs):
    for example in examples:
        inputs = example[1:]  # 16 inputs after the target class at index 0
        for i in range(len(inputs)):  # for indices 0-15 from the lists of means and standard deviations
            example_i_prime = float(example[i + 1] - training_mu[i]) / training_stddevs[i]
            example[i + 1] = example_i_prime


def standard_deviations_of_each_training_feature(examples, mu):
    squared_differences = [list() for i in range(16)]
    stddevs = [0.0 for i in range(16)]
    for example in examples:
        inputs = example[1:]
        for i in range(len(inputs)):
            # keep track of all the differences between x_i and mean(x_i) == mu[i]
            squared_difference = (abs(inputs[i] - mu[i])) ** 2
            squared_differences[i].append(squared_difference)
    means_of_squared_differences = [0.0 for i in range(16)]
    for i in range(len(squared_differences)):
        means_of_squared_differences[i] = np.mean(np.array(squared_differences[i])).tolist()
    # square the differences added together, get the mean
    # calculate the standard deviations for each of the 16 features
    for i in range(len(squared_differences)):
        stddev_i = np.sqrt(np.mean(np.array(squared_differences[i])))
        stddevs[i] = stddev_i

    return stddevs


def means_of_each_training_feature(examples):
    totals = [0.0 for i in range(16)]  # 16 totals for each feature i to be divided by the number of examples (10007)
    for example in examples:
        inputs = example[1:]
        for j in range(len(inputs)):
            totals[j] += inputs[j]
    # mu = [totals[i] / len(examples) for i in range(len(totals))]
    mu = [float(totals[i]) / len(examples) for i in range(len(totals))]
    return mu


def save_standardized_inputs(filename, inputs):
    # Create new file every time or overwrite old file? one and same?
    # f = open(filename + '.txt', 'w')
    f = open(filename, 'w')
    for row in inputs:
        f.write(str(row[0]))
        for col in row[1:]:
            truncated_float = "%.*f" % (6, col)
            f.write(',' + truncated_float)
        f.write('\n')
    f.close()


# ----------------------------------------------------------------------------------------------------------------------
# Definition of main routine.
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # Training algorithm:
    # Design: bias, inputs 1..6 in input layer
    # activations 0..n-1 + n == bias 2 in hidden layer, with the intention of preventing edges between inputs and bias2.
    # Weights between input and hidden layer: 17*_number_hidden_nodes
    # bias node in input layer = index 0

    # Initialize the weights between the input and hidden layer.
    weights1 = initialize_weights1(_number_hidden_nodes)
    old_weights1 = initialize_old_changes_in_weights1(_number_hidden_nodes)

    # Initialize the weights between hidden and output layer: 26*_number_hidden_nodes + 26 bias->output connections
    weights2 = initialize_weights2(_number_hidden_nodes)
    old_weights2 = initialize_old_changes_in_weights2(_number_hidden_nodes)

    # Open the training and testing files
    try:
        training_examples_preformatted = open('./mnn_training_data/training.txt')
        testing_examples_preformatted = open('./mnn_testing_data/testing.txt')
        # training_examples_preformatted = open('./mnn_training_data/training10.txt')
        # testing_examples_preformatted = open('./mnn_testing_data/testing10.txt')
    except IOError:
        print 'Could not find the specified files.'
        sys.exit(-1)

    # Pre-standardized training and test examples
    examples = build_from_example_encodings(training_examples_preformatted)
    # for example in examples:
    #     if len(example) < 15:
    #         print 'found an inappropriately small training example'
    #         sys.exit(-1)

    tests = build_from_example_encodings(testing_examples_preformatted)
    # for example in tests:
    #     if len(example) < 15:
    #         print 'found an inappropriately small test example'
    #         sys.exit(-1)

    # Mean values of each feature from training
    training_mu = means_of_each_training_feature(examples)
    # for i in training_mu:
    #     if math.isnan(i):
    #         print 'found a NaN mean'
    #         sys.exit(-1)

    # Standard deviations of each feature from training
    training_stddevs = standard_deviations_of_each_training_feature(examples, training_mu)
    # for i in training_stddevs:
    #     if math.isnan(i):
    #         print 'found a NaN standard deviation'
    #         sys.exit(-1)

    # Standardize the training examples
    standardize_examples(examples, training_mu, training_stddevs)
    # for example in examples:
    #     if len(example) < 17:
    #         print 'A standardized training example lost data'
    #         sys.exit(-1)

    # Standardize the test examples
    standardize_examples(tests, training_mu, training_stddevs)
    # for example in tests:
    #     if len(example) < 17:
    #         print 'A standardized test example lost data'
    #         sys.exit(-1)

    if _truncate_bool is True:
        save_standardized_inputs('standardized_training.txt', examples)
        print 'full-precision examples: ', examples[0]
        truncated_examples_file = open('standardized_training.txt')
        truncated_examples = build_from_example_encodings(truncated_examples_file)
        print 'truncated examples: ', truncated_examples[0]

        save_standardized_inputs('standardized_testing.txt', tests)

        print 'full-precision tests: ', tests[0]
        truncated_tests_file = open('standardized_testing.txt')
        truncated_tests = build_from_example_encodings(truncated_tests_file)
        print 'truncated tests: ', truncated_tests[0]

        examples = truncated_examples
        tests = truncated_tests

    print 'Successfully pre-processed data. Beginning training.'

    # Records the counts of correct classifications in both training and test sets for each epoch.
    accuracies_per_epoch = dict()

    # ==================================================================================================================
    # Begin training and testing the network.
    # ==================================================================================================================
    for epoch_number in range(_number_of_epochs):
        print 'Beginning epoch %d' % (epoch_number + 1)
        correct_predictions_training = 0
        correct_predictions_testing = 0

        # ==============================================================================================================
        # Train the network on the training examples every epoch
        # ==============================================================================================================
        for example in examples:
            # example = ['A', ...16 standardized features for input represented as strings...]

            # ==========================================================================================================
            # (1) Propagate the inputs forward
            # ==========================================================================================================
            # input x_array to the network

            # Get inputs array including the bias's input x_0 == 1
            inputs = list()
            inputs.append(1.0)  # Bias is index 0
            inputs.extend(example[1:])  # [x_0 = 1.0, ...16 standardized features...]

            # Get target vector
            target_k_array = compute_target_array(example[0])

            # compute activation h_j for each j
            activations_h_j = compute_hidden_layer_activations(inputs, weights1)

            for h_j in activations_h_j:
                if h_j < 0:
                    print 'Negative activation in hidden layer!'
                    sys.exit(-2)

            # compute the activation o_k for each k
            activations_o_k = compute_output_layer_activations(activations_h_j, weights2)
            for o_k in activations_o_k:
                if o_k < 0:
                    print 'Negative activation in output layer!'
                    sys.exit(-2)

            which_letter_classified = activations_o_k.index(max(activations_o_k))
            if example[0] is alphabet[which_letter_classified]: correct_predictions_training += 1
            # ----------------------------------------------------------------------------------------------------------

            # ==========================================================================================================
            # (2) Calculate the error terms
            # ==========================================================================================================
            # calculate delta_k and delta_j for each k and j
            delta_k_array = compute_delta_k_array(target_k_array, activations_o_k)
            delta_j_array = compute_delta_j_array(weights2, activations_h_j, delta_k_array)
            # ----------------------------------------------------------------------------------------------------------

            # ==========================================================================================================
            # (3) Update weights
            # ==========================================================================================================
            # for each weight w_kj from hidden->output layers
            # print "Updating weights between hidden and output layer..."
            update_wkj(weights2, _eta, delta_k_array, activations_h_j, _alpha, old_weights2)
            # for each weight w_ji from input->hidden layers
            # print "Updating weights between input and hidden layer..."
            update_wji(weights1, _eta, delta_j_array, inputs, _alpha, old_weights1)
            # ----------------------------------------------------------------------------------------------------------

        # ==============================================================================================================
        # Test the network on the test examples every epoch
        # ==============================================================================================================
        for test in tests:
            # ==========================================================================================================
            # (1) Propagate the inputs forward
            # ==========================================================================================================
            # Get inputs array including the bias's input x_0 == 1
            test_inputs = list()
            test_inputs.append(1.0)
            test_inputs.extend(test[1:])  # [x_0 = 1.0, ...16 standardized features...]
            # Get target vector
            target_k_array = compute_target_array(test[0])

            # compute activation h_j for each j
            activations_h_j = compute_hidden_layer_activations(test_inputs, weights1)

            # compute the activation o_k for each k
            activations_o_k = compute_output_layer_activations(activations_h_j, weights2)

            which_letter_classified = activations_o_k.index(max(activations_o_k))
            if test[0] is alphabet[which_letter_classified]: correct_predictions_testing += 1

        # --------------------------------------------------------------------------------------------------------------
        # Recording accuracies after iterating over training and test sets.
        # --------------------------------------------------------------------------------------------------------------
        accuracies_per_epoch[epoch_number] = [correct_predictions_training, correct_predictions_testing]
        # --------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------------------
    # Print the accuracies of the network on both sets for each epoch.
    # ------------------------------------------------------------------------------------------------------------------
    epoch_count = 1
    experiment_data_pathname = './ExperimentData/'
    experiment_data_filename = 'learning_rate=%s_momentum=%s_hidden_nodes=%d_epochs=%d' % (
        str(round(_eta,3)), str(round(_alpha,3)), _number_hidden_nodes, _number_of_epochs
    )
    experiment_data = open(experiment_data_pathname + experiment_data_filename + '.csv', 'w')
    experiment_data.write('Epoch, TrainingAccuracy, TestAccuracy\n')
    for epoch in range(len(accuracies_per_epoch)):
        training_acc = float(accuracies_per_epoch[epoch][0]) / len(examples)
        testing_acc = float(accuracies_per_epoch[epoch][1]) / len(tests)
        print 'Accuracies of epoch %d: Training: %f, Testing: %f' % (epoch_count, training_acc, testing_acc)
        line = '%d,%f,%f\n' % (epoch_count, round(training_acc, 6), round(testing_acc, 6))
        experiment_data.write(line)
        epoch_count += 1
    # ------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------

main()
