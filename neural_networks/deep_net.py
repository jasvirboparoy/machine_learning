import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import input_data

'''
input > weight > hidden L1 (activation function) > weights > hidden L2 
(Activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)
optimisation function (optimiser) > minimise cost (AdamOptimizer ... SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 10 classes, 0-9
'''
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# height x width
x = tf.compat.v1.placeholder('float', [None, 784])
y = tf.compat.v1.placeholder('float')

def neural_network_model(data):
    
    # (input_data * weights) + biases

    hidden_1_layer = {'weights': tf.Variable(tf.random.normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random.normal(n_nodes_hl1))}



