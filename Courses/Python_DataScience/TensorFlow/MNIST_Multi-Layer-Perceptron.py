import tensorflow as tf

# mnist comprises thousands of images containing handwritten digits

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

#print(type(mnist))

sample = mnist.train.images[2].reshape(28,28)

import matplotlib.pyplot as plt

#plt.imshow(sample,cmap='Greys')
#plt.show()

# how quickly is the cost function adjusted
learning_rate = 0.001

# number of training cycles
training_epochs = 30

# size of the batches of training data
batch_size = 100

# expect 10 possibles outcomes (because we have 0-9 digits)
n_classes = 10

# number of training samples
n_sample = mnist.train.num_examples

# every image is 28x28 in size, so 784. but input is a flattened version of that array
n_input = 784

# how many neurons in hidden layer
n_hidden_1 = 256
n_hidden_2 = 256

def multilayer_perceptron(x, weights, biases):
    """
    :param x: Placeholder for Data Input
    :param weigths: Dict of weights
    :param biases: Dict of bias values
    :return:

    Perceptron steps:

    1. receive inputs
    2. weight inputs
    3. sum inputs
    4. generate output

    """

    # first hidden layer with RELU Activation (neurons needs activation function, which is RELU?)

    # matrix multiplication (x * weights + biases)
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    # relu(x * weights + biases) -> f(x) = max(0,x)
    layer_1 = tf.nn.relu(layer_1)

    # second hidden layer

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Last Output Layer
    out_layer = tf.matmul(layer_2, weights['out'] + biases['out'])

    return out_layer

# want weights to be randomly initialized (tf.random_normal outputs values from a normal distribution)
weights = {
        # n_input = rows, n_hidden_1 = columns
        'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))

}

# input
x = tf.placeholder('float',[None, n_input])

# output
y = tf.placeholder('float', [None, n_classes])

pred = multilayer_perceptron(x,weights,biases)

# cost and optimization function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# training the model

sess = tf.InteractiveSession()

init = tf.initialize_all_variables()

sess.run(init)

# 15 loops
for epoch in range(training_epochs):

    # cost
    avg_cost = 0.0

    total_batch = int(n_sample/batch_size)

    for i in range(total_batch):

        batch_x, batch_y = mnist.train.next_batch(batch_size)

        _,c = sess.run([optimizer,cost],feed_dict={x:batch_x, y:batch_y})

        avg_cost += c/total_batch


    print("Epoch: {} cost{:.4f}".format(epoch+1,avg_cost))

print("Model has completed {} Epochs of training ".format(training_epochs))

# model evaluation

correct_predictions = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

correct_predictions = tf.cast(correct_predictions,'float')

accuracy = tf.reduce_mean(correct_predictions)

acc = accuracy.eval({x:mnist.test.images, y:mnist.test.labels})

print(acc)