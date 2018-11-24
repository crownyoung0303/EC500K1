################################################################
# Inputs: arr1 and arr2 have shape [batch_size, hidden_sizes[-1]]
# Output: return tensor of shape [batch_size, ], the cosine 
#         similarity betwwen arr1 
# Hint: use tf.l2_normalize, tf.mul, tf.reduce_sum
#################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def mlp(x=tf.placeholder(tf.float32, [None, 784]), hidden_sizes= [64, 10], activation_fn= tf.nn.relu):
  h_in = x
  w= []
  b= []
  io_sizes=[784] + hidden_sizes

  for i in range(len(io_sizes)-1):
    with tf.variable_scope("layer"+str(i)):
            weights= tf.get_variable("weights", [io_sizes[i], io_sizes[i+1]], initializer=tf.random_normal_initializer(stddev= 0.1))
            biases = tf.get_variable("biases", [io_sizes[i+1]], initializer=tf.constant_initializer(0.1))
            h_out=activation_fn(tf.matmul(h_in, weights) + biases) 
            h_in= h_out
  return h_out


def cosine_similarity(arr1, arr2):
    logits = arr1 *  arr2
    y_logits = tf.reduce_sum(logits, 1)/(tf.norm(arr1, 2, 1) * tf.norm(arr2, 2, 1))
    return y_logits
    ###################################
    ####     PUT YOUR CODE HERE    ####
    ###################################
def loss_with_spring(arr1, arr2, y_= tf.placeholder(tf.float32, [None])):
    margin = 5.0
    labels_t = y_
    labels_f = tf.subtract(1.0, y_, name="1-yi")          # labels_ = !labels;
    eucd2 = tf.pow(tf.subtract(arr1, arr2), 2)
    eucd2 = tf.reduce_sum(eucd2, 1)
    eucd = tf.sqrt(eucd2+1e-6, name="eucd")
    C = tf.constant(margin, name="C")
    pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
    neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
    losses = tf.add(pos, neg, name="losses")
    loss = tf.reduce_mean(losses, name="loss")
    return loss

    
#################################################################
# Inputs: mlp_args is a dictionary of arguments to the mlp() 
#         function. 
#         Example: mlp_args = {'hidden_sizes':[64, 64, 32]}
#################################################################
def build_model(mlp_args):
    with tf.Graph().as_default() as g:
        with tf.device("/gpu:0"):
            x1 = tf.placeholder(tf.float32, [None, 784])
            x2 = tf.placeholder(tf.float32, [None, 784])
            y = tf.placeholder(tf.float32, [None])

            with tf.variable_scope("siamese") as var_scope:
                x_repr1 = mlp(x1, **mlp_args)  # hidden representation of x1
                var_scope.reuse_variables()    # weight sharing! 
                x_repr2 = mlp(x2, **mlp_args)  # hidden representation of x2
                logits = cosine_similarity(x_repr1, x_repr2)  # similarity
                y_prob = tf.exp(logits) / (1 + tf.exp(logits))
                y_pred = tf.cast(tf.equal(tf.sign(logits), 1), tf.float32)
                loss = -tf.reduce_mean(y * tf.log(y_prob) + (1 - y) * tf.log(1 - y_prob))  
                y_logits= tf.cast(tf.equal(tf.argmax(x_repr1, 1), tf.argmax(x_repr2, 1)), tf.float32), 
                #accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_logits), tf.float32))
                accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_pred), tf.float32))
            ###################################
            ####     PUT YOUR CODE HERE    ####
            ###################################
            
            # define scalar: loss 
            # define vector: y_prob as sigmoid(cosine_similarity)
            # define vector: y_pred as sign(cosine_similarity)
            # define scalar: accuracy as the fraction of correct predictions
            
    return {'graph': g, 'inputs': [x1, x2, y], 'pred': y_pred, 'logits': logits,
            'prob': y_prob, 'loss': loss, 'accuracy': accuracy}
# data preparation
def mnist_siamese_dataset_iterator(batch_size, dataset_name):
    assert dataset_name in ['train', 'test']
    assert batch_size > 0 or batch_size == -1 # -1 for entire dataset
    mnist = input_data.read_data_sets('./datasets/mnist/', one_hot=True)
    dataset = getattr(mnist, dataset_name)
    
    while True:
        if batch_size > 0:
            X1, y1 = dataset.next_batch(batch_size)
            X2, y2 = dataset.next_batch(batch_size)
            y = np.argmax(y1, axis=1) == np.argmax(y2, axis=1)
            yield X1, X2, y
        else:
            X1 = dataset.images
            idx = np.arange(len(X1))
            np.random.shuffle(idx)
            X2 = X1[idx]
            y1 = dataset.labels
            y2 = y1[idx]
            y = np.argmax(y1, axis=1) == np.argmax(y2, axis=1)
            yield X1, X2, y
try:
    from itertools import izip as zip
except ImportError:
    print('This is Python 3')

def run_training(model_dict, train_data_iterator, test_full_iter,  
                 train_full_iter, n_iter=20000, print_every=100):
    with model_dict['graph'].as_default():
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
        train_op = optimizer.minimize(model_dict['loss'])
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for iter_i, data_batch in zip(range(n_iter), train_data_iterator):
                batch_feed_dict = dict(zip(model_dict['inputs'], data_batch))
                sess.run(train_op, feed_dict=batch_feed_dict)
                if iter_i % print_every == 0:
                    print_zip_iter = zip([test_full_iter, train_full_iter], ['test', 'train'])
                    for data_iterator, data_name in print_zip_iter:
                        test_batch = next(data_iterator)
                        batch_feed_dict = dict(zip(model_dict['inputs'], test_batch))
                        to_compute = [model_dict['accuracy'], model_dict['loss']]
                        acc_value, loss_val = sess.run(to_compute, batch_feed_dict)
                        fmt = (iter_i, acc_value, loss_val)
                        print(data_name, 'iteration %d\t accuracy: %.3f, loss: %.3f'%fmt)

train_data_iterator = mnist_siamese_dataset_iterator(100, 'train')
test_full_iter = mnist_siamese_dataset_iterator(-1, 'test')
train_full_iter = mnist_siamese_dataset_iterator(-1, 'train')

mlp_args = {'hidden_sizes':[64, 64, 32]}
model = build_model(mlp_args)
run_training(model, train_data_iterator, test_full_iter, train_full_iter)
