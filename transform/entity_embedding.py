import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.base import TransformerMixin
from tensorflow.python.framework import ops


class ColumnEmbedder(TransformerMixin):
    def __init__(self, embedding_column, embedding_size, minibatch_size=1024, learning_rate=0.0001, num_epochs=1500, l2_beta=0):
        self.embedding_column = embedding_column
        self.embedding_size = embedding_size
        self.minibatch_size = minibatch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.l2_beta = l2_beta
        self.nn_parameters = None

    def transform(self, X_df, *_):
        embedding = compute_embedding(pd.get_dummies(X_df[self.embedding_column], prefix=self.embedding_column),
                                             self.nn_parameters)
        embedding_df = pd.DataFrame(
            embedding,
            columns=['{}_embedding_dim_{}'.format(self.embedding_column, i) for i in range(0, embedding.shape[1])],
            index=X_df.index
        )

        return pd.concat([X_df, embedding_df], axis=1).drop(self.embedding_column, axis=1)

    def fit(self, X_df, y, *_):
        column_to_encode_one_hot = pd.get_dummies(X_df[self.embedding_column], prefix=self.embedding_column)
        m = len(X_df)
        X_nn = np.transpose(column_to_encode_one_hot.values)
        y_nn = y.values.reshape(1, m)

        self.nn_parameters = model(X_nn, y_nn, self.embedding_size, self.learning_rate, self.num_epochs, self.minibatch_size, self.l2_beta)
        return self


def initialize_parameters(num_categories, embedding_size):
    tf.set_random_seed(1)
    embedding = tf.get_variable("embedding", [embedding_size, num_categories],
                                initializer=tf.contrib.layers.xavier_initializer(seed=1))
    embedding_bias = tf.get_variable("embedding_bias", [embedding_size, 1], initializer=tf.zeros_initializer())
    W1 = tf.get_variable("W1", [1, embedding_size], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [1, 1], initializer=tf.zeros_initializer())

    parameters = {"embedding": embedding,
                  "embedding_bias": embedding_bias,
                  "W1": W1,
                  "b1": b1,
                  }

    return parameters


def forward_propagation(X, parameters):
    Z_embedded = tf.add(tf.matmul(parameters['embedding'], X), parameters['embedding_bias'])
    A_embedded = tf.nn.relu(Z_embedded)
    Z1 = tf.add(tf.matmul(parameters['W1'], A_embedded), parameters['b1'])

    return Z1


def compute_embedding(X, parameters):
    Z_embedded = np.dot(X, np.transpose(parameters['embedding'])) + np.transpose(parameters['embedding_bias'])
    return Z_embedded


def compute_cost(last_linear_Z, Y, parameters, beta):
    embedding = tf.reshape(parameters['embedding'], [-1])
    W1 = tf.reshape(parameters['W1'], [-1])

    logits = tf.transpose(last_linear_Z)
    labels = tf.transpose(Y)

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    regularizer = tf.nn.l2_loss(tf.concat([W1, embedding], 0))
    loss = tf.reduce_mean(cost) + beta * regularizer

    return loss


def random_mini_batches(X, y, minibatch_size):
    m_train = X.shape[1]
    n_minibatches = m_train // minibatch_size
    last_batch_size = m_train - n_minibatches * minibatch_size
    mini_batches_list = [
        (X[:, i * minibatch_size:(i + 1) * minibatch_size],
         y[:, i * minibatch_size:(i + 1) * minibatch_size])
        for i in range(0, n_minibatches)
    ]
    mini_batches_list.append(
        (
            X[:, n_minibatches * minibatch_size:n_minibatches * minibatch_size + last_batch_size],
            y[:, n_minibatches * minibatch_size:n_minibatches * minibatch_size + last_batch_size]
        )
    )
    return mini_batches_list


def model(X, y, embedding_size, learning_rate, num_epochs, minibatch_size, beta):

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    n_categories, m = X.shape
    X_tensor = tf.placeholder(tf.float32, shape=(n_categories, None), name="X")
    y_tensor = tf.placeholder(tf.float32, shape=(1, None), name="Y")
    parameters = initialize_parameters(n_categories, embedding_size)
    last_linear_Z = forward_propagation(X_tensor, parameters)
    cost = compute_cost(last_linear_Z, y_tensor, parameters, beta)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=6)) as sess:

        sess.run(init)
        for epoch in range(num_epochs):

            minibatches = random_mini_batches(X, y, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X_tensor: minibatch_X, y_tensor: minibatch_Y})


        parameters = sess.run(parameters)
        return parameters