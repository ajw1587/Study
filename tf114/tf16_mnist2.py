import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])
w = tf.Variable(tf.random_normal([784, 10]), name = 'weight')
b = tf.Variable(tf.random_normal([10]), name = 'bias')

# 2. 모델구성
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

# 4. train
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(2001):
        _, cost_val = sess.run([optimizer, cost], feed_dict = {x:x_train, y:y_train})

        if epoch % 200 == 0:
            print(epoch, '\t', cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {x:x_test, y:y_test})
    print('예측값:\n', h, '\n원래값:\n', c, '\nAccuracy: ', a)