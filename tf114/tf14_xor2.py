import tensorflow as tf
import numpy as np
tf.set_random_seed(77)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)


x = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])

w1 = tf.Variable(tf.random_normal([2, 100], name = 'weight'))
b1 = tf.Variable(tf.random_normal([100], name = 'bias1'))
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([100, 7]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([7]), name = 'bias2')
layer2 = tf.matmul(layer1, w2) + b2

w3 = tf.Variable(tf.random_normal([7, 1]), name = 'weight2')
b3 = tf.Variable(tf.random_normal([1]), name = 'bias3')
hypothesis = tf.sigmoid(tf.matmul(layer2, w3) + b3)

#####################################################

cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))
#####################################################

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for step in range(5001):
        cost_val, _ = sess.run([cost, train], feed_dict = {x:x_data, y:y_data})

        if step % 30 == 0:
            print(step, cost_val)
    
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {x:x_data, y:y_data})
    print('예측값:\n', h, '\n0 or 1:\n', c, '\nAccuracy: ', a)